"""
    TimeStep!(sim::MarineSimulation)

The primary orchestration loop for a single model iteration. 
Handles calendar progression, environment updates, agent biology, and I/O.
Includes the application of instantaneous natural mortality and school consolidation.
"""

# Runtime cadence configuration, set in `setup_and_run_model`. Kept as a small
# mutable global so the MarineModel struct layout does not have to change.
#   merge_every      : run agent consolidation every N steps (1 = every step)
#   checkpoint_every : write a restart checkpoint every N steps (0 = never)
const RUNTIME_CONFIG = Dict{Symbol, Int}(:merge_every => 1, :checkpoint_every => 0)

# Post-spinup "real" calendar start, configurable via params.csv (start_year /
# optional start_month, start_day). Set in setup_and_run_model; defaults to
# 2026-01-01 if those params are absent so existing configs are unchanged.
const MODEL_START_DATETIME = Ref{DateTime}(DateTime(2026, 1, 1, 0, 0))

function TimeStep!(sim::MarineSimulation)
    model = sim.model
    envi = model.environment
    fisheries = model.fishing
    outputs = sim.outputs
    species::Int32 = model.n_species
    arch = model.arch

    # Increment iteration and update internal day-time counter [0, 1440]
    model.iteration += 1
    model.t = (model.t + sim.ΔT) % 1440 

    # Rebuild the spatial cell index so the prey-search kernel can scan only
    # the agents in each neighbour cell rather than brute-forcing all prey.
    build_spatial_index!(model)


    # --- 1. Calendar & Date Resolution ---
    # Post-spinup "real" calendar start (set from params.csv start_year in setup).
    target_start_date = MODEL_START_DATETIME[]
    
    elapsed_minutes = (model.iteration - 1) * sim.ΔT
    current_datetime = target_start_date + Minute(round(Int, elapsed_minutes - model.spinup))
    current_date = Date(current_datetime)
    
    # Calculate previous date to handle daily/annual resets
    previous_elapsed = (model.iteration - 2) * sim.ΔT
    previous_datetime = target_start_date + Minute(round(Int, previous_elapsed - model.spinup))
    previous_date = Date(previous_datetime)
    
    month_index = month(current_date)
    day_index = dayofyear(current_date)

    # --- 2. Environmental Dynamics ---
    month_changed = (month_index != envi.ts)
    if month_changed
        move_resources!(model, month_index)
        envi.ts = month_index
    end
    # If time-varying ASC forcing is active, refresh the driver/temperature layers
    # and recompute the SST-driven habitat-capacity slice for this absolute month.
    # Runs on the first iteration and on every month change; the layer cache makes
    # within-month repeats cheap. No-op when ENV_FORCING is unset (NetCDF path).
    if ENV_FORCING[] !== nothing && (model.iteration == 1 || month_changed)
        try
            update_environment_from_asc!(model, current_date)
        catch err
            @warn "ASC environmental update failed at $(current_date): $err"
        end
    end

    # Diel Vertical Migration (DVM) of resource pools
    is_night_now = model.t < 360 || model.t > 1080
    t_prev_step = (model.t - sim.ΔT + 1440) % 1440
    was_night_before = t_prev_step < 360 || t_prev_step > 1080
    
    if is_night_now != was_night_before
        vertical_resource_movement!(model)
    end

    # --- 3. Counters & Accumulator Resets ---
    if current_date != previous_date
        model.daily_birth_counters = zeros(Int, species)
    end

    if year(current_date) != year(previous_date)
        for fishery in fisheries
            fishery.cumulative_catch = 0.0
            fishery.cumulative_inds = 0
            fishery.effort_days = 0.0
            fishery.bycatch_tonnage = 0.0
            fishery.bycatch_inds = 0
        end
        # Biomass-linked quotas: recompute the new year's allowable catch from the
        # current target-species biomass (fixed-quota fisheries are unchanged).
        update_dynamic_quotas!(model)
    end

    # --- 4. Agent Life History & Behavior ---
    print(current_date, ": ")
    
    for spec in 1:species
        # Dynamic storage resizing pass
        if model.iteration % 10 == 0
            n_alive = count(x -> x == 1.0f0, Array(model.individuals.animals[spec].data.alive))
            current_maxN = length(model.individuals.animals[spec].data.x)
            if n_alive > current_maxN * 0.90
                resize_agent_storage!(model, spec, floor(Int, current_maxN * 1.5))
            end
        end

        species_data = model.individuals.animals[spec].data
        species_chars = model.individuals.animals[spec].p
        larval_duration = species_chars.Larval_Duration[2][spec]

        # Filter indices on CPU for biological eligibility
        cpu_alive = Array(species_data.alive)
        cpu_age = Array(species_data.age)
        
        alive_indices = Vector{Int32}(findall(i -> cpu_alive[i] == 1.0f0, eachindex(cpu_alive)))
        modeled_indices = Vector{Int32}(findall(i -> cpu_alive[i] == 1.0f0 && cpu_age[i] >= larval_duration, eachindex(cpu_alive)))

        if !isempty(modeled_indices)
            # Update Population Totals
            model.abund[spec] = Int64(round(sum(Array(species_data.abundance[alive_indices]))))
            model.bioms[spec] = sum(Array(species_data.biomass_school[alive_indices]))
            
            print(length(alive_indices), " Agents | ")
            print(sum(model.abund[spec]), " Abundance | ")

            # Capture initial state for consumption delta analysis
            species_data.biomass_init[alive_indices] = species_data.biomass_school[alive_indices] 

            # Decision, Energetics, and Human Impacts
            print("behave | ")
            behavior(model, spec, modeled_indices, outputs)

            ind_temp = individual_temp!(model, spec)
            print("energy | ")
            energy!(model, spec, ind_temp, modeled_indices, outputs, current_date)

            print("fish | ")
            fishing!(model, spec, day_index, outputs)

            print("mort | ")
            natural_mortality!(model, spec)

            # --- AGENT CONSOLIDATION ---
            # Merge agents sharing location, size bin, generation, and birth step (dt).
            # This pass reduces agent count while preserving specific growth trajectories.
            # Gated to a cadence: the merge copies the whole StructArray to the host,
            # so running it every step is a major transfer cost. `merge_every` lets
            # it run periodically with no change to the merge algorithm itself.
            if RUNTIME_CONFIG[:merge_every] <= 1 || (model.iteration % RUNTIME_CONFIG[:merge_every] == 0)
                print("merge | ")
                data_cpu = StructArray(NamedTuple(k => Array(v) for (k, v) in pairs(StructArrays.components(species_data))))
                size_bin_thresholds_cpu = Array(model.size_bin_thresholds)

                merge_agents!(data_cpu, spec, size_bin_thresholds_cpu, Int(sim.ΔT),species_chars.School_Size[2][spec])

                # Sync consolidated results back to device memory
                copyto!(species_data, data_cpu)
            end
        end
        
        # Increment age in days
        species_data.age .+= (sim.ΔT / 1440.0)
    end

    # --- 5. Trophic Dynamics ---
    print("resources | ")
    resource_predation!(model, outputs)
    resource_mortality!(model)
    resource_growth!(model, current_date)
   
    # --- 6. Results Export (Gated by Spin-up) ---
    if (model.iteration % model.output_dt == 0)
        if elapsed_minutes > model.spinup
            print("results | ")
            timestep_results(sim)
            resource_results(model, sim.run, model.iteration)
            fishery_results(sim)
            println("saved.")
        else
            println("spinup (no output saved).")
        end
    else
        println("")
    end

    # --- 6b. Checkpoint (Gated by cadence) ---
    if RUNTIME_CONFIG[:checkpoint_every] > 0 && (model.iteration % RUNTIME_CONFIG[:checkpoint_every] == 0)
        try
            save_checkpoint(sim)
        catch err
            @warn "Checkpoint write failed at iteration $(model.iteration): $err"
        end
    end

    # --- 7. Buffer Maintenance ---
    for spec in 1:species
        model.individuals.animals[spec].data.ration_biomass .= 0.0f0
        model.individuals.animals[spec].data.ration_energy .= 0.0f0
        model.individuals.animals[spec].data.active .= 0.0f0
    end
end