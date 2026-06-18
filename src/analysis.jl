# ===================================================================
# Top-Level Output Functions
# ===================================================================

# Tracks whether outputs.biomass_ref has been seeded yet. Until it has (first
# output interval), the rate denominator falls back to the current standing
# biomass; thereafter it is the standing biomass at the START of each interval.
const _BIOMASS_REF_READY = Ref(false)
const _LAST_ITERATION = Ref(0)

reset_biomass_ref_state!() = (
    _BIOMASS_REF_READY[] = false; 
    _LAST_ITERATION[] = 0; 
    nothing
)

"""
    timestep_results(sim::MarineSimulation)

Gather individual-level data and population-level matrices for the 
current timestep and export them to CSV and HDF5 formats.
"""
function timestep_results(sim::MarineSimulation)
    model = sim.model
    outputs = sim.outputs
    arch = model.arch
    ts = Int(model.iteration)
    run = Int(sim.run)

    # Get the base results directory from the model's files DataFrame
    files_df = model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]

    # Construct full paths for output subdirectories
    individual_dir = joinpath(res_dir, "Individual")
    population_dir = joinpath(res_dir, "Population")

    # FIX: Ensure directories exist before writing. 
    # The previous 'ts == 1' check failed because the first output occurs at ts = 4.
    !isdir(individual_dir) && mkpath(individual_dir)
    !isdir(population_dir) && mkpath(population_dir)

    # --- 1. Gather individual data for CSV output ---
    Sp, Ind, x, y, z, lengths, abundance, biomass, biomass_init, gut_fullness, ration_biomass, ration_energy, energy, cost, age, generation = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

    for (species_index, animal) in enumerate(model.individuals.animals)
        spec_dat = animal.data
        
        # Pull data from the device to the CPU and filter for living agents
        alive_mask = Array(spec_dat.alive) .== 1.0
        
        append!(Sp, fill(species_index, count(alive_mask)))
        # Match unique_id as defined in the agent StructArray
        append!(Ind, Array(spec_dat.unique_id)[alive_mask])
        append!(x, Array(spec_dat.x)[alive_mask])
        append!(y, Array(spec_dat.y)[alive_mask])
        append!(z, Array(spec_dat.z)[alive_mask])
        append!(lengths, Array(spec_dat.length)[alive_mask])
        append!(abundance, Array(spec_dat.abundance)[alive_mask])
        # Match biomass_school as defined in the agent StructArray
        append!(biomass, Array(spec_dat.biomass_school)[alive_mask])
        append!(biomass_init, Array(spec_dat.biomass_init)[alive_mask])
        append!(gut_fullness, Array(spec_dat.gut_fullness)[alive_mask])
        append!(ration_biomass, Array(spec_dat.ration_biomass)[alive_mask])
        append!(ration_energy, Array(spec_dat.ration_energy)[alive_mask])
        append!(energy, Array(spec_dat.energy)[alive_mask])
        append!(cost, Array(spec_dat.cost)[alive_mask])
        append!(age, Array(spec_dat.age)[alive_mask])
        append!(generation, Array(spec_dat.generation)[alive_mask])
    end
    
    # Save individual data to CSV
    # FIX: Exporting Biomass_init prevents post-predation biomass deflation from skewing % ration plots.
    ind_df = DataFrame(
        Species = Sp, Individual = Ind, X = x, Y = y, Z = z, 
        Length = lengths, Abundance = abundance, Biomass = biomass, Biomass_init = biomass_init,
        Fullness = gut_fullness, Ration_b = ration_biomass, Ration_e = ration_energy, 
        Energy = energy, Cost = cost, Age = age, Generation = generation
    )
    CSV.write(joinpath(individual_dir, "IndividualResults_$(run)_$(ts).csv"), ind_df)

    # --- 2. Calculate time interval in Years for annualizing rates ---
    elapsed_iterations = _BIOMASS_REF_READY[] ? (model.iteration - _LAST_ITERATION[]) : model.iteration
    if elapsed_iterations <= 0
        elapsed_iterations = 1
    end
    interval_minutes = elapsed_iterations * model.dt
    interval_years = Float32(interval_minutes / (365.0 * 1440.0))

    # --- 3. Generate Spatially-Explicit Population Arrays (Annualized) ---
    # Populate the spatial biomass- and abundance-by-size grids from the living agents
    populate_population_grids!(model, outputs)

    # Denominator = standing biomass-by-size at the START of this interval (biomass_ref)
    denom = _BIOMASS_REF_READY[] ? outputs.biomass_ref : outputs.biomass

    arch_arr = array_type(arch)
    F_rate = arch_arr(zeros(Float32, size(outputs.Fmort)))
    S_rate = arch_arr(zeros(Float32, size(outputs.Smort)))
    P_rate = arch_arr(zeros(Float32, size(outputs.Pmort)))
    O_rate = arch_arr(zeros(Float32, size(outputs.Omort)))

    kF = fishing_mortality_kernel!(device(arch), (8, 8, 1, 1, 1, 1), size(outputs.Fmort))
    kF(F_rate, denom, outputs.Fmort, interval_years)
    kS = starvation_mortality_kernel!(device(arch), (8, 8, 1, 1, 1), size(outputs.Smort))
    kS(S_rate, denom, outputs.Smort, interval_years)
    kP = starvation_mortality_kernel!(device(arch), (8, 8, 1, 1, 1), size(outputs.Pmort))
    kP(P_rate, denom, outputs.Pmort, interval_years)
    kO = starvation_mortality_kernel!(device(arch), (8, 8, 1, 1, 1), size(outputs.Omort))
    kO(O_rate, denom, outputs.Omort, interval_years)
    KernelAbstractions.synchronize(device(arch))

    cpu_F       = Array(F_rate)
    cpu_S       = Array(S_rate)
    cpu_P       = Array(P_rate)
    cpu_O       = Array(O_rate)
    cpu_biomass  = Array(outputs.biomass)
    cpu_abund    = Array(outputs.abundance)

    cpu_DC_full = Array(outputs.consumption)
    cpu_DC = dropdims(sum(cpu_DC_full, dims = 7), dims = 7)  # [lon,lat,depth,pred_sp,prey_sp,pred_bin]
    cpu_DC_full = nothing

    # Save spatially-explicit Population Results to HDF5
    h5_path = joinpath(population_dir, "Population_Results_$(run)_$(ts).h5")
    h5open(h5_path, "w") do file
        file["F",         deflate = 4, shuffle = true] = cpu_F   # fishing      [lon,lat,depth,fishery,sp,bin]
        file["S",         deflate = 4, shuffle = true] = cpu_S   # starvation   [lon,lat,depth,sp,bin]
        file["P",         deflate = 4, shuffle = true] = cpu_P   # predation    [lon,lat,depth,sp,bin]
        file["O",         deflate = 4, shuffle = true] = cpu_O   # other/senesc [lon,lat,depth,sp,bin]
        file["Diet",      deflate = 4, shuffle = true] = cpu_DC
        file["Biomass",   deflate = 4, shuffle = true] = cpu_biomass
        file["Abundance", deflate = 4, shuffle = true] = cpu_abund
    end

    # --- 3b. Spatially-integrated mortality time series (Annualized) ---
    bmort = Array(outputs.Fmort); smort = Array(outputs.Smort)
    pmort = Array(outputs.Pmort); omort = Array(outputs.Omort)
    cpu_denom = Array(denom)
    
    # Mathematical correction: Dividies the instant rate by elapsed years to get a true annual rate (Z)
    inst(kill, bio) = (bio > 0f0 && interval_years > 0.0f0) ? -log(max(1f-6, 1f0 - clamp(kill / bio, 0f0, 1f0))) / interval_years : 0f0
    
    ts_path = joinpath(population_dir, "MortalityTimeSeries_$(run).csv")
    write_header = !isfile(ts_path)
    open(ts_path, "a") do io
        write_header && println(io, "run,iteration,species,F,S,P,O,Z,biomass,abundance")
        for sp in 1:model.n_species
            denom_sp = sum(@view cpu_denom[:, :, :, sp, :])
            F_sp = inst(sum(@view bmort[:, :, :, :, sp, :]), denom_sp)
            S_sp = inst(sum(@view smort[:, :, :, sp, :]), denom_sp)
            P_sp = inst(sum(@view pmort[:, :, :, sp, :]), denom_sp)
            O_sp = inst(sum(@view omort[:, :, :, sp, :]), denom_sp)
            bio_sp = sum(@view cpu_biomass[:, :, :, sp, :])
            ab_sp  = sum(@view cpu_abund[:, :, :, sp, :])
            println(io, "$run,$ts,$sp,$F_sp,$S_sp,$P_sp,$O_sp,$(F_sp+S_sp+P_sp+O_sp),$bio_sp,$ab_sp")
        end
    end

    # --- 4. Reset accumulators and snapshot start-of-next-interval biomass ---
    fill!(outputs.Fmort, 0.0f0)
    fill!(outputs.Smort, 0.0f0)
    fill!(outputs.Pmort, 0.0f0)
    fill!(outputs.Omort, 0.0f0)
    fill!(outputs.consumption, 0.0f0)
    copyto!(outputs.biomass_ref, outputs.biomass)
    
    _BIOMASS_REF_READY[] = true
    _LAST_ITERATION[] = model.iteration
end

"""
    resource_results(model::MarineModel, run::Integer, ts::Integer)

Export the current spatial biomass density of all resource groups.
Uses Integer type to handle both Int32 and Int64 inputs safely.
"""
function resource_results(model::MarineModel, run::Integer, ts::Integer)
    files_df = model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]
    
    resource_dir = joinpath(res_dir, "Resource")
    !isdir(resource_dir) && mkpath(resource_dir)
    
    # Gathering data for Resource results
    cpu_res = Array(model.resources.biomass)
    lonres, latres, depthres, n_res = size(cpu_res)
    
    lons, lats, depths, res_ids, biomass_density = [], [], [], [], []
    
    for r in 1:n_res, k in 1:depthres, j in 1:latres, i in 1:lonres
        val = cpu_res[i, j, k, r]
        if val > 1e-6 # Sparse threshold for CSV output
            push!(lons, i); push!(lats, j); push!(depths, k)
            push!(res_ids, r); push!(biomass_density, val)
        end
    end
    
    res_df = DataFrame(LonIndex=lons, LatIndex=lats, DepthIndex=depths, ResourceID=res_ids, BiomassDensity=biomass_density)
    CSV.write(joinpath(resource_dir, "resource_results_$(run)_$(ts).csv"), res_df)
end

"""
    fishery_results(sim::MarineSimulation)

Export catch and effort metrics for all active fishing fleets.
"""
function fishery_results(sim::MarineSimulation)
    ts = Int(sim.model.iteration)
    run = Int(sim.run)
    fisheries = sim.model.fishing

    # Get the base results directory from the model's files DataFrame
    files_df = sim.model.files
    res_dir = files_df[files_df.File .== "res_dir", :Destination][1]
    
    # Construct full path for the fishery output subdirectory
    fish_dir = joinpath(res_dir, "Fishery")
    !isdir(fish_dir) && mkpath(fish_dir)

    name, quotas, catches_t, catches_ind = [], [], [], []
    effort, cpue, mean_len, bycatch_t, bycatch_n = [], [], [], [], []  

    for fishery in sim.model.fishing
        push!(name, fishery.name)
        push!(quotas, fishery.quota)
        push!(catches_t, fishery.cumulative_catch)
        push!(catches_ind, fishery.cumulative_inds)
        
        push!(effort, fishery.effort_days)
        current_cpue = fishery.effort_days > 0 ? fishery.cumulative_catch / fishery.effort_days : 0.0
        push!(cpue, current_cpue)
        push!(mean_len, fishery.mean_length_catch)
        push!(bycatch_t, fishery.bycatch_tonnage)
        push!(bycatch_n, fishery.bycatch_inds)
    end

    df = DataFrame(
        Name=name, Quota=quotas, Tonnage=catches_t, Individuals=catches_ind,
        Effort_Days=effort, CPUE_T_per_Day=cpue, Mean_Length_mm=mean_len,
        Bycatch_Tonnage=bycatch_t, Bycatch_Individuals=bycatch_n
    )    # Use the constructed path for writing the CSV file
    csv_path = joinpath(fish_dir, "FisheryResults_$run-$ts.csv")
    CSV.write(csv_path, df)

    for fishery in fisheries
        fishery.mean_length_catch = 0.0
        fishery.mean_weight_catch = 0.0
    end
end

# --- Post-Processing Helpers & Kernels ---

"""
    populate_population_grids!(model, outputs)

Aggregate living-agent biomass and abundance into the spatially explicit,
size-structured population grids (`outputs.biomass`, `outputs.abundance`),
and add resource biomass into the resource-species slots.
"""
function populate_population_grids!(model, outputs)
    arch = model.arch
    fill!(outputs.biomass, 0.0f0)
    fill!(outputs.abundance, 0.0f0)

    animals_all = Tuple(animal.data for animal in model.individuals.animals)
    max_agents = maximum(length(a.x) for a in animals_all)

    # Focal agents -> [lon,lat,depth,sp,bin]
    k1 = aggregate_biomass_abundance_kernel!(device(arch), 256, (max_agents,))
    k1(outputs.biomass, outputs.abundance, animals_all,
       model.size_bin_thresholds, Int32(model.n_species))

    # Resource biomass -> resource species slots [lon,lat,depth, n_species+r, 1]
    if model.n_resource > 0
        rb = model.resources.biomass
        k2 = aggregate_resource_biomass_kernel!(device(arch), (8, 8, 4, 1), size(rb))
        k2(outputs.biomass, rb, Int32(model.n_species))
    end
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

# Per-agent aggregation of biomass and abundance into [lon,lat,depth,sp,bin].
@kernel function aggregate_biomass_abundance_kernel!(
    biomass_grid, abundance_grid, animals_all, size_bin_thresholds, n_species::Int32
)
    i = @index(Global)
    for sp in 1:length(animals_all)
        animal = animals_all[sp]
        if i <= length(animal.x) && animal.alive[i] == 1.0f0
            px, py, pz = animal.pool_x[i], animal.pool_y[i], animal.pool_z[i]
            lonres = size(biomass_grid, 1); latres = size(biomass_grid, 2); depthres = size(biomass_grid, 3)
            if px > 0 && px <= lonres && py > 0 && py <= latres && pz > 0 && pz <= depthres
                bin = find_species_size_bin(animal.length[i], sp, size_bin_thresholds)
                if bin > 0 && bin <= size(biomass_grid, 5)
                    @atomic biomass_grid[px, py, pz, sp, bin]   += animal.biomass_school[i]
                    @atomic abundance_grid[px, py, pz, sp, bin] += Float32(animal.abundance[i])
                end
            end
        end
    end
end

# Resource biomass density summed into the resource species slot (bin 1).
@kernel function aggregate_resource_biomass_kernel!(
    biomass_grid, resource_biomass, n_species::Int32
)
    lon, lat, depth, r = @index(Global, NTuple)
    val = resource_biomass[lon, lat, depth, r]
    if val > 0.0f0
        sp_slot = Int(n_species) + r
        if sp_slot <= size(biomass_grid, 4)
            @atomic biomass_grid[lon, lat, depth, sp_slot, 1] += val
        end
    end
end

"""
Calculates instantaneous fishing mortality (F) from the 6D fishing mortality array (Annualized).
"""
@kernel function fishing_mortality_kernel!(Rate, biomass_by_size, fishing_mortality, interval_years::Float32)
    lon, lat, depth, fishery, prey_sp, prey_bin = @index(Global, NTuple)
    
    @inbounds if fishing_mortality[lon, lat, depth, fishery, prey_sp, prey_bin] > 0
        biomass_val = biomass_by_size[lon, lat, depth, prey_sp, prey_bin]
        mort_val = fishing_mortality[lon, lat, depth, fishery, prey_sp, prey_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            mort_frac = FT(mort_val) / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, fishery, prey_sp, prey_bin] = -log(FT(1.0) - mort_frac) / FT(interval_years)
            else
                Rate[lon, lat, depth, fishery, prey_sp, prey_bin] = FT(10.0) / FT(interval_years)
            end
        end
    end
end

"""
Calculates instantaneous starvation/predation/other mortality from raw arrays (Annualized).
"""
@kernel function starvation_mortality_kernel!(Rate, biomass_by_size, mortality, interval_years::Float32)
    lon, lat, depth, sp, size_bin = @index(Global, NTuple)
    
    @inbounds if mortality[lon, lat, depth, sp, size_bin] > 0
        biomass_val = biomass_by_size[lon, lat, depth, sp, size_bin]
        mort_val = mortality[lon, lat, depth, sp, size_bin]
        
        @inbounds if biomass_val > 0
            FT = eltype(Rate)
            mort_frac = FT(mort_val) / biomass_val
            mort_frac = clamp(mort_frac, FT(0.0), FT(1.0))
            
            if mort_frac < FT(1.0)
                Rate[lon, lat, depth, sp, size_bin] = -log(FT(1.0) - mort_frac) / FT(interval_years)
            else
                Rate[lon, lat, depth, sp, size_bin] = FT(10.0) / FT(interval_years)
            end
        end
    end
end

# ===================================================================
# Checkpoint / Restart
# ===================================================================
"""
    save_checkpoint(sim)

Serialize enough model state to resume a run.
"""
function save_checkpoint(sim::MarineSimulation)
    model = sim.model
    res_dir = model.files[model.files.File .== "res_dir", :Destination][1]
    ckpt_dir = joinpath(res_dir, "Checkpoint")
    !isdir(ckpt_dir) && mkpath(ckpt_dir)

    agent_data_cpu = [
        StructArray(NamedTuple(k => Array(v) for (k, v) in pairs(StructArrays.components(a.data))))
        for a in model.individuals.animals
    ]

    state = (
        iteration            = model.iteration,
        t                    = model.t,
        run                  = sim.run,
        envi_ts              = model.environment.ts,
        daily_birth_counters = copy(model.daily_birth_counters),
        abund                = copy(model.abund),
        bioms                = copy(model.bioms),
        agent_data           = agent_data_cpu,
        resources_biomass    = Array(model.resources.biomass),
        fishing              = deepcopy(model.fishing),
    )

    path = joinpath(ckpt_dir, "checkpoint_run$(sim.run).jls")
    tmp  = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, state)
    end
    mv(tmp, path; force = true)
    @info "Checkpoint written: $path (iteration $(model.iteration))"
    return path
end

"""
    load_checkpoint!(sim, path)

Restore model state previously written by `save_checkpoint`.
"""
function load_checkpoint!(sim::MarineSimulation, path::String)
    model = sim.model
    state = open(deserialize, path)

    model.iteration       = state.iteration
    model.t               = state.t
    model.environment.ts  = state.envi_ts
    model.daily_birth_counters = state.daily_birth_counters
    model.abund          .= state.abund
    model.bioms          .= state.bioms

    copyto!(model.resources.biomass, array_type(model.arch)(state.resources_biomass))

    for (sp, cpu) in enumerate(state.agent_data)
        model.individuals.animals[sp].data = replace_storage(array_type(model.arch), cpu)
    end
    for (i, f) in enumerate(state.fishing)
        model.fishing[i] = f
    end

    # Restore the analytics interval tracker
    _LAST_ITERATION[] = model.iteration
    _BIOMASS_REF_READY[] = true

    @info "Resumed from checkpoint: $path (iteration $(model.iteration))"
    return nothing
end