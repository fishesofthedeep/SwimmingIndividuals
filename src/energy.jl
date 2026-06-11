# ===================================================================
# Recruitment parameter reader
# -------------------------------------------------------------------
# Reads per-species recruitment-forcing parameters from an optional input file
# registered in files.csv under File == "recruitment_env". Columns (any subset):
#   Species, T_opt, T_sd, sigma_R, dd_halfsat
# where
#   T_opt      = thermal optimum (deg C) for egg/larval survival,
#   T_sd       = thermal window s.d. (deg C),
#   sigma_R    = lognormal recruitment-deviation s.d. (process error),
#   dd_halfsat = LOCAL spawner biomass (same units as biomass_school) at which
#                early-life survival is halved; dd_beta = 1/dd_halfsat.
#
# If the file or species row is absent, literature-based King-Mackerel-style
# defaults are used (and logged once): T_opt=27, T_sd=4, sigma_R=0.5, and a
# dd_halfsat scaled to the current mean occupied-cell spawner biomass so density
# dependence is active but gentle. Set values explicitly to control precisely;
# set sigma_R=0 and dd_halfsat=Inf to recover the deterministic behaviour.
# ===================================================================
const _RECRUIT_PARAM_WARNED = Ref(false)

function read_recruitment_params(model::MarineModel, sp::Int, spawner_map::AbstractMatrix)
    species_name = model.individuals.animals[sp].p.SpeciesLong.second[sp]

    # sensible defaults
    T_opt   = 27.0
    T_sd    = 4.0
    sigma_R = 0.5
    dd_halfsat = NaN   # resolved below if not provided

    have_file = any(model.files.File .== "recruitment_env")
    row = nothing
    if have_file
        df = CSV.read(model.files[model.files.File .== "recruitment_env", :Destination][1], DataFrame)
        ridx = findfirst(==(species_name), df.Species)
        if ridx !== nothing
            row = df[ridx, :]
            hasproperty(row, :T_opt)      && !ismissing(row.T_opt)      && (T_opt   = Float64(row.T_opt))
            hasproperty(row, :T_sd)       && !ismissing(row.T_sd)       && (T_sd    = Float64(row.T_sd))
            hasproperty(row, :sigma_R)    && !ismissing(row.sigma_R)    && (sigma_R = Float64(row.sigma_R))
            hasproperty(row, :dd_halfsat) && !ismissing(row.dd_halfsat) && (dd_halfsat = Float64(row.dd_halfsat))
        end
    end

    # Resolve density-dependence half-saturation if not supplied: tie it to the
    # current typical occupied-cell spawner biomass so compensation is active but
    # not overwhelming. dd_beta = 1/dd_halfsat.
    if !isfinite(dd_halfsat)
        occupied = filter(>(0f0), vec(spawner_map))
        ref_sb = isempty(occupied) ? 0.0 : Float64(sum(occupied) / length(occupied))
        dd_halfsat = ref_sb > 0 ? 2.0 * ref_sb : Inf   # halve survival at ~2x mean local SSB
    end
    dd_beta = isfinite(dd_halfsat) && dd_halfsat > 0 ? 1.0 / dd_halfsat : 0.0

    if !have_file && !_RECRUIT_PARAM_WARNED[]
        @info "No 'recruitment_env' file in files.csv; using default recruitment forcing " *
              "(T_opt=$T_opt, T_sd=$T_sd, sigma_R=$sigma_R, dd_halfsat≈$(round(dd_halfsat, sigdigits=3))). " *
              "Add recruitment_env to control these per species."
        _RECRUIT_PARAM_WARNED[] = true
    end

    return T_opt, T_sd, dd_beta, sigma_R
end

function energy!(model::MarineModel, sp::Int, temp::AbstractArray, indices, outputs,current_date)
    arch = model.arch
    p_cpu = model.individuals.animals[sp].p
    dt = model.dt

    # --- 1. GATHER DATA ---
    agent_data_device = model.individuals.animals[sp].data

    data_cpu = StructArray(
        unique_id = Array(agent_data_device.unique_id),x = Array(agent_data_device.x), y = Array(agent_data_device.y), z = Array(agent_data_device.z),
        length = Array(agent_data_device.length), abundance = Array(agent_data_device.abundance),
        biomass_ind = Array(agent_data_device.biomass_ind), biomass_school = Array(agent_data_device.biomass_school), biomass_init = Array(agent_data_device.biomass_init),
        energy = Array(agent_data_device.energy), gut_fullness = Array(agent_data_device.gut_fullness),
        cost = Array(agent_data_device.cost), pool_x = Array(agent_data_device.pool_x),
        pool_y = Array(agent_data_device.pool_y), pool_z = Array(agent_data_device.pool_z),
        active = Array(agent_data_device.active), ration_biomass = Array(agent_data_device.ration_biomass),ration_energy = Array(agent_data_device.ration_energy),
        alive = Array(agent_data_device.alive), vis_prey = Array(agent_data_device.vis_prey),
        target_pool_x = Array(agent_data_device.target_pool_x),
        target_pool_y = Array(agent_data_device.target_pool_y),
        mature = Array(agent_data_device.mature), age = Array(agent_data_device.age),
        cell_id = Array(agent_data_device.cell_id), sorted_id = Array(agent_data_device.sorted_id),
        repro_energy = Array(agent_data_device.repro_energy),
        best_prey_dist = Array(agent_data_device.best_prey_dist),
        best_prey_idx = Array(agent_data_device.best_prey_idx),
        best_prey_sp = Array(agent_data_device.best_prey_sp),
        best_prey_type = Array(agent_data_device.best_prey_type),
        successful_ration = Array(agent_data_device.successful_ration),
        temp_idx = Array(agent_data_device.temp_idx),
        cell_starts = Array(agent_data_device.cell_starts),
        cell_ends = Array(agent_data_device.cell_ends),
        mig_status = Array(agent_data_device.mig_status),
        target_z = Array(agent_data_device.target_z),
        interval = Array(agent_data_device.interval),
        generation = Array(agent_data_device.generation)
    )
    
    daily_births = model.daily_birth_counters[sp]
    temp_cpu = Array(temp)
    size_bin_thresholds_cpu = Array(model.size_bin_thresholds)
    smort_cpu = Array(outputs.Smort)
    omort_cpu = Array(outputs.Omort)   # "other" mortality (senescence etc.)

    # --- 2. COMPUTE BIOENERGETICS ON CPU ---
    spawn_season = CSV.read(model.files[model.files.File .== "reproduction", :Destination][1], DataFrame)
    species_name = p_cpu.SpeciesLong.second[sp]
    row_idx = findfirst(==(species_name), spawn_season.Species)
    spawn_val = row_idx !== nothing ? spawn_season[row_idx, model.environment.ts + 1] : 0.0
    
    for ind in 1:length(data_cpu.x)
        if (data_cpu.alive[ind] == 1.0) && (data_cpu.age[ind] > p_cpu.Larval_Duration.second[sp])

            # Gather Agent Properties
            my_temp = temp_cpu[ind]; my_consumed = data_cpu.ration_energy[ind]
            my_weight_ind = data_cpu.biomass_ind[ind]
            my_active_time = data_cpu.active[ind] / dt
            my_abundance = data_cpu.abundance[ind]
            my_z = data_cpu.z[ind]; my_length = data_cpu.length[ind]
            my_energy = data_cpu.energy[ind]; my_mature = data_cpu.mature[ind]

            # Get Species-Specific Trait Codes
            energy_ed = p_cpu.Energy_density.second[sp]; taxa_code = p_cpu.Taxa.second[sp]
            energy_type_code = p_cpu.MR_type.second[sp]; max_size = p_cpu.Max_Size.second[sp]

            activity_multiplier = p_cpu.Activity_Mult.second[sp]

            # --- Respiration (R) ---
            R = 0.0
            oxy_joules_per_mg = 13.6
            if my_weight_ind > 0 && my_abundance > 0
                if energy_type_code == 2 # "cetacean"
                    joules_per_kcal = 4184.0
                    min_fmr_kcal_day = 350.0 * (my_weight_ind / 1000.0)^0.75
                    max_fmr_kcal_day = 420.0 * (my_weight_ind / 1000.0)^0.75
                    min_fmr_J_ts = min_fmr_kcal_day * joules_per_kcal * (dt / 1440.0)
                    max_fmr_J_ts = max_fmr_kcal_day * joules_per_kcal * (dt / 1440.0)
                    rmr = min_fmr_J_ts + (max_fmr_J_ts - min_fmr_J_ts) * my_active_time
                    cost_resting = rmr * (1.0 - my_active_time)
                    cost_active = (rmr * activity_multiplier) * my_active_time
                    R_ind = cost_resting + cost_active
                elseif energy_type_code == 3 # "deepsea"
                    depth = max(1.0, my_z)
                    log_weight = log(my_weight_ind); inv_temp = 1000.0 / (273.15 + my_temp); log_depth = log(depth)
                    lnr = (taxa_code == 1 ? 19.491 + 0.885*log_weight - 5.770*inv_temp - 0.261*log_depth :
                           taxa_code == 2 ? 28.326 + 0.779*log_weight - 7.903*inv_temp - 0.365*log_depth :
                           18.775 + 0.766*log_weight - 5.265*inv_temp - 0.113*log_depth)
                    rate_mg_per_kg_per_hr = exp(lnr); my_weight_kg = my_weight_ind / 1000.0
                    rate_mg_per_ind_per_hr = rate_mg_per_kg_per_hr * my_weight_kg
                    rate_J_per_hr = rate_mg_per_ind_per_hr * oxy_joules_per_mg
                    rmr = rate_J_per_hr * (dt / 60.0)
                    cost_resting = rmr * (1.0 - my_active_time)
                    cost_active = (rmr * activity_multiplier) * my_active_time
                    R_ind = cost_resting + cost_active
                else # Default
                    resp_a = p_cpu.Resp_a.second[sp]; resp_b = p_cpu.Resp_b.second[sp]; resp_q = p_cpu.Resp_q.second[sp]
                    # 1. Calculate mass-specific rate (mg O2 / g / day)
                    smr_mass_specific = resp_a * my_weight_ind^resp_b
                    
                    # 2. Apply temperature correction
                    smr_temp_corrected = smr_mass_specific * exp(resp_q * my_temp)
                    
                    # 3. Calculate total rate for the individual (mg O2 / ind / day)
                    smr_total_daily = smr_temp_corrected * my_weight_ind
                    
                    # 4. Convert to Joules per timestep
                    smr_joules_per_day = smr_total_daily * oxy_joules_per_mg
                    smr_per_timestep = smr_joules_per_day * (dt / 1440.0)
                    
                    cost_resting = smr_per_timestep * (1.0 - my_active_time)
                    cost_active = (smr_per_timestep * activity_multiplier) * my_active_time
                    R_ind = cost_resting + cost_active
                end
                R = R_ind * my_abundance
            end

            # --- Net Energy and Gut Evacuation ---
            sda_coeff, egestion_coeff, excretion_coeff = 0.15,0.2,0.08
            total_waste_and_cost = R + (my_consumed * (sda_coeff + egestion_coeff + excretion_coeff))
            if ismissing(total_waste_and_cost)
                continue
            end
            net_energy = my_consumed - total_waste_and_cost
            data_cpu.cost[ind] = total_waste_and_cost
            
            # The agent's total energy reserve is updated
            my_energy += net_energy

            # --- Gut evacuation -------------------------------------------------
            # `evac_prop` is the per-HOUR evacuated fraction (Elliott & Persson 1978
            # exponential evacuation, temperature-dependent). The update
            # gut *= (1 - evac_prop)^(hours) is a continuous decay and is therefore
            # dt-robust: at a 30-min step it removes a little, at a weekly step it
            # drives the gut to ~0 (a mackerel gut turns over in well under a day,
            # so full evacuation across a week is correct). NOTE: gut evacuation does
            # NOT gate movement here — `decision()` always passes the full timestep
            # to `movement_toward_habitat!`, so feeding cannot "use up" the time an
            # agent has to move. (Guard against non-finite temperatures, which can
            # now occur where an ASC driver layer has NODATA.)
            safe_temp = isfinite(my_temp) ? my_temp : 20.0
            evac_prop = clamp(0.053 * exp(0.073 * safe_temp), 0.0, 1.0)
            if evac_prop < 1.0
                data_cpu.gut_fullness[ind] *= exp((dt / 60.0) * log(1.0 - evac_prop))
            else
                data_cpu.gut_fullness[ind] = 0.0
            end

            # ===================================================================
            # --- Growth & Reproduction Logic ---
            # ===================================================================
            lwr_a = p_cpu.LWR_a.second[sp]; lwr_b = p_cpu.LWR_b.second[sp]
            
            # Surplus energy is any energy available above metabolic costs.
            surplus_energy = max(0.0, net_energy)

            if my_length < max_size && surplus_energy > 0.0
                L50 = p_cpu.L_mat.second[sp]
                growth_energy = 0.0
                repro_energy_gain = 0.0

                if my_mature == 0.0
                    # --- IMMATURE: All surplus energy goes to somatic growth ---
                    growth_energy = surplus_energy
                else
                    # --- MATURE: Partition energy between growth and reproduction ---
                    # This function creates a smooth switch from growth to reproduction
                    # as the agent approaches its maximum size.
                    min_growth_allocation = 0.1 
                    energy_for_repro = surplus_energy * (1.0 - min_growth_allocation)

                    repro_prop = (my_length / max_size)^1.5
                    repro_prop = clamp(repro_prop, 0.0, 1.0)
                    
                    repro_energy_gain = energy_for_repro * repro_prop
                    growth_energy = (surplus_energy * min_growth_allocation) + (energy_for_repro - repro_energy_gain)
                end

                # --- Apply Somatic Growth ---
                if growth_energy > 0.0
                    # Convert growth energy to biomass for an individual
                    growth_biomass_ind = (growth_energy / my_abundance) / energy_ed
                    new_biomass_ind = my_weight_ind + growth_biomass_ind
                    new_length = 10.0 * (new_biomass_ind / lwr_a)^(1.0 / lwr_b)
                    
                    # Update agent state
                    data_cpu.biomass_ind[ind] = new_biomass_ind
                    data_cpu.length[ind] = new_length
                    data_cpu.biomass_school[ind] += (growth_energy / energy_ed)

                    # Check for maturation
                    if new_length >= L50 && my_mature == 0.0
                        data_cpu.mature[ind] = 1.0
                    end
                end

                # --- Apply Reproductive Energy ---
                if my_mature == 1.0 && repro_energy_gain > 0.0
                    data_cpu.repro_energy[ind] += repro_energy_gain
                end
            end

            data_cpu.energy[ind] = my_energy
            
            # --- Starvation ---
            if my_energy < 0.0
                data_cpu.alive[ind] = 0.0
                x = data_cpu.pool_x[ind]; y = data_cpu.pool_y[ind]; z = data_cpu.pool_z[ind]
                size_bin = find_species_size_bin(data_cpu.length[ind], sp, size_bin_thresholds_cpu)
                if size_bin > 0
                    smort_cpu[x, y, z, sp, size_bin] += my_weight_ind
                end
            end

            # --- Senescence ---
            # Treat the size-dependent term as an ANNUAL instantaneous hazard and
            # convert to a per-timestep probability, so the senescence schedule is
            # invariant to dt (the old per-step form made weekly vs 5-min runs
            # die at wildly different effective rates).
            senescence_rate_annual = exp(50.0 * (my_length / max_size - 0.95))  # yr^-1
            minutes_per_year = 365.0 * 1440.0
            senescence_prob_step = 1.0 - exp(-senescence_rate_annual * (dt / minutes_per_year))
            if rand(Float32) < senescence_prob_step
                data_cpu.alive[ind] = 0.0
                # Record as "other" (non-predation/starvation/fishing) mortality so
                # the mortality budget closes against the abundance decline.
                xo = data_cpu.pool_x[ind]; yo = data_cpu.pool_y[ind]; zo = data_cpu.pool_z[ind]
                sbin_o = find_species_size_bin(data_cpu.length[ind], sp, size_bin_thresholds_cpu)
                if sbin_o > 0 && 1 <= xo <= size(omort_cpu,1) && 1 <= yo <= size(omort_cpu,2) && 1 <= zo <= size(omort_cpu,3)
                    omort_cpu[xo, yo, zo, sp, sbin_o] += my_weight_ind
                end
            end
        end
    end

    ## Double check that larvae cannot lose energy
    larval_mask = (data_cpu.alive .== 1.0) .& (data_cpu.age .<= p_cpu.Larval_Duration.second[sp])
    if any(larval_mask)
        data_cpu.cost[larval_mask] .= 0.0f0
    end
    
    # --- Process Reproduction on the CPU ---
    # Agents only spawn when their reproductive energy pool reaches 5% of their total body energy (GSI threshold)
    # This prevents them from continuously "micro-spawning" every single timestep during the spawning month.
    # FIXED: Compares school reproductive energy to school body energy.
    repro_inds = findall((data_cpu.repro_energy .> (data_cpu.biomass_school .* Float32(p_cpu.Energy_density.second[sp]) .* 0.05f0)) .& 
                         (data_cpu.alive .== 1.0) .& 
                         (data_cpu.age .> p_cpu.Larval_Duration.second[sp]))

    if !isempty(repro_inds) && spawn_val > 0
        parent_data_for_repro = (
            x = data_cpu.x[repro_inds], y = data_cpu.y[repro_inds], z = data_cpu.z[repro_inds],
            pool_x = data_cpu.pool_x[repro_inds], pool_y = data_cpu.pool_y[repro_inds], pool_z = data_cpu.pool_z[repro_inds],
            biomass_ind = data_cpu.biomass_ind[repro_inds], generation = data_cpu.generation[repro_inds]
        )

        # --- Environmental & density-dependent recruitment inputs ---------------
        # (a) Per-parent ambient temperature (drives thermal early-life survival).
        parent_temp = Float32.(temp_cpu[repro_inds])

        # (b) Local conspecific spawner biomass per parent cell. Density-dependent
        #     egg/larval survival uses this so Beverton-Holt compensation emerges
        #     from LOCAL crowding rather than being imposed at the stock level.
        grid_r = model.depths.grid
        lonres_r = Int(grid_r[grid_r.Name .== "lonres", :Value][1])
        latres_r = Int(grid_r[grid_r.Name .== "latres", :Value][1])
        spawner_map = zeros(Float32, lonres_r, latres_r)
        @inbounds for k in eachindex(data_cpu.alive)
            if data_cpu.alive[k] == 1.0f0 && data_cpu.mature[k] == 1.0f0
                px = data_cpu.pool_x[k]; py = data_cpu.pool_y[k]
                if 1 <= px <= lonres_r && 1 <= py <= latres_r
                    spawner_map[px, py] += data_cpu.biomass_school[k]
                end
            end
        end
        local_spawner_biomass = Float32[
            (1 <= data_cpu.pool_x[i] <= lonres_r && 1 <= data_cpu.pool_y[i] <= latres_r) ?
                spawner_map[data_cpu.pool_x[i], data_cpu.pool_y[i]] : 0.0f0
            for i in repro_inds
        ]

        # (c) Recruitment parameters (per species), read from recruitment_env file
        #     if present, otherwise literature-based King-Mackerel-style defaults.
        T_opt, T_sd, dd_beta, sigma_R = read_recruitment_params(model, sp, spawner_map)

        repro_result = calculate_new_offspring_cpu(
            p_cpu, parent_data_for_repro, data_cpu.repro_energy[repro_inds],
            spawn_val, sp, current_date, daily_births;
            parent_temp = parent_temp,
            local_spawner_biomass = local_spawner_biomass,
            T_opt = T_opt, T_sd = T_sd, dd_beta = dd_beta, sigma_R = sigma_R
        )
        
        if repro_result !== nothing
            new_offspring, daily_births_update = repro_result
            model.daily_birth_counters[sp] = daily_births_update

            # calculate_new_offspring_cpu returns (nothing, daily_births) when no
            # parent produced a whole egg this season, so guard on new_offspring
            # (the tuple itself is never `nothing`).
            num_new = new_offspring === nothing ? 0 : length(new_offspring.x)

            if num_new > 0
                dead_slots = findall(data_cpu.alive .== 0)
                
                if num_new > length(dead_slots)
                    current_maxN = length(data_cpu.x)
                    needed_slots = num_new - length(dead_slots)
                    new_maxN = max(floor(Int, current_maxN * 1.5), current_maxN + needed_slots)
                    
                    copyto!(model.individuals.animals[sp].data, data_cpu)
                    copyto!(outputs.Smort, smort_cpu)
                    copyto!(outputs.Omort, omort_cpu)

                    resize_agent_storage!(model, sp, new_maxN)

                    agent_data_device = model.individuals.animals[sp].data
                    data_cpu = StructArray(NamedTuple(k => Array(v) for (k, v) in pairs(StructArrays.components(agent_data_device))))
                    smort_cpu = Array(outputs.Smort)
                    omort_cpu = Array(outputs.Omort)
                    
                    dead_slots = findall(data_cpu.alive .== 0)
                end

                num_to_add = min(num_new, length(dead_slots))

                if num_to_add > 0
                    slots_to_fill = @view dead_slots[1:num_to_add]
                    for (field, values) in pairs(new_offspring)
                        getproperty(data_cpu, field)[slots_to_fill] .= @view(values[1:num_to_add])
                    end
                    data_cpu.alive[slots_to_fill] .= 1.0
                end
            end
        end
        
        safe_spawn_val = clamp(Float32(spawn_val), 0.0f0, 1.0f0)
        data_cpu.repro_energy[repro_inds] .*= (1.0f0 - safe_spawn_val)
    end

    # --- 3. UPDATE: Copy the modified CPU data back to the original device arrays ---
    copyto!(outputs.Smort, smort_cpu)
    copyto!(outputs.Omort, omort_cpu)
    copyto!(model.individuals.animals[sp].data, data_cpu)

    return nothing
end