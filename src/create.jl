# ===================================================================
# Agent Creation and Initialization
# ===================================================================

function construct_individuals(arch::Architecture, params::Dict, maxN)
    # Defines the full data structure for an agent.
    rawdata = StructArray(
        unique_id = zeros(Int, maxN), x = zeros(Float32,maxN), y = zeros(Float32,maxN), z = zeros(Float32,maxN),
        length = zeros(Float32,maxN), abundance = zeros(Int64,maxN),
        biomass_ind = zeros(Float32,maxN), biomass_school = zeros(Float32,maxN),biomass_init = zeros(Float32,maxN),
        energy = zeros(Float32,maxN), gut_fullness = zeros(Float32,maxN),
        cost = zeros(Float32,maxN), pool_x = zeros(Int,maxN), pool_y = zeros(Int,maxN),
        pool_z = zeros(Int,maxN), active = zeros(Float32,maxN),
        ration_energy = zeros(Float32,maxN),ration_biomass = zeros(Float32,maxN), alive = zeros(Float32,maxN),
        vis_prey = zeros(Float32,maxN), mature = zeros(Float32,maxN),
        age = zeros(Float32,maxN),
        generation = zeros(Int32,maxN),
        cell_id = zeros(Int, maxN),
        sorted_id = zeros(Int, maxN),
        repro_energy = zeros(Float32, maxN),
        best_prey_dist = zeros(Float32, maxN),
        best_prey_idx = zeros(Int, maxN),
        best_prey_sp = zeros(Int, maxN),
        best_prey_type = zeros(Int, maxN),
        successful_ration = zeros(Float32, maxN),
        temp_idx = zeros(Int, maxN),
        cell_starts = zeros(Int, maxN),
        cell_ends = zeros(Int, maxN),
        mig_status = zeros(Float32, maxN),
        target_z = zeros(Float32, maxN),
        interval = zeros(Float32, maxN),
        target_pool_x = zeros(Float32, maxN),
        target_pool_y = zeros(Float32, maxN)
    )

    data = replace_storage(array_type(arch), rawdata)

    param_names=(:Dive_Interval,:Min_Prey,:LWR_b, :Surface_Interval,:Resp_b,:SpeciesLong, :LWR_a, :Max_Stomach_a, :Larval_Size,:Max_Prey, :Max_Size,:Resp_q,:School_Size,:Activity_Mult,:Taxa, :Larval_Duration,:Max_Stomach_b, :Sex_Ratio,:SpeciesShort,:M, :Handling_Time,:Dive_Min_Night,:Energy_density,:Min_Size, :Hatch_Survival, :MR_type, :Dive_Min_Day, :Dive_Max_Day, :Swim_velo, :Biomass,:Dive_Max_Night,:L_mat,:Resp_a, :Type)
    
    p = NamedTuple{param_names}(params)
    return plankton(data, p)
end

function initialize_individuals(arch, plank, B::Float32, sp::Int, depths::MarineDepths, capacities, dt, envi::MarineEnvironment, params::Dict, start_date::Date)
    # --- 1. Pre-calculate constants and create temporary CPU arrays ---
    grid = depths.grid
    night_profs = depths.focal_night
    depthres = grid[findfirst(grid.Name .== "depthres"), :Value]
    maxdepth = grid[findfirst(grid.Name .== "depthmax"), :Value]
    lonmax = grid[findfirst(grid.Name .== "xurcorner"), :Value]
    lonmin = grid[findfirst(grid.Name .== "xllcorner"), :Value]
    latmax = grid[findfirst(grid.Name .== "yulcorner"), :Value]
    latmin = grid[findfirst(grid.Name .== "yllcorner"), :Value]
    mean_lat_rad = deg2rad((latmin + latmax) / 2)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mean_lat_rad)
    area_km2 = abs(latmax - latmin) * km_per_deg_lat * abs(lonmax - lonmin) * km_per_deg_lon
    target_b = B * 1e6 * area_km2
    school_size = plank.p.School_Size[2][sp]
    max_size = plank.p.Max_Size[2][sp]
    min_size = plank.p.Min_Size[2][sp]

    cpu_lengths, cpu_biomass_ind, cpu_biomass_school,cpu_biomass_init = Float32[], Float32[], Float32[], Float32[]
    
    # --- 2. Generate core data on the CPU until target biomass is met ---
    current_b = 0.0
    μ, σ = lognormal_params_from_minmax(min_size,max_size)
    dist = LogNormal(μ, σ)
    while current_b < target_b
        new_length = rand(dist)
        while new_length > max_size; new_length = rand(dist); end
        ind_biomass = plank.p.LWR_a[2][sp] * (new_length / 10)^plank.p.LWR_b[2][sp]
        school_biomass = ind_biomass * school_size
        push!(cpu_lengths, new_length)
        push!(cpu_biomass_ind, ind_biomass)
        push!(cpu_biomass_school, school_biomass)
        push!(cpu_biomass_init, school_biomass)
        current_b += school_biomass
    end

    n_agents = length(cpu_lengths)
    current_maxN = length(plank.data.x)

    if n_agents > current_maxN
        @info "Initial population ($n_agents) exceeds initial capacity ($current_maxN). Resizing buffer..."
        new_agents = Int(ceil(n_agents * 1.2)) #Creates 20% more agents than necessary
        plank = construct_individuals(arch, params, new_agents)
    end

    # --- 3. Generate remaining data in efficient, vectorized calls on the CPU ---
    if n_agents > 0
        cpu_abundance = fill(Float32(school_size), n_agents)

        # --- Generate unique IDs for the initial population ---
        cpu_unique_ids = zeros(Int64, n_agents)
        date_int = parse(Int, Dates.format(start_date, "mmddyy"))
        species_counter = 0

        for i in 1:n_agents
            species_counter += 1
            # Formula for CCCCSSMMDDYY format
            cpu_unique_ids[i] = (Int64(species_counter) * 100000000) + (Int64(sp) * 1000000) + Int64(date_int)
        end
        
        # Land mask for placement. Historically read from envi.data["bathymetry"];
        # it is optional now (initial_ind_placement actually keys placement off
        # capacity>0, which already excludes land because the driver layers are
        # NODATA/NaN there). If a "bathymetry" layer is present (e.g. a static
        # bathymetry.asc loaded via the env XML) it is used; otherwise we default to
        # an all-water mask so a missing layer never aborts the run.
        if haskey(envi.data, "bathymetry")
            land_mask = coalesce.(Array(envi.data["bathymetry"]), 0.0) .> 0
        else
            land_mask = falses(size(capacities, 1), size(capacities, 2))
        end
        res = initial_ind_placement(Array(capacities), sp, grid, n_agents, 1, land_mask)
        
        cpu_x, cpu_y, cpu_pool_x, cpu_pool_y = res.lons, res.lats, res.grid_x, res.grid_y
        cpu_z = gaussmix(n_agents, night_profs[sp, "mu1"], night_profs[sp, "mu2"], night_profs[sp, "mu3"], 
                                 night_profs[sp, "sigma1"], night_profs[sp, "sigma2"], night_profs[sp, "sigma3"], 
                                 night_profs[sp, "lambda1"], night_profs[sp, "lambda2"])
        cpu_z = clamp.(cpu_z, 1.0, maxdepth)
        cpu_pool_z = max.(1, ceil.(Int, cpu_z ./ (maxdepth / depthres)))
        cpu_pool_z = clamp.(cpu_pool_z, 1, Int(depthres))
        cpu_mature = min.(1.0, cpu_lengths ./ plank.p.L_mat[2][sp])
        cpu_vis_prey = visual_range_preys_init(cpu_lengths, cpu_z, plank.p.Min_Prey[2][sp], plank.p.Max_Prey[2][sp], n_agents) .* dt
        
        # --- 4. Copy all data from CPU arrays to the target device (CPU or GPU) in one batch ---
        copyto!(plank.data.unique_id, 1, cpu_unique_ids, 1, n_agents)
        copyto!(plank.data.length, 1, cpu_lengths, 1, n_agents)
        copyto!(plank.data.biomass_ind, 1, cpu_biomass_ind, 1, n_agents)
        copyto!(plank.data.biomass_school, 1, cpu_biomass_school, 1, n_agents)
        copyto!(plank.data.biomass_init, 1, cpu_biomass_init, 1, n_agents)
        copyto!(plank.data.abundance, 1, cpu_abundance, 1, n_agents)
        copyto!(plank.data.x, 1, cpu_x, 1, n_agents)
        copyto!(plank.data.y, 1, cpu_y, 1, n_agents)
        copyto!(plank.data.z, 1, cpu_z, 1, n_agents)
        copyto!(plank.data.pool_x, 1, cpu_pool_x, 1, n_agents)
        copyto!(plank.data.pool_y, 1, cpu_pool_y, 1, n_agents)
        copyto!(plank.data.pool_z, 1, cpu_pool_z, 1, n_agents)
        copyto!(plank.data.mature, 1, cpu_mature, 1, n_agents)
        copyto!(plank.data.vis_prey, 1, cpu_vis_prey, 1, n_agents)
        
        # Initialize other fields
        @views plank.data.alive[1:n_agents] .= 1.0
        @views plank.data.generation[1:n_agents] .= 1
        @views plank.data.energy[1:n_agents] .= plank.data.biomass_ind[1:n_agents] .* plank.data.abundance[1:n_agents] .* plank.p.Energy_density[2][sp] .* 0.2
        # Portable RNG: draw on the host and copy to the device (works on CPU/GPU).
        copyto!(plank.data.gut_fullness, 1, rand(Float32, n_agents), 1, n_agents)
        @views plank.data.age[1:n_agents] .= plank.p.Larval_Duration[2][sp]+1
        
        # Set remaining unused slots to non-alive
        @views plank.data.alive[n_agents+1:end] .= 0.0
    end
    return plank, species_counter
end

function generate_individuals(params::Dict, arch::Architecture, Nsp::Int32, B::Vector{Float32}, maxN::Int64, depths::MarineDepths, capacities, dt::Int32, envi::MarineEnvironment, start_date::Date)
    plank_names = Symbol[]
    plank_data=[]

    daily_birth_counters = zeros(Int,Nsp)

    for i in 1:Nsp
        name = Symbol("sp"*string(i))
        plank = construct_individuals(arch, params, maxN)
        plank, species_counter = initialize_individuals(arch, plank, B[i], i, depths, capacities, dt, envi, params, start_date)
        daily_birth_counters[i] = species_counter
        push!(plank_names, name)
        push!(plank_data, plank)
    end

    planks = NamedTuple{Tuple(plank_names)}(plank_data)
    return individuals(planks), daily_birth_counters
end

# ===================================================================
# Resource Creation and Dynamics
# ===================================================================

@kernel function initialize_resources_kernel!(
    resource_biomass, 
    resource_capacity, 
    capacities, 
    total_capacity_targets,
    total_suitability_sum,
    normalized_vertical_profiles,
    n_spec
)
    lon, lat, depth_idx, res_sp = @index(Global, NTuple)

    total_K = total_capacity_targets[res_sp]
    total_sum = total_suitability_sum[res_sp]
    
    if total_K > 0 && total_sum > 0
        avg_capacity_local = 0.0f0
        for month in 1:size(capacities, 3)
            avg_capacity_local += capacities[lon, lat, month, res_sp + n_spec]
        end
        avg_capacity_local /= size(capacities, 3)

        horizontal_proportion = avg_capacity_local / total_sum
        column_capacity = total_K * horizontal_proportion
        vertical_proportion = normalized_vertical_profiles[depth_idx, res_sp]
        
        # 1. Calculate the stable carrying capacity for this cell
        final_capacity = column_capacity * vertical_proportion
        
        if final_capacity > 0
            # 2. Set the initial biomass to a fraction of the capacity (e.g., 50%)
            initial_biomass_fraction = 0.5f0
            final_biomass = final_capacity * initial_biomass_fraction
            
            resource_biomass[lon, lat, depth_idx, res_sp] = final_biomass
            resource_capacity[lon, lat, depth_idx, res_sp] = final_capacity
        end
    end
end

function initialize_resources(
    traits::DataFrame, n_spec::Int32, n_resource::Int32, 
    depths::MarineDepths, capacities::AbstractArray, arch::Architecture
)
    grid = depths.grid
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    
    resource_biomass = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_resource))
    resource_capacity = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_resource))
    
    lonmax = grid[findfirst(grid.Name .== "xurcorner"), :Value][1]
    lonmin = grid[findfirst(grid.Name .== "xllcorner"), :Value][1]
    latmax = grid[findfirst(grid.Name .== "yulcorner"), :Value][1]
    latmin = grid[findfirst(grid.Name .== "yllcorner"), :Value][1]
    mean_lat_rad = deg2rad((latmin + latmax) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mean_lat_rad)
    area_km2 = abs(latmax - latmin) * km_per_deg_lat * abs(lonmax - lonmin) * km_per_deg_lon

    # The 'Biomass' column is now treated as the total carrying capacity (K) in mt/km^2
    total_capacity_targets_cpu = Float32.(traits.Biomass .* area_km2 .* 1e6)

    total_suitability_sum_cpu = zeros(Float32, n_resource)
    capacities_cpu = Array(capacities)
    for res_sp in 1:n_resource
        avg_capacity_map = mean(capacities_cpu[:, :, :, res_sp + n_spec], dims=3)[:,:,1]
        total_suitability_sum_cpu[res_sp] = sum(avg_capacity_map)
    end
    # --- 1. PREPARE DISTRIBUTION DATA ON CPU ---

    total_capacity_sum_cpu = zeros(Float32, n_resource)
    capacities_cpu = Array(capacities)
    for res_sp in 1:n_resource
        avg_capacity_map = mean(capacities_cpu[:, :, :, res_sp + n_spec], dims=3)[:,:,1]
        total_capacity_sum_cpu[res_sp] = sum(avg_capacity_map)
    end

    max_depth = Int(grid[grid.Name .== "depthmax", :Value][1])
    depth_res_m = max_depth / depthres
    depths_m = [(d - 0.5f0) * depth_res_m for d in 1:depthres] # Center of each depth bin
    
    # Create a matrix to hold the profile for each resource species
    vertical_profiles_cpu = zeros(Float32, depthres, n_resource)
    night_profs = depths.resource_night # Assumes this field exists

    for res_sp in 1:n_resource
        # Use a Gaussian mixture model for the profile, similar to agent initialization
        mu1 = night_profs[res_sp, "mu1"]; mu2 = night_profs[res_sp, "mu2"]
        sigma1 = night_profs[res_sp, "sigma1"]; sigma2 = night_profs[res_sp, "sigma2"]
        lambda1 = night_profs[res_sp, "lambda1"]
        
        # Calculate the profile for this species
        profile = lambda1 .* pdf.(Normal(mu1, sigma1), depths_m) .+ (1-lambda1) .* pdf.(Normal(mu2, sigma2), depths_m)
        
        # Normalize the profile so it sums to 1
        sum_prof = sum(profile)
        if sum_prof > 0
            vertical_profiles_cpu[:, res_sp] .= profile ./ sum_prof
        end
    end

    # --- 2. UPLOAD DATA TO GPU ---
    total_capacity_targets = array_type(arch)(total_capacity_targets_cpu)
    total_suitability_sum = array_type(arch)(total_suitability_sum_cpu)
    normalized_vertical_profiles = array_type(arch)(vertical_profiles_cpu)

    # --- 3. LAUNCH THE KERNEL ---
    kernel_dims = (lonres, latres, depthres, Int64(n_resource))
    kernel! = initialize_resources_kernel!(device(arch), (8, 8, 4, 1), kernel_dims)
    kernel!(
        resource_biomass, 
        resource_capacity, 
        capacities, 
        total_capacity_targets,
        total_suitability_sum,
        normalized_vertical_profiles,
        n_spec
    )
    
    KernelAbstractions.synchronize(device(arch))
    return (biomass = resource_biomass, capacity = resource_capacity)
end

function resource_growth!(model::MarineModel, current_date)
    arch = model.arch
    
    # --- 1. Get the seasonal multipliers for the current month ---
    # PERFORMANCE NOTE: CSV.read is extremely slow to do every timestep. 
    # Consider loading this into model.files or model.environment during setup_and_run_model!
    seasonality_df = cached_csv_for(model.files, "growth_seasonality")
    
    # Get the multipliers for the current month
    current_month_name = Dates.monthname(current_date)
    monthly_multipliers = seasonality_df[!, current_month_name]

    # --- 2. Calculate the adjusted growth rates ---
    # Get the base annual growth rates from the main trait file
    annual_growth_rates = model.resource_trait.Growth
    
    # Apply the seasonal multiplier to the base rates
    adjusted_annual_rates = annual_growth_rates .* monthly_multipliers
    
    # Convert the adjusted annual rate to a per-timestep rate
    minutes_per_year = 365.0f0 * 1440.0f0
    per_timestep_rates = array_type(arch)(Float32.(adjusted_annual_rates ./ (minutes_per_year / model.dt)))

    # --- 2b. Primary-production multiplier on carrying capacity K --------------
    # When time-varying ASC forcing is active and an `npp` (LayerRelPP) variable is
    # present, scale each cell's carrying capacity by the normalised NPP ratio for
    # the current month. Productivity sets the standing stock a cell can support, so
    # NPP multiplies K (not the intrinsic rate r); a ratio of 1 reproduces the
    # trait-table baseline. If no ASC/NPP forcing is active this is a no-op (the
    # kernel receives an all-ones map) and behaviour is identical to before.
    lonres, latres = size(model.resources.biomass, 1), size(model.resources.biomass, 2)
    npp_mult_cpu = current_npp_multiplier(model, current_date)   # lon×lat or nothing
    if npp_mult_cpu === nothing
        npp_mult = array_type(arch)(ones(Float32, lonres, latres))
    else
        npp_mult = array_type(arch)(Float32.(npp_mult_cpu))
    end

    # --- 3. Launch the kernel ---
    kernel! = resource_growth_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel!(model.resources.biomass, model.resources.capacity, per_timestep_rates, npp_mult)
    KernelAbstractions.synchronize(device(arch))
end

# Kernel for resource growth.
# `npp_mult` is a (lon,lat) multiplier applied to the carrying capacity K, so the
# logistic attractor itself tracks productivity in space and time.
@kernel function resource_growth_kernel!(biomass_grid, capacity_grid, per_timestep_rates, npp_mult)
    lon, lat, depth, sp = @index(Global, NTuple)
    
    @inbounds biomass = biomass_grid[lon, lat, depth, sp]
    @inbounds base_capacity = capacity_grid[lon, lat, depth, sp]
    @inbounds rate = per_timestep_rates[sp]
    @inbounds m = npp_mult[lon, lat]
    capacity = base_capacity * (isfinite(m) ? m : 1.0f0)

    if capacity > 0.0f0
        if biomass > 1f-9
            # Exact analytical solution to the logistic differential equation.
            # This is unconditionally stable and will never overshoot or oscillate, even if 'rate' is massive.
            @inbounds biomass_grid[lon, lat, depth, sp] = capacity / (1.0f0 + ((capacity - biomass) / biomass) * exp(-rate))
        else
            # Seed heavily depleted populations to prevent complete extinction lock
            @inbounds biomass_grid[lon, lat, depth, sp] = 1f-9
        end
    end
end

# ===================================================================
# Reproduction System
# ===================================================================
function calculate_new_offspring_cpu(p_cpu, parent_data, repro_energy_list, spawn_val, sp, current_date::Date, daily_births::Int;
                                     parent_temp::AbstractVector=Float32[],
                                     local_spawner_biomass::AbstractVector=Float32[],
                                     T_opt::Float64=NaN, T_sd::Float64=NaN,
                                     dd_beta::Float64=0.0, sigma_R::Float64=0.0,
                                     rng=Random.default_rng())
    for i in 1:length(parent_data.biomass_ind)
        if !isfinite(parent_data.biomass_ind[i]) || parent_data.biomass_ind[i] <= 0.0f0
            @warn """
            DIAGNOSTIC: Invalid parent biomass detected before reproduction.
            Parent Index: $i, Species: $sp, Biomass: $(parent_data.biomass_ind[i])
            Excluding this parent from reproduction.
            """
            repro_energy_list[i] = 0.0f0 # Exclude this parent by zeroing its reproductive energy
        end
    end

    # --- 1. Calculate the potential number of eggs for each parent (as a Float) ---
    egg_volume = 0.15f0 .* parent_data.biomass_ind .^ 0.14f0
    egg_energy = 2.15f0 .* egg_volume .^ 0.77f0
    spent_energy = repro_energy_list .* spawn_val
    sex_ratio = p_cpu.Sex_Ratio.second[sp]
    hatch_survival = p_cpu.Hatch_Survival.second[sp]
    
    num_eggs_per_parent_float = spent_energy ./ (egg_energy .+ 1.0f-9) .* sex_ratio .* hatch_survival

    # ===================================================================
    # Environmentally-driven, density-dependent, and stochastic recruitment.
    # ===================================================================
    # The base calculation above is the deterministic egg-energy budget. We now
    # convert egg production into *realised* recruits by multiplying through three
    # survival/process terms, each of which is a no-op under its default so the
    # original behaviour is recovered when recruitment parameters are absent.
    #
    #  (1) TEMPERATURE-DEPENDENT EARLY-LIFE SURVIVAL (environmental driver).
    #      A Gaussian thermal window for egg/larval survival centred on T_opt with
    #      width T_sd, evaluated at each parent's ambient temperature. This is the
    #      standard "thermal performance / spawning-habitat suitability" form used
    #      for recruitment in temperature-driven fish models (e.g. Pörtner & Peck
    #      2010 J. Fish Biol.; Hare et al. 2010 for SST-recruitment links; SEDAR
    #      King Mackerel work links recruitment to spring/summer SST). For King
    #      Mackerel a spawning/larval optimum near 26-28 C is appropriate.
    #
    #  (2) DENSITY-DEPENDENT COMPENSATION (emergent Beverton-Holt).
    #      Early-life survival declines with LOCAL conspecific spawner biomass as
    #      1/(1 + dd_beta * local_spawner_biomass). Because egg production is ~linear
    #      in spawner energy (≈ proportional to local SSB) while per-egg survival
    #      falls as local SSB rises, the realised recruits-vs-SSB curve saturates:
    #      this *is* Beverton-Holt compensation, but it EMERGES from local crowding
    #      rather than being imposed as a stock-level function (Beverton & Holt 1957;
    #      the local-density formulation follows the spatial/IBM compensation logic
    #      in e.g. Rose et al. 2001 and the "emergent density dependence" discussion
    #      in DeAngelis & Grimm 2014). dd_beta=0 disables it.
    #
    #  (3) RECRUITMENT PROCESS ERROR (interannual variability).
    #      A single lognormal deviation per spawning event, exp(sigma_R*z - sigma_R^2/2)
    #      with z~N(0,1), bias-corrected to mean 1. This is the standard
    #      log-normal recruitment deviation of stock-assessment practice (Methot &
    #      Wetzel 2013, Stock Synthesis; Thorson et al. 2014 on sigma_R), and is the
    #      single highest-leverage change for producing realistic dynamic SSB/recruit
    #      trends. sigma_R=0 disables it.
    n_parents = length(num_eggs_per_parent_float)

    # (3) one cohort-level deviation, shared across this spawning event
    rec_dev = sigma_R > 0.0 ? Float32(exp(sigma_R * randn(rng) - 0.5 * sigma_R^2)) : 1.0f0

    @inbounds for i in 1:n_parents
        surv = 1.0f0
        # (1) thermal survival
        if isfinite(T_opt) && isfinite(T_sd) && T_sd > 0 && i <= length(parent_temp)
            Ti = parent_temp[i]
            if isfinite(Ti)
                surv *= Float32(exp(-0.5 * ((Ti - T_opt) / T_sd)^2))
            end
        end
        # (2) local density-dependent survival
        if dd_beta > 0.0 && i <= length(local_spawner_biomass)
            sb = local_spawner_biomass[i]
            if isfinite(sb) && sb > 0
                surv *= Float32(1.0 / (1.0 + dd_beta * sb))
            end
        end
        num_eggs_per_parent_float[i] *= surv * rec_dev
    end

    if any(!isfinite, num_eggs_per_parent_float)
        invalid_indices = findall(!isfinite, num_eggs_per_parent_float)
        @warn """
        DIAGNOSTIC: Non-finite number (Inf/NaN) detected in egg calculation.
        Species: $sp
        Problematic Parent Indices: $invalid_indices
        Problematic Values: $(num_eggs_per_parent_float[invalid_indices])
        """
        # Replace non-finite values with 0 to prevent them from propagating
        num_eggs_per_parent_float[invalid_indices] .= 0.0f0
    end

    # --- 2. Identify which parents can produce at least one whole egg ---
    producing_parents_mask = num_eggs_per_parent_float .>= 1.0
    if !any(producing_parents_mask); return nothing, daily_births; end

    # --- 3. Get the indices and egg counts for ONLY the valid parents ---
    parent_indices = (1:length(parent_data.x))[producing_parents_mask]
    num_eggs_to_create = floor.(Int, num_eggs_per_parent_float[producing_parents_mask])

    # --- 4. The number of new agents is the number of successful parents ---
    total_new_agents = length(parent_indices)
    if total_new_agents == 0; return nothing, daily_births; end

    # --- 5. Initialize arrays for the new agents ---
    new_unique_ids = zeros(Int64, total_new_agents)
    new_x = zeros(Float32, total_new_agents)
    new_y = zeros(Float32, total_new_agents)
    new_z = zeros(Float32, total_new_agents)
    new_pool_x = zeros(Int, total_new_agents)
    new_pool_y = zeros(Int, total_new_agents)
    new_pool_z = zeros(Int, total_new_agents)
    new_generation = zeros(Int, total_new_agents)
    new_abundance = zeros(Float32, total_new_agents)
    new_biomass_school = zeros(Float32, total_new_agents)
    new_energy = zeros(Float32, total_new_agents)
    
    date_int = parse(Int, Dates.format(current_date, "mmddyy"))

    # --- 6. Loop through each PRODUCING PARENT to create one new agent ---
    for (i, parent_idx) in enumerate(parent_indices)
        px, py, pz = parent_data.x[parent_idx], parent_data.y[parent_idx], parent_data.z[parent_idx]
        if !isfinite(px) || !isfinite(py) || !isfinite(pz)
            @warn """
            DIAGNOSTIC: Invalid parent coordinates detected! Offspring would inherit bad data.
            Parent Index: $parent_idx, Species: $sp
            Parent Coords (x,y,z): ($px, $py, $pz)
            Skipping creation of this offspring.
            """
            # Set values to zero so this new agent is effectively inert
            new_abundance[i] = 0.0f0 
            new_biomass_school[i] = 0.0f0
            new_energy[i] = 0.0f0
            # We use continue to skip the rest of the loop for this specific parent
            continue
        end

        # Get the number of eggs for this specific parent
        num_eggs = num_eggs_to_create[i]
        
        # A. Generate a unique ID for this new agent
        new_id = (Int64(daily_births + i) * 100000000) + (Int64(sp) * 1000000) + Int64(date_int)
        new_unique_ids[i] = new_id

        # B. Inherit location and generation from the (now validated) parent
        new_x[i] = px
        new_y[i] = py
        new_z[i] = pz
        new_pool_x[i] = parent_data.pool_x[parent_idx]
        new_pool_y[i] = parent_data.pool_y[parent_idx]
        new_pool_z[i] = parent_data.pool_z[parent_idx]
        new_generation[i] = parent_data.generation[parent_idx] + 1

        # C. The new agent's abundance is the number of eggs from THIS parent
        new_abundance[i] = Float32(num_eggs)
        
        # D. Calculate biomass and energy for this new agent based on its abundance
        larval_size = Float32(p_cpu.Larval_Size.second[sp])
        lwr_a = p_cpu.LWR_a.second[sp]
        lwr_b = p_cpu.LWR_b.second[sp]
        energy_ed = p_cpu.Energy_density.second[sp]
        
        biomass_ind = lwr_a * (larval_size / 10.0f0) ^ lwr_b
        biomass_sch = biomass_ind * num_eggs
        
        new_biomass_school[i] = biomass_sch
        new_energy[i] = biomass_sch * energy_ed * 0.2f0
    end

    daily_births += total_new_agents

    if any(!isfinite, new_biomass_school) || any(!isfinite, new_energy)
        @error """
        FATAL DIAGNOSTIC: Attempting to return new agents with invalid biomass or energy!
        Species: $sp
        Halting to prevent data corruption.
        """
        # This is a critical error, so we stop the simulation
        error("Invalid offspring data generated in calculate_new_offspring_cpu.")
    end

    # --- 7. Return the properties for all new agents ---
    larval_size = Float32(p_cpu.Larval_Size.second[sp])
    lwr_a = p_cpu.LWR_a.second[sp]
    lwr_b = p_cpu.LWR_b.second[sp]
    
    return (
        unique_id = new_unique_ids,
        x = new_x, y = new_y, z = new_z,
        length = fill(larval_size, total_new_agents),
        abundance = new_abundance,
        biomass_ind = fill(lwr_a * (larval_size / 10.0f0) ^ lwr_b, total_new_agents),
        biomass_school = new_biomass_school,
        energy = new_energy,
        gut_fullness = fill(1.0f0, total_new_agents),
        age = zeros(Float32, total_new_agents),
        pool_x = new_pool_x, pool_y = new_pool_y, pool_z = new_pool_z,
        generation = new_generation,
        cost = zeros(Float32, total_new_agents),
        active = zeros(Float32, total_new_agents),
        ration_biomass = zeros(Float32, total_new_agents),
        ration_energy = zeros(Float32, total_new_agents),
        vis_prey = zeros(Float32, total_new_agents),
        target_pool_x = zeros(Int32, total_new_agents),
        target_pool_y = zeros(Int32, total_new_agents),
        mature = zeros(Float32, total_new_agents),
        cell_id = zeros(Int32, total_new_agents),
        sorted_id = zeros(Int32, total_new_agents),
        repro_energy = zeros(Float32, total_new_agents),
        best_prey_dist = fill(Inf32, total_new_agents),
        best_prey_idx = zeros(Int32, total_new_agents),
        best_prey_sp = zeros(Int32, total_new_agents),
        best_prey_type = zeros(Int32, total_new_agents),
        successful_ration = zeros(Float32, total_new_agents),
        temp_idx = zeros(Int32, total_new_agents),
        cell_starts = zeros(Int32, total_new_agents),
        cell_ends = zeros(Int32, total_new_agents),
        mig_status = zeros(Int32, total_new_agents),
        target_z = zeros(Float32, total_new_agents),
        interval = zeros(Float32, total_new_agents)
    ), daily_births
end

"""
    merge_agents!(data_cpu, sp, thresholds, dt, school_size_trait)

Refined consolidation:
1. Groups by cell, size bin, cohort, and timestep.
2. Respects a maximum school size (trait School_Size).
3. Merges smaller schools into larger anchors first.
"""
function merge_agents!(data_cpu, sp::Int, thresholds::Matrix{Float32}, dt::Int, school_size_trait::Int64)
    alive_mask = data_cpu.alive .== 1.0
    if count(alive_mask) < 2 return end
    
    dt_days = dt / 1440.0
    max_school_size = school_size_trait
    
    # 1. Group indices
    groups = Dict{Tuple{Int, Int, Int, Int, Int, Int}, Vector{Int}}()
    
    for i in findall(alive_mask)
        bin = find_species_size_bin(data_cpu.length[i], sp, thresholds)
        age_bin = floor(Int, data_cpu.age[i] / dt_days)
        
        # Key: location, size, generation, and birth-step
        key = (data_cpu.pool_x[i], data_cpu.pool_y[i], data_cpu.pool_z[i], bin, data_cpu.generation[i], age_bin)
        
        if !haskey(groups, key)
            groups[key] = Int[]
        end
        push!(groups[key], i)
    end
    
    # 2. Perform Merging
    for (key, idxs) in groups
        if length(idxs) > 1
            # Largest school acts as the 'anchor'
            sort!(idxs, by = i -> data_cpu.abundance[i], rev = true)
            target = idxs[1]
            
            # Skip if anchor is already nearly full
            if data_cpu.abundance[target] >= max_school_size * 0.95 continue end
            
            for k in 2:length(idxs)
                source = idxs[k]
                
                # Check if target can accommodate source without exceeding School_Size
                if data_cpu.abundance[target] + data_cpu.abundance[source] <= max_school_size
                    
                    n_t = Float32(data_cpu.abundance[target])
                    n_s = Float32(data_cpu.abundance[source])
                    total_n = n_t + n_s
                    
                    if total_n <= 0f0 continue end
                    
                    # Weights for intensive properties
                    w_t = n_t / total_n
                    w_s = n_s / total_n
                    
                    # --- A. INTENSIVE PROPERTIES (Weighted Averages) ---
                    # Spatial coordinates
                    data_cpu.x[target] = data_cpu.x[target] * w_t + data_cpu.x[source] * w_s
                    data_cpu.y[target] = data_cpu.y[target] * w_t + data_cpu.y[source] * w_s
                    data_cpu.z[target] = data_cpu.z[target] * w_t + data_cpu.z[source] * w_s
                    
                    # Biological traits
                    data_cpu.length[target] = data_cpu.length[target] * w_t + data_cpu.length[source] * w_s
                    data_cpu.age[target] = data_cpu.age[target] * w_t + data_cpu.age[source] * w_s
                    data_cpu.biomass_ind[target] = data_cpu.biomass_ind[target] * w_t + data_cpu.biomass_ind[source] * w_s
                    data_cpu.gut_fullness[target] = data_cpu.gut_fullness[target] * w_t + data_cpu.gut_fullness[source] * w_s
                    data_cpu.mature[target] = data_cpu.mature[target] * w_t + data_cpu.mature[source] * w_s
                    data_cpu.vis_prey[target] = data_cpu.vis_prey[target] * w_t + data_cpu.vis_prey[source] * w_s
                    
                    # State timers/goals
                    data_cpu.target_z[target] = data_cpu.target_z[target] * w_t + data_cpu.target_z[source] * w_s
                    data_cpu.interval[target] = data_cpu.interval[target] * w_t + data_cpu.interval[source] * w_s

                    # --- B. EXTENSIVE PROPERTIES (Simple Sums) ---
                    # Abundance and Biomass
                    data_cpu.abundance[target] += data_cpu.abundance[source]
                    data_cpu.biomass_school[target] += data_cpu.biomass_school[source]
                    data_cpu.biomass_init[target] += data_cpu.biomass_init[source]
                    
                    # Energy and Rations
                    data_cpu.energy[target] += data_cpu.energy[source]
                    data_cpu.repro_energy[target] += data_cpu.repro_energy[source]
                    data_cpu.ration_energy[target] += data_cpu.ration_energy[source]
                    data_cpu.ration_biomass[target] += data_cpu.ration_biomass[source]
                    data_cpu.successful_ration[target] += data_cpu.successful_ration[source]
                    
                    # Costs and Activity
                    data_cpu.cost[target] += data_cpu.cost[source]
                    data_cpu.active[target] += data_cpu.active[source]
                    
                    # --- C. DEACTIVATE SOURCE ---
                    data_cpu.alive[source] = 0.0
                    data_cpu.abundance[source] = 0
                    data_cpu.biomass_school[source] = 0.0f0
                    data_cpu.energy[source] = 0.0f0
                    
                    # Reset decision variables to prevent bad pointers
                    data_cpu.best_prey_dist[source] = Inf32
                    data_cpu.best_prey_idx[source] = 0
                end
            end
        end
    end
end