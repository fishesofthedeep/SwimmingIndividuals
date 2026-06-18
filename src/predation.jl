# ===================================================================
# GPU Utility Helpers for Memory Safety
# ===================================================================

# A high-performance, GPU-safe, compile-time unrolled helper to index tuples using a runtime index.
# This prevents illegal memory accesses resulting from dynamic tuple lookup on the GPU.
@generated function gpu_get_tuple_element(t::NTuple{N, Any}, i::Integer) where {N}
    ex = quote
        if i == 1
            return t[1]
        end
    end
    for k in 2:N
        push!(ex.args, :(if i == $k; return t[$k]; end))
    end
    push!(ex.args, :(return t[1])) # Safe fallback to prevent crashes if index is out of bounds
    return ex
end

# GPU-compliant function to calculate the squared distance in meters.
@inline function haversine_distance_sq(lat1, lon1, z1, lat2, lon2, z2)
    R = 6371000.0f0  # Earth's radius in meters
    
    # Convert degrees to radians
    lat1_rad, lon1_rad = lat1 * 0.0174533f0, lon1 * 0.0174533f0
    lat2_rad, lon2_rad = lat2 * 0.0174533f0, lon2 * 0.0174533f0

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat / 2f0)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2f0)^2
    c = 2f0 * atan(sqrt(a), sqrt(1f0 - a))
    horizontal_dist = R * c

    vertical_dist = z2 - z1
    
    return horizontal_dist^2 + vertical_dist^2
end

@kernel function assign_cell_ids_kernel!(cell_id, pool_x, pool_y, pool_z, lonres, latres)
    i = @index(Global)
    if i >= 1 && i <= length(cell_id) && i <= length(pool_x) && i <= length(pool_y) && i <= length(pool_z)
        @inbounds cell_id[i] = get_cell_id(pool_x[i], pool_y[i], pool_z[i], lonres, latres)
    end
end

function build_spatial_index!(model::MarineModel)
    arch = model.arch
    g = model.depths.grid
    lonres = Int(g[g.Name .== "lonres", :Value][1])
    latres = Int(g[g.Name .== "latres", :Value][1])
    depthres = Int(g[g.Name .== "depthres", :Value][1])
    n_cells = lonres * latres * depthres

    for sp in 1:model.n_species
        agents = model.individuals.animals[sp].data
        n_agents = length(agents.x)
        if n_agents == 0; continue; end

        if length(agents.cell_starts) < n_cells
            @error "The pre-allocated 'cell_starts' array is too small for the grid size."
            return
        end

        kernel_assign = assign_cell_ids_kernel!(device(arch), 256, (n_agents,))
        kernel_assign(agents.cell_id, agents.pool_x, agents.pool_y, agents.pool_z, lonres, latres)
        KernelAbstractions.synchronize(device(arch))

        cell_id_cpu = Array(@view agents.cell_id[1:n_agents])
        order = sortperm(cell_id_cpu)
        sorted_cell_ids = cell_id_cpu[order]

        cell_starts_cpu = zeros(eltype(agents.cell_starts), n_cells)
        cell_ends_cpu   = zeros(eltype(agents.cell_ends),   n_cells)
        if n_agents > 0
            run_start = 1
            @inbounds for i in 2:n_agents
                if sorted_cell_ids[i] != sorted_cell_ids[i-1]
                    c = sorted_cell_ids[i-1]
                    if 1 <= c <= n_cells
                        cell_starts_cpu[c] = run_start
                        cell_ends_cpu[c]   = i - 1
                    end
                    run_start = i
                end
            end
            c_last = sorted_cell_ids[n_agents]
            if 1 <= c_last <= n_cells
                cell_starts_cpu[c_last] = run_start
                cell_ends_cpu[c_last]   = n_agents
            end
        end

        copyto!(@view(agents.sorted_id[1:n_agents]), eltype(agents.sorted_id).(order))
        copyto!(@view(agents.cell_starts[1:n_cells]), cell_starts_cpu)
        copyto!(@view(agents.cell_ends[1:n_cells]), cell_ends_cpu)
    end
end

# ===================================================================
# Agent Predation System
# ===================================================================

# --- STEP 1: Find Best Prey (OFT Profitability) ---
@kernel function find_best_prey_kernel!(
    best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    pred_alive, pred_x, pred_y, pred_z, pred_pool_x, pred_pool_y, pred_pool_z, pred_length, pred_vis_prey,
    all_prey_cell_starts, all_prey_cell_ends, all_prey_sorted_id, all_prey_alive, all_prey_length, all_prey_biomass_ind, all_prey_x, all_prey_y, all_prey_z,
    resource_biomass_grid, resource_trait,
    agent_energy_densities,
    pred_inds, grid_params, pred_params, pred_sp_idx::Int32
)
    j_idx = @index(Global)
    if j_idx >= 1 && j_idx <= length(pred_inds)
        pred_idx = pred_inds[j_idx]
        if pred_idx >= 1 && pred_idx <= length(pred_alive) && pred_idx <= length(pred_x) && pred_idx <= length(pred_y) && pred_idx <= length(pred_z) && pred_idx <= length(pred_pool_x) && pred_idx <= length(pred_pool_y) && pred_idx <= length(pred_pool_z) && pred_idx <= length(pred_length) && pred_idx <= length(pred_vis_prey)
            @inbounds if pred_alive[pred_idx] == 1.0f0
                my_x, my_y, my_z = pred_x[pred_idx], pred_y[pred_idx], pred_z[pred_idx]
                my_pool_x, my_pool_y, my_pool_z = pred_pool_x[pred_idx], pred_pool_y[pred_idx], pred_pool_z[pred_idx]
                
                # Calculate raw physical length bounds in millimeters 
                min_size = pred_length[pred_idx] * pred_params.min_prey_ratio
                max_size = pred_length[pred_idx] * pred_params.max_prey_ratio
                detection_radius_sq = pred_vis_prey[pred_idx]^2

                best_score = -1.0f0
                best_dist_sq = Inf32
                best_id = 0
                best_sp = 0
                best_type = 0

                for dy in -1:1, dx in -1:1
                    search_x = my_pool_x + dx
                    search_y = my_pool_y + dy
                    search_z = my_pool_z

                    # Bound checks on resource grid dimensions to prevent out-of-bounds lookups
                    if (1 <= search_x <= size(resource_biomass_grid, 1) && 
                        1 <= search_y <= size(resource_biomass_grid, 2) && 
                        1 <= search_z <= size(resource_biomass_grid, 3))
                        
                        # Search Focal Species Prey
                        for prey_sp_idx in 1:length(all_prey_cell_starts)
                            # No Cannibalism: Predators cannot target prey belonging to the same species
                            if prey_sp_idx != pred_sp_idx
                                prey_cell_starts_arr = gpu_get_tuple_element(all_prey_cell_starts, prey_sp_idx)
                                prey_cell_ends_arr   = gpu_get_tuple_element(all_prey_cell_ends, prey_sp_idx)
                                prey_sorted_id_arr   = gpu_get_tuple_element(all_prey_sorted_id, prey_sp_idx)
                                prey_alive_arr       = gpu_get_tuple_element(all_prey_alive, prey_sp_idx)
                                prey_length_arr      = gpu_get_tuple_element(all_prey_length, prey_sp_idx)
                                prey_biomass_ind_arr = gpu_get_tuple_element(all_prey_biomass_ind, prey_sp_idx)
                                prey_x_arr           = gpu_get_tuple_element(all_prey_x, prey_sp_idx)
                                prey_y_arr           = gpu_get_tuple_element(all_prey_y, prey_sp_idx)
                                prey_z_arr           = gpu_get_tuple_element(all_prey_z, prey_sp_idx)

                                prey_ed = prey_sp_idx <= length(agent_energy_densities) ? agent_energy_densities[prey_sp_idx] : 0.0f0
                                
                                cell = get_cell_id(search_x, search_y, search_z, grid_params.lonres, grid_params.latres)
                                n_cells_sp = length(prey_cell_starts_arr)
                                if 1 <= cell <= n_cells_sp
                                    c_start = prey_cell_starts_arr[cell]
                                    c_end   = prey_cell_ends_arr[cell]
                                    if c_start > 0
                                        for kk in c_start:c_end
                                            if kk >= 1 && kk <= length(prey_sorted_id_arr)
                                                k = prey_sorted_id_arr[kk]
                                                if k >= 1 && k <= length(prey_alive_arr) && k <= length(prey_length_arr) && k <= length(prey_biomass_ind_arr) && k <= length(prey_x_arr) && k <= length(prey_y_arr) && k <= length(prey_z_arr)
                                                    if prey_alive_arr[k] == 1.0f0 && min_size <= prey_length_arr[k] <= max_size

                                                        prey_x, prey_y, prey_z = prey_x_arr[k], prey_y_arr[k], prey_z_arr[k]
                                                        dist_sq = haversine_distance_sq(my_y, my_x, my_z, prey_y, prey_x, prey_z)

                                                        if dist_sq <= detection_radius_sq
                                                            # Convert to km squared so the penalty isn't astronomically huge
                                                            dist_km_sq = dist_sq / 1000000.0f0
                                                            prey_mass = prey_biomass_ind_arr[k]
                                                            profit = (prey_mass * prey_ed) / (dist_km_sq + 1.0f0)

                                                            if profit > best_score
                                                                best_score = profit
                                                                best_dist_sq = Float32(dist_sq)
                                                                best_id = k
                                                                best_sp = prey_sp_idx
                                                                best_type = 1
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        
                        # Search Resource Grid
                        if my_pool_x == search_x && my_pool_y == search_y && my_pool_z == search_z
                            for res_sp in 1:size(resource_biomass_grid, 4)
                                biomass_density = resource_biomass_grid[search_x, search_y, search_z, res_sp]
                                
                                if biomass_density > 0f0
                                    res_min = res_sp <= size(resource_trait.Min_Size, 1) ? resource_trait.Min_Size[res_sp] : 0.0f0
                                    res_max = res_sp <= size(resource_trait.Max_Size, 1) ? resource_trait.Max_Size[res_sp] : 0.0f0
                                    μ = (log(res_min) + log(res_max)) / 2f0
                                    σ = (log(res_max) - log(res_min)) / 4f0
                                    mean_size = exp(μ + 0.5f0 * σ^2)

                                    if min_size <= mean_size <= max_size
                                        a = res_sp <= size(resource_trait.LWR_a, 1) ? resource_trait.LWR_a[res_sp] : 0.0f0
                                        b = res_sp <= size(resource_trait.LWR_b, 1) ? resource_trait.LWR_b[res_sp] : 0.0f0
                                        mean_weight = a * (mean_size / 10f0)^b

                                        if mean_weight > 0f0
                                            abundance = biomass_density / mean_weight
                                            if abundance > 0f0
                                                lat_rad = my_y * 0.0174532925f0 
                                                deg2m = 111320.0f0
                                                width_m = grid_params.cell_size_deg * deg2m * cos(lat_rad)
                                                height_m = grid_params.cell_size_deg * deg2m
                                                volume_m3 = width_m * height_m * grid_params.depth_res_m
                                                
                                                volume_per_individual = volume_m3 / abundance
                                                dist_sq = cbrt(volume_per_individual)^2

                                                # Convert to km squared so the penalty isn't astronomically huge
                                                dist_km_sq = dist_sq / 1000000.0f0
                                                prey_ed = res_sp <= size(resource_trait.Energy_density, 1) ? resource_trait.Energy_density[res_sp] : 0.0f0
                                                profit = (mean_weight * prey_ed) / (dist_km_sq + 1.0f0)

                                                if profit > best_score
                                                    linear_idx = search_x + (search_y-1)*grid_params.lonres + (search_z-1)*grid_params.lonres*grid_params.latres
                                                    best_score = profit
                                                    best_dist_sq = Float32(dist_sq)
                                                    best_id = linear_idx
                                                    best_sp = res_sp
                                                    best_type = 2
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

                # Write final winner and its actual distance to global memory
                if pred_idx <= length(best_prey_dist) && pred_idx <= length(best_prey_idx) && pred_idx <= length(best_prey_sp) && pred_idx <= length(best_prey_type)
                    best_prey_dist[pred_idx] = best_dist_sq
                    best_prey_idx[pred_idx]   = best_id
                    best_prey_sp[pred_idx]    = best_sp
                    best_prey_type[pred_idx] = best_type
                end
            end
        end
    end
end

function calculate_distances_prey!(model::MarineModel, sp::Int, inds::Vector{Int32})
    arch = model.arch
    pred_data = model.individuals.animals[sp].data
    pred_params = (min_prey_ratio = model.individuals.animals[sp].p.Min_Prey[2][sp], max_prey_ratio = model.individuals.animals[sp].p.Max_Prey[2][sp])
    grid = model.depths.grid
    grid_params = (
        lonres = Int(grid[grid.Name .== "lonres", :Value][1]),
        latres = Int(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int(grid[grid.Name .== "depthres", :Value][1]),
        lon_min = grid[grid.Name .== "xllcorner", :Value][1],
        lat_min = grid[grid.Name .== "yllcorner", :Value][1],
        cell_size_deg = grid[grid.Name .== "cellsize", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / Int(grid[grid.Name .== "depthres", :Value][1])
    )
    trait_df = model.resource_trait
    # Build ONLY the numeric trait columns this kernel actually reads, each as a
    # bits-typed Float32 device array. The previous version converted *every* column
    # of the trait table -- including String name columns and Union{Missing,Float64}
    # numerics -- into device arrays and nested them in a NamedTuple. A device array
    # whose element type is not `isbits` (String, or a Union containing Missing) is a
    # classic source of asynchronous illegal-memory-access ("synchronization") errors
    # when indexed inside a kernel. Restricting to clean Float32 columns removes that
    # whole failure mode (and surfaces any bad trait data as a clear host-side error).
    _res_cols = (:Min_Size, :Max_Size, :LWR_a, :LWR_b, :Energy_density)
    resource_trait_gpu = (; (c => array_type(arch)(Float32.(trait_df[!, c])) for c in _res_cols)...)

    agent_ed_cpu = [Float32(a.p.Energy_density.second[i]) for (i, a) in enumerate(model.individuals.animals)]
    agent_energy_densities_gpu = array_type(arch)(agent_ed_cpu)

    # Convert all focal prey data struct properties to standard 1D CuArray tuples
    all_prey_cell_starts = tuple((animal.data.cell_starts for animal in model.individuals.animals)...)
    all_prey_cell_ends   = tuple((animal.data.cell_ends for animal in model.individuals.animals)...)
    all_prey_sorted_id   = tuple((animal.data.sorted_id for animal in model.individuals.animals)...)
    all_prey_alive       = tuple((animal.data.alive for animal in model.individuals.animals)...)
    all_prey_length      = tuple((animal.data.length for animal in model.individuals.animals)...)
    all_prey_biomass_ind = tuple((animal.data.biomass_ind for animal in model.individuals.animals)...)
    all_prey_x           = tuple((animal.data.x for animal in model.individuals.animals)...)
    all_prey_y           = tuple((animal.data.y for animal in model.individuals.animals)...)
    all_prey_z           = tuple((animal.data.z for animal in model.individuals.animals)...)

    kernel! = find_best_prey_kernel!(device(arch), 256, (length(inds),))
    kernel!(
        pred_data.best_prey_dist, pred_data.best_prey_idx, pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data.alive, pred_data.x, pred_data.y, pred_data.z, pred_data.pool_x, pred_data.pool_y, pred_data.pool_z, pred_data.length, pred_data.vis_prey,
        all_prey_cell_starts, all_prey_cell_ends, all_prey_sorted_id, all_prey_alive, all_prey_length, all_prey_biomass_ind, all_prey_x, all_prey_y, all_prey_z,
        model.resources.biomass, resource_trait_gpu,
        agent_energy_densities_gpu,
        array_type(arch)(inds), grid_params, pred_params, Int32(sp)
    )
    KernelAbstractions.synchronize(device(arch))
    return nothing
end

# --- STEP 2: Resolve Consumption Conflicts (CPU Referee) ---
function resolve_consumption!(model::MarineModel, sp::Int, to_eat::Vector{Int32})
    pred_data = model.individuals.animals[sp].data
    pred_char = model.individuals.animals[sp].p
    dt = model.dt
    days_in_step = dt / 1440.0f0

    # BULK COPY to CPU: Prevents scalar indexing by bringing entire arrays to host memory first
    all_best_prey_idx_cpu = Array(pred_data.best_prey_idx)
    all_best_prey_sp_cpu = Array(pred_data.best_prey_sp)
    all_best_prey_type_cpu = Array(pred_data.best_prey_type)
    all_pred_ind_biomass_cpu = Array(pred_data.biomass_ind)
    all_pred_abundance_cpu = Array(pred_data.abundance)
    all_pred_gut_full_cpu = Array(pred_data.gut_fullness)

    # Now safely slice the CPU arrays
    best_prey_idx_cpu = all_best_prey_idx_cpu[to_eat]
    best_prey_sp_cpu = all_best_prey_sp_cpu[to_eat]
    best_prey_type_cpu = all_best_prey_type_cpu[to_eat]
    pred_ind_biomass_cpu = all_pred_ind_biomass_cpu[to_eat] 
    pred_abundance_cpu = all_pred_abundance_cpu[to_eat]     
    pred_gut_full_cpu = all_pred_gut_full_cpu[to_eat]

    prey_biomass_all_cpu = [Array(animal.data.biomass_school) for animal in model.individuals.animals]
    res_biomass_cpu = Array(model.resources.biomass)

    agent_energy_densities = [animal.p.Energy_density.second[i] for (i, animal) in enumerate(model.individuals.animals)]
    resource_energy_densities = model.resource_trait.Energy_density

    successful_rations_cpu = zeros(Float64, length(pred_data.x))
    prey_claimed = Dict{Tuple{Int, Int}, Float64}()

    grid = model.depths.grid
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])

    for i in 1:length(to_eat)
        pred_idx = to_eat[i]
        prey_idx = best_prey_idx_cpu[i]
        if prey_idx == 0; continue; end

        prey_sp = best_prey_sp_cpu[i]
        prey_type = best_prey_type_cpu[i]
        
        # No Cannibalism: Double-check to reject same-species consumption targets
        if prey_type == 1 && prey_sp == sp
            continue
        end

        prey_key = (prey_type, prey_idx)
        claimed_biomass = get(prey_claimed, prey_key, 0.0)
        
        if prey_type == 1
            total_biomass = prey_biomass_all_cpu[prey_sp][prey_idx]
        else
            # DECODE: Convert the 1D linear cell index back into spatial coordinates
            # This completely avoids indexing `pred_data.pool_x` from the GPU
            z_coord = div(prey_idx - 1, lonres * latres) + 1
            rem_id = (prey_idx - 1) % (lonres * latres)
            y_coord = div(rem_id, lonres) + 1
            x_coord = rem_id % lonres + 1
            total_biomass = res_biomass_cpu[x_coord, y_coord, z_coord, prey_sp]
        end

        available_biomass = total_biomass - claimed_biomass
        if available_biomass <= 0; continue; end
        
        # Calculate stomach capacity on a per-individual basis first
        ind_biomass = max(0.0, pred_ind_biomass_cpu[i])
        my_abundance = max(1.0, Float64(pred_abundance_cpu[i]))
        ind_max_stomach_allometric = pred_char.Max_Stomach_a.second[sp] * (ind_biomass ^ pred_char.Max_Stomach_b.second[sp])
        ind_max_stomach_capped = ind_biomass 
        school_max_stomach = min(ind_max_stomach_allometric, ind_max_stomach_capped) * my_abundance
        
        current_stomach_prop = pred_gut_full_cpu[i]
        empty_stomach_biomass = max(0.0, school_max_stomach * (1.0 - current_stomach_prop))

        total_consumption_potential = empty_stomach_biomass * days_in_step
        ration_biomass = min(available_biomass, total_consumption_potential)
        
        if ration_biomass > 0
            energy_density = (prey_type == 1) ? agent_energy_densities[prey_sp] : resource_energy_densities[prey_sp]
            ration_joules = ration_biomass * energy_density

            successful_rations_cpu[pred_idx] = ration_joules
            prey_claimed[prey_key] = get(prey_claimed, prey_key, 0.0) + ration_biomass
        end
    end

    copyto!(@view(pred_data.successful_ration[1:end]), successful_rations_cpu)
end

# ===================================================================
# Predation and Consumption Resolution (GPU Kernels)
# ===================================================================

@kernel function apply_consumption_kernel!(
    alive, best_prey_dist, best_prey_idx, best_prey_sp, best_prey_type,
    x, y, z, pool_x, pool_y, pool_z, length_arr, biomass_school, gut_fullness, abundance,
    ration_energy, ration_biomass, active, successful_ration,
    all_prey_x::NTuple{N, Any}, all_prey_y::NTuple{N, Any}, all_prey_z::NTuple{N, Any},
    all_prey_length::NTuple{N,Any}, all_prey_biomass::NTuple{N, Any},
    all_prey_biomass_school::NTuple{N, Any}, all_prey_alive::NTuple{N, Any},
    all_prey_abundance::NTuple{N,Any}, all_prey_energy::NTuple{N,Any},
    agent_energy_densities,
    resource_biomass_grid,
    resource_energy_density,
    resource_trait,
    consumption_array,
    Pmort,
    size_bin_thresholds,
    swim_velo::Float32, handling_time::Float32, time_array,
    predator_sp_idx::Int, n_species::Int32,
    grid_params,
    max_stomach_a::Float32, max_stomach_b::Float32, dt::Float32
) where {N}
    pred_idx = @index(Global)

    # Bound check and check if alive safely using nested structures rather than early exits
    if (pred_idx >= 1 && 
        pred_idx <= length(alive) && 
        pred_idx <= length(best_prey_dist) && 
        pred_idx <= length(best_prey_idx) && 
        pred_idx <= length(best_prey_sp) && 
        pred_idx <= length(best_prey_type) && 
        pred_idx <= length(x) && 
        pred_idx <= length(y) && 
        pred_idx <= length(z) && 
        pred_idx <= length(pool_x) && 
        pred_idx <= length(pool_y) && 
        pred_idx <= length(pool_z) && 
        pred_idx <= length(length_arr) && 
        pred_idx <= length(biomass_school) && 
        pred_idx <= length(gut_fullness) && 
        pred_idx <= length(abundance) && 
        pred_idx <= length(ration_energy) && 
        pred_idx <= length(ration_biomass) && 
        pred_idx <= length(active) && 
        pred_idx <= length(successful_ration) && 
        pred_idx <= length(time_array) && 
        alive[pred_idx] == 1.0f0)

        @inbounds begin
            s_ration = successful_ration[pred_idx]
            
            if s_ration > 0.0f0
                dist = sqrt(best_prey_dist[pred_idx])
                time_left = time_array[pred_idx]
                
                swim_v = swim_velo * (length_arr[pred_idx] / 1000.0f0)
                time_to_prey = swim_v > 0.0f0 ? dist / swim_v : 0.0f0

                if time_to_prey <= time_left
                    time_left -= time_to_prey
                    @atomic active[pred_idx] += time_to_prey / 60.0f0

                    prey_type = best_prey_type[pred_idx]
                    prey_idx = best_prey_idx[pred_idx]
                    prey_sp_idx = best_prey_sp[pred_idx]

                    # No Cannibalism: Enforce that the prey species index must not equal the predator species index
                    is_valid_agent = (prey_type == 1 && prey_sp_idx >= 1 && prey_sp_idx <= N && prey_sp_idx != predator_sp_idx)
                    is_valid_resource = (prey_type == 2 && prey_sp_idx >= 1 && prey_sp_idx <= length(resource_energy_density))

                    if is_valid_agent || is_valid_resource
                        prey_ind_biom = 0.0f0
                        prey_energy_density = 0.0f0
                        local prey_size::Float32 = 0.0f0

                        if prey_type == 1 
                            # Safe Tuple Lookups
                            prey_biomass_arr = gpu_get_tuple_element(all_prey_biomass, prey_sp_idx)
                            prey_length_arr = gpu_get_tuple_element(all_prey_length, prey_sp_idx)
                            
                            if prey_idx >= 1 && prey_idx <= length(prey_biomass_arr) && prey_idx <= length(prey_length_arr)
                                prey_ind_biom = prey_biomass_arr[prey_idx]
                                prey_size = prey_length_arr[prey_idx]
                            end
                            prey_energy_density = prey_sp_idx <= length(agent_energy_densities) ? agent_energy_densities[prey_sp_idx] : 0.0f0
                        else 
                            res_min = prey_sp_idx <= size(resource_trait.Min_Size, 1) ? resource_trait.Min_Size[prey_sp_idx] : 0.0f0
                            res_max = prey_sp_idx <= size(resource_trait.Max_Size, 1) ? resource_trait.Max_Size[prey_sp_idx] : 0.0f0
                            μ = (log(res_min) + log(res_max)) / 2.0f0
                            σ = (log(res_max) - log(res_min)) / 4.0f0
                            prey_size = exp(μ + 0.5f0 * σ^2)
                            
                            # Apply appropriate size unit conversion (/ 10f0)
                            a_res = prey_sp_idx <= size(resource_trait.LWR_a, 1) ? resource_trait.LWR_a[prey_sp_idx] : 0.0f0
                            b_res = prey_sp_idx <= size(resource_trait.LWR_b, 1) ? resource_trait.LWR_b[prey_sp_idx] : 0.0f0
                            prey_ind_biom = a_res * (prey_size / 10.0f0)^b_res
                            prey_energy_density = prey_sp_idx <= length(resource_energy_density) ? resource_energy_density[prey_sp_idx] : 0.0f0
                        end

                        predator_abundance = Float32(abundance[pred_idx])
                        
                        handling_time_s = handling_time * 60.0f0
                        num_can_handle_f = (handling_time_s > 0.0f0 && prey_ind_biom > 0.0f0) ? 
                                           floor(Float32, (time_left / handling_time_s) * predator_abundance) : 
                                           999999999.0f0
                        
                        prey_ind_energy = prey_ind_biom * prey_energy_density
                        max_consumable_energy = num_can_handle_f * prey_ind_energy
                        effective_ration = min(s_ration, max_consumable_energy)
                        
                        if effective_ration > 0.0f0
                            effective_biomass = effective_ration / prey_energy_density

                            my_abundance = max(1.0f0, predator_abundance)
                            ind_biomass = biomass_school[pred_idx] / my_abundance
                            ind_base_stomach = max_stomach_a * (ind_biomass ^ max_stomach_b)
                            school_base_stomach = ind_base_stomach * my_abundance
                            
                            days_in_step = dt / 1440.0f0
                            max_stomach_timestep = school_base_stomach * max(1.0f0, days_in_step)

                            if effective_biomass > max_stomach_timestep
                                effective_biomass = max_stomach_timestep
                                effective_ration = effective_biomass * prey_energy_density
                            end
                            
                            time_spent = ((effective_biomass / max(1.0f-6, prey_ind_biom)) * handling_time_s) / my_abundance

                            pred_size_bin = find_species_size_bin(length_arr[pred_idx], predator_sp_idx, size_bin_thresholds)
                            prey_dim_idx = (prey_type == 1) ? prey_sp_idx : Int(n_species) + prey_sp_idx
                            prey_size_bin = find_species_size_bin(prey_size, prey_dim_idx, size_bin_thresholds)

                            px, py, pz = pool_x[pred_idx], pool_y[pred_idx], pool_z[pred_idx]
                            
                            # Comprehensive boundary safety checks before modifying multi-dimensional grid arrays
                            if (px > 0 && px <= size(consumption_array, 1) && 
                                py > 0 && py <= size(consumption_array, 2) && 
                                pz > 0 && pz <= size(consumption_array, 3) &&
                                predator_sp_idx > 0 && predator_sp_idx <= size(consumption_array, 4) &&
                                prey_dim_idx > 0 && prey_dim_idx <= size(consumption_array, 5) &&
                                pred_size_bin > 0 && pred_size_bin <= size(consumption_array, 6) &&
                                prey_size_bin > 0 && prey_size_bin <= size(consumption_array, 7))
                                
                                @atomic consumption_array[px, py, pz, predator_sp_idx, prey_dim_idx, pred_size_bin, prey_size_bin] += effective_biomass
                            end

                            if prey_type == 1
                                # Safe Tuple Lookups & Bounds Checks
                                prey_x_arr = gpu_get_tuple_element(all_prey_x, prey_sp_idx)
                                prey_y_arr = gpu_get_tuple_element(all_prey_y, prey_sp_idx)
                                prey_z_arr = gpu_get_tuple_element(all_prey_z, prey_sp_idx)
                                
                                if prey_idx >= 1 && prey_idx <= length(prey_x_arr) && prey_idx <= length(prey_y_arr) && prey_idx <= length(prey_z_arr)
                                    x[pred_idx] = prey_x_arr[prey_idx]
                                    y[pred_idx] = prey_y_arr[prey_idx]
                                    z[pred_idx] = prey_z_arr[prey_idx]
                                end
                                
                                new_px = clamp(floor(Int32, (x[pred_idx] - grid_params.lonmin) / grid_params.cell_size_deg) + 1, 1, grid_params.lonres)
                                new_py = clamp(floor(Int32, (y[pred_idx] - grid_params.latmin) / grid_params.cell_size_deg) + 1, 1, grid_params.latres)
                                pool_x[pred_idx] = new_px
                                pool_y[pred_idx] = new_py
                                
                                prey_biom_school_arr = gpu_get_tuple_element(all_prey_biomass_school, prey_sp_idx)
                                prey_energy_arr = gpu_get_tuple_element(all_prey_energy, prey_sp_idx)
                                prey_abundance_arr = gpu_get_tuple_element(all_prey_abundance, prey_sp_idx)
                                prey_alive_arr = gpu_get_tuple_element(all_prey_alive, prey_sp_idx)

                                if prey_idx >= 1 && prey_idx <= length(prey_biom_school_arr) && prey_idx <= length(prey_energy_arr) && prey_idx <= length(prey_abundance_arr) && prey_idx <= length(prey_alive_arr)
                                    prey_init_biom = prey_biom_school_arr[prey_idx]
                                    if prey_init_biom > 0.0f0
                                        energy_removed = effective_biomass * prey_energy_density
                                        inds_removed = floor(Int64, effective_biomass / max(1.0f-6, prey_ind_biom))
                                        
                                        @atomic prey_energy_arr[prey_idx] -= energy_removed
                                        @atomic prey_biom_school_arr[prey_idx] -= effective_biomass
                                        @atomic prey_abundance_arr[prey_idx] -= eltype(prey_abundance_arr)(inds_removed)
                                    end

                                    if prey_abundance_arr[prey_idx] <= 0
                                        prey_alive_arr[prey_idx] = 0.0f0
                                    end
                                end

                                if (px > 0 && px <= size(Pmort, 1) && 
                                    py > 0 && py <= size(Pmort, 2) &&
                                    pz > 0 && pz <= size(Pmort, 3) &&
                                    prey_sp_idx > 0 && prey_sp_idx <= size(Pmort, 4) &&
                                    prey_size_bin > 0 && prey_size_bin <= size(Pmort, 5))
                                    
                                    @atomic Pmort[px, py, pz, prey_sp_idx, prey_size_bin] += effective_biomass
                                end
                            else
                                # Safely deduct from the cell where the resource was eaten
                                z_coord = div(prey_idx - 1, grid_params.lonres * grid_params.latres) + 1
                                rem_id = (prey_idx - 1) % (grid_params.lonres * grid_params.latres)
                                y_coord = div(rem_id, grid_params.lonres) + 1
                                x_coord = rem_id % grid_params.lonres + 1
                                
                                if (x_coord > 0 && x_coord <= size(resource_biomass_grid, 1) &&
                                    y_coord > 0 && y_coord <= size(resource_biomass_grid, 2) &&
                                    z_coord > 0 && z_coord <= size(resource_biomass_grid, 3) &&
                                    prey_sp_idx > 0 && prey_sp_idx <= size(resource_biomass_grid, 4))
                                    
                                    @atomic resource_biomass_grid[x_coord, y_coord, z_coord, prey_sp_idx] -= effective_biomass
                                end
                            end
                        
                            @atomic gut_fullness[pred_idx] += effective_biomass / max(1.0f0, max_stomach_timestep)
                            @atomic ration_energy[pred_idx] += effective_ration
                            @atomic ration_biomass[pred_idx] += effective_biomass

                            time_array[pred_idx] = max(0.0f0, time_left - time_spent)
                        end
                    else
                        successful_ration[pred_idx] = 0.0f0
                    end
                end
                successful_ration[pred_idx] = 0.0f0
            end
        end
    end
end

function apply_consumption!(model::MarineModel, sp::Int, time::AbstractArray, outputs::MarineOutputs)
    arch = model.arch
    pred_data = model.individuals.animals[sp].data
    n_species = model.n_species

    p_row = model.individuals.animals[sp].p
    swim_velo = Float32(p_row.Swim_velo.second[sp])
    handling_time = Float32(p_row.Handling_Time.second[sp])
    max_stomach_a = Float32(p_row.Max_Stomach_a.second[sp])
    max_stomach_b = Float32(p_row.Max_Stomach_b.second[sp])

    grid = model.depths.grid
    grid_params = (
        lonres = Int(grid[grid.Name .== "lonres", :Value][1]),
        latres = Int(grid[grid.Name .== "latres", :Value][1]),
        depthres = Int(grid[grid.Name .== "depthres", :Value][1]),
        lonmin = grid[grid.Name .== "xllcorner", :Value][1],
        latmin = grid[grid.Name .== "yllcorner", :Value][1],
        cell_size_deg = grid[grid.Name .== "cellsize", :Value][1],
        depth_res_m = grid[grid.Name .== "depthmax", :Value][1] / Int(grid[grid.Name .== "depthres", :Value][1])
    )

    all_prey_x = tuple((animal.data.x for animal in model.individuals.animals)...)
    all_prey_y = tuple((animal.data.y for animal in model.individuals.animals)...)
    all_prey_z = tuple((animal.data.z for animal in model.individuals.animals)...)
    all_prey_biomass = tuple((animal.data.biomass_ind for animal in model.individuals.animals)...)
    all_prey_biomass_school = tuple((animal.data.biomass_school for animal in model.individuals.animals)...)
    all_prey_length = tuple((animal.data.length for animal in model.individuals.animals)...)
    all_prey_alive = tuple((animal.data.alive for animal in model.individuals.animals)...)
    all_prey_abundance = tuple((animal.data.abundance for animal in model.individuals.animals)...)
    all_prey_energy = tuple((animal.data.energy for animal in model.individuals.animals)...)

    agent_ed_cpu = [Float32(a.p.Energy_density.second[i]) for (i, a) in enumerate(model.individuals.animals)]
    agent_energy_densities = array_type(arch)(agent_ed_cpu)
    resource_energy_density = array_type(arch)(Float32.(model.resource_trait.Energy_density))

    res_df = model.resource_trait
    # Match calculate_distances_prey!: pass only the numeric trait columns the kernel
    # reads, as bits-typed Float32 device arrays. (Energy density is passed separately
    # as resource_energy_density, so it is not needed here.)
    _res_cols = (:Min_Size, :Max_Size, :LWR_a, :LWR_b)
    res_trait_gpu = (; (c => array_type(arch)(Float32.(res_df[!, c])) for c in _res_cols)...)

    n = length(pred_data.x)
    kernel! = apply_consumption_kernel!(device(arch), 256, (n,))

    kernel!(
        pred_data.alive, pred_data.best_prey_dist, pred_data.best_prey_idx,
        pred_data.best_prey_sp, pred_data.best_prey_type,
        pred_data.x, pred_data.y, pred_data.z, pred_data.pool_x, pred_data.pool_y, pred_data.pool_z, pred_data.length,
        pred_data.biomass_school, pred_data.gut_fullness, pred_data.abundance,
        pred_data.ration_energy, pred_data.ration_biomass, pred_data.active, pred_data.successful_ration,
        all_prey_x, all_prey_y, all_prey_z, all_prey_length, all_prey_biomass, 
        all_prey_biomass_school, all_prey_alive, all_prey_abundance, all_prey_energy,
        agent_energy_densities,
        model.resources.biomass,
        resource_energy_density,
        res_trait_gpu,
        outputs.consumption,
        outputs.Pmort,
        model.size_bin_thresholds,
        swim_velo, handling_time, time,
        sp, n_species,
        grid_params, max_stomach_a, max_stomach_b, Float32(model.dt)
    )

    KernelAbstractions.synchronize(device(arch))
    return nothing
end

# ===================================================================
# Background Resource Predation System
# ===================================================================

@inline function find_size_bin(value, bins)
    for i in 1:length(bins)
        if value < bins[i]
            return i
        end
    end
    return length(bins) + 1
end

@kernel function aggregate_prey_by_size_kernel!(
    prey_biomass_grid_by_size,
    all_prey_alive,
    all_prey_pool_x,
    all_prey_pool_y,
    all_prey_pool_z,
    all_prey_length,
    all_prey_biomass_school,
    size_bins
)
    i = @index(Global)
    for sp in 1:length(all_prey_alive)
        alive_arr = gpu_get_tuple_element(all_prey_alive, sp)
        pool_x_arr = gpu_get_tuple_element(all_prey_pool_x, sp)
        pool_y_arr = gpu_get_tuple_element(all_prey_pool_y, sp)
        pool_z_arr = gpu_get_tuple_element(all_prey_pool_z, sp)
        length_arr = gpu_get_tuple_element(all_prey_length, sp)
        biomass_school_arr = gpu_get_tuple_element(all_prey_biomass_school, sp)

        if i <= length(alive_arr) && alive_arr[i] == 1.0f0
            px, py, pz = pool_x_arr[i], pool_y_arr[i], pool_z_arr[i]
            bin_idx = find_size_bin(length_arr[i], size_bins)
            if (px > 0 && px <= size(prey_biomass_grid_by_size, 1) &&
                py > 0 && py <= size(prey_biomass_grid_by_size, 2) &&
                pz > 0 && pz <= size(prey_biomass_grid_by_size, 3) &&
                sp > 0 && sp <= size(prey_biomass_grid_by_size, 4) &&
                bin_idx > 0 && bin_idx <= size(prey_biomass_grid_by_size, 5))
                @atomic prey_biomass_grid_by_size[px, py, pz, sp, bin_idx] += biomass_school_arr[i]
            end
        end
    end
end

@kernel function aggregate_agent_props_kernel!(
    prey_biomass_grid,
    agent_total_length_grid,
    agent_abundance_grid,
    all_prey_alive,
    all_prey_pool_x,
    all_prey_pool_y,
    all_prey_pool_z,
    all_prey_biomass_school,
    all_prey_length,
    all_prey_abundance
)
    i = @index(Global)
    for sp in 1:length(all_prey_alive)
        alive_arr = gpu_get_tuple_element(all_prey_alive, sp)
        pool_x_arr = gpu_get_tuple_element(all_prey_pool_x, sp)
        pool_y_arr = gpu_get_tuple_element(all_prey_pool_y, sp)
        pool_z_arr = gpu_get_tuple_element(all_prey_pool_z, sp)
        biomass_school_arr = gpu_get_tuple_element(all_prey_biomass_school, sp)
        length_arr = gpu_get_tuple_element(all_prey_length, sp)
        abundance_arr = gpu_get_tuple_element(all_prey_abundance, sp)

        if i <= length(alive_arr) && alive_arr[i] == 1.0f0
            px, py, pz = pool_x_arr[i], pool_y_arr[i], pool_z_arr[i]
            if (px > 0 && px <= size(prey_biomass_grid, 1) &&
                py > 0 && py <= size(prey_biomass_grid, 2) &&
                pz > 0 && pz <= size(prey_biomass_grid, 3) &&
                sp > 0 && sp <= size(prey_biomass_grid, 4))
                @atomic prey_biomass_grid[px, py, pz, sp] += biomass_school_arr[i]
                @atomic agent_total_length_grid[px, py, pz, sp] += (length_arr[i] * abundance_arr[i])
                @atomic agent_abundance_grid[px, py, pz, sp] += abundance_arr[i]
            end
        end
    end
end

# --- STEP 3: Calculate Potential Mortality (OFT Weighted) ---
@kernel function calculate_potential_mortality!(
    agent_biomass_eaten_grid,
    resource_biomass,
    all_prey_cell_starts,
    all_prey_cell_ends,
    all_prey_sorted_id,
    all_prey_alive,
    all_prey_length,
    all_prey_biomass_school,
    all_prey_biomass_ind,
    consumption_array,
    size_bin_thresholds,
    n_species::Int32,
    n_resources::Int32,
    n_thresholds::Int32,
    resource_traits,
    agent_energy_densities,
    dt::Float32,
    cell_size_deg::Float32,
    depth_res_m::Float32,
    lonres::Int32,
    latres::Int32
)
    x, y, z, r = @index(Global, NTuple)

    n_resource_size_bins = n_thresholds + 1

    # Bounds check on resource grid access wrapping entire logic to eliminate banned early returns
    if (x >= 1 && x <= size(resource_biomass, 1) && 
        y >= 1 && y <= size(resource_biomass, 2) && 
        z >= 1 && z <= size(resource_biomass, 3) && 
        r >= 1 && r <= size(resource_biomass, 4) &&
        r <= size(resource_traits, 1))

        total_predator_biomass::Float32 = resource_biomass[x, y, z, r]

        if total_predator_biomass > 1.0f-9 && size(resource_traits, 2) >= 9
            pred_μ = Float32(resource_traits[r,8])
            pred_σ = Float32(resource_traits[r,9])

            for pred_bin in 1:n_resource_size_bins
                predator_dim_idx = n_species + r
                
                # Verify that row indices and column indices of size_bin_thresholds are strictly inside boundary layout
                if (pred_bin >= 1 && (pred_bin + 1) <= size(size_bin_thresholds, 1) &&
                    predator_dim_idx >= 1 && predator_dim_idx <= size(size_bin_thresholds, 2))

                    lower_bound_pred = size_bin_thresholds[pred_bin, predator_dim_idx]
                    upper_bound_pred = size_bin_thresholds[pred_bin+1, predator_dim_idx]
                    
                    # True biological mean size (lognormal)
                    predator_mean_size = exp(pred_μ + 0.5f0 * pred_σ^2)

                    proportion_in_bin = calculate_proportion_in_bin(lower_bound_pred, upper_bound_pred, pred_μ, pred_σ)
                    pred_biom::Float32 = total_predator_biomass * proportion_in_bin

                    if pred_biom > 1.0f-9
                        min_prey_ratio = resource_traits[r, 1]
                        max_prey_ratio = resource_traits[r, 2]
                        
                        min_prey_size = min_prey_ratio * predator_mean_size
                        max_prey_size = max_prey_ratio * predator_mean_size
                        
                        total_agent_biomass::Float32 = 0.0f0
                        total_agent_energy::Float32 = 0.0f0
                        total_weighted_preference::Float32 = 0.0f0 

                        # Agent Survey
                        cell_idx_surv = get_cell_id(x, y, z, lonres, latres)
                        
                        # Unrolled compile-time safe element loops over NTuple fields
                        for sp in 1:length(all_prey_cell_starts)
                            prey_cell_starts_arr = gpu_get_tuple_element(all_prey_cell_starts, sp)
                            prey_cell_ends_arr   = gpu_get_tuple_element(all_prey_cell_ends, sp)
                            prey_sorted_id_arr   = gpu_get_tuple_element(all_prey_sorted_id, sp)
                            prey_alive_arr       = gpu_get_tuple_element(all_prey_alive, sp)
                            prey_length_arr      = gpu_get_tuple_element(all_prey_length, sp)
                            prey_biomass_school_arr = gpu_get_tuple_element(all_prey_biomass_school, sp)
                            prey_biomass_ind_arr = gpu_get_tuple_element(all_prey_biomass_ind, sp)

                            n_cells_sp = length(prey_cell_starts_arr)
                            
                            if 1 <= cell_idx_surv <= n_cells_sp
                                c_start = prey_cell_starts_arr[cell_idx_surv]
                                c_end   = prey_cell_ends_arr[cell_idx_surv]
                                if c_start > 0
                                    for kk in c_start:c_end
                                        if kk >= 1 && kk <= length(prey_sorted_id_arr)
                                            i = prey_sorted_id_arr[kk]
                                            if i >= 1 && i <= length(prey_alive_arr) && prey_alive_arr[i] == 1.0f0
                                                # Guard lookups using each array's own individual bounds
                                                if i <= length(prey_length_arr) && i <= length(prey_biomass_school_arr) && i <= length(prey_biomass_ind_arr)
                                                    if prey_length_arr[i] >= min_prey_size && prey_length_arr[i] <= max_prey_size
                                                        biom = prey_biomass_school_arr[i]
                                                        energy_density_sp = sp <= length(agent_energy_densities) ? agent_energy_densities[sp] : 0.0f0
                                                        energy = biom * energy_density_sp
                                                        total_agent_biomass += biom
                                                        total_agent_energy += energy
                                                        total_weighted_preference += energy * prey_biomass_ind_arr[i]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end

                        total_resource_biomass::Float32 = 0.0f0
                        total_resource_energy::Float32 = 0.0f0
                        
                        # Resource Survey
                        for prey_r in 1:n_resources
                            # No Cannibalism: Background resource species r cannot target background resource species prey_r if they match
                            if prey_r != r && prey_r <= size(resource_biomass, 4)
                                total_prey_biomass = resource_biomass[x, y, z, prey_r]
                                if total_prey_biomass > 1.0f-9 && prey_r <= size(resource_traits, 1) && size(resource_traits, 2) >= 9
                                    prey_μ = resource_traits[prey_r, 8]
                                    prey_σ = resource_traits[prey_r, 9]
                                    energy_density_r = resource_traits[prey_r, 7]
                                    for prey_bin in 1:n_resource_size_bins
                                        prey_dim_idx = n_species + prey_r
                                        
                                        # Strict bounds-verification before indexing size thresholds array
                                        if (prey_bin >= 1 && (prey_bin + 1) <= size(size_bin_thresholds, 1) &&
                                            prey_dim_idx >= 1 && prey_dim_idx <= size(size_bin_thresholds, 2))

                                            lower_bound_prey = size_bin_thresholds[prey_bin, prey_dim_idx]
                                            upper_bound_prey = size_bin_thresholds[prey_bin+1, prey_dim_idx]
                                            prey_mean_size = lower_bound_prey + (upper_bound_prey - lower_bound_prey) * 0.5f0

                                            if prey_mean_size >= min_prey_size && prey_mean_size <= max_prey_size
                                                proportion_in_prey_bin = calculate_proportion_in_bin(lower_bound_prey, upper_bound_prey, prey_μ, prey_σ)

                                                biomass_in_prey_bin = total_prey_biomass * proportion_in_prey_bin
                                                if biomass_in_prey_bin > 0.0f0
                                                    a_prey = resource_traits[prey_r, 5]
                                                    b_prey = resource_traits[prey_r, 6]
                                                    prey_ind_biom = a_prey * (prey_mean_size / 10f0)^b_prey
                                                    
                                                    energy = biomass_in_prey_bin * energy_density_r
                                                    bin_weight = energy * prey_ind_biom
                                                    
                                                    total_resource_biomass += biomass_in_prey_bin
                                                    total_resource_energy += energy
                                                    total_weighted_preference += energy * prey_ind_biom
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end

                        total_available_energy = total_agent_energy + total_resource_energy
                        total_available_biomass = total_agent_biomass + total_resource_biomass
                        
                        if total_available_energy > 1.0f-9 && total_weighted_preference > 1.0f-9
                            avg_energy_density = total_available_energy / total_available_biomass
                            
                            max_ingestion = resource_traits[r, 3] # Daily_Ration
                            h_time = resource_traits[r, 4]        # Handling_Time (days)
                            
                            cell_volume_m3 = (cell_size_deg * 111320.0f0)^2.0f0 * depth_res_m
                            a = max_ingestion 
                            N = total_available_energy / cell_volume_m3 
                            
                            consumption_rate_per_biomass = ((a * N) / (1.0f0 + a * h_time * N)) * (dt/1440.0f0) * avg_energy_density
                            max_consumption_J_rate = max_ingestion * avg_energy_density * (dt/1440.0f0)
                            total_consumption_J = min(consumption_rate_per_biomass * pred_biom, max_consumption_J_rate * pred_biom)
                            
                            # --- Trophic Feedback Loop ---
                            conversion_efficiency = 0.15f0
                            biomass_growth = (total_consumption_J / avg_energy_density) * conversion_efficiency
                            @atomic resource_biomass[x, y, z, r] += biomass_growth

                            if total_agent_energy > 1.0f-9
                                for sp in 1:length(all_prey_cell_starts)
                                    prey_cell_starts_arr = gpu_get_tuple_element(all_prey_cell_starts, sp)
                                    prey_cell_ends_arr   = gpu_get_tuple_element(all_prey_cell_ends, sp)
                                    prey_sorted_id_arr   = gpu_get_tuple_element(all_prey_sorted_id, sp)
                                    prey_alive_arr       = gpu_get_tuple_element(all_prey_alive, sp)
                                    prey_length_arr      = gpu_get_tuple_element(all_prey_length, sp)
                                    prey_biomass_school_arr = gpu_get_tuple_element(all_prey_biomass_school, sp)
                                    prey_biomass_ind_arr = gpu_get_tuple_element(all_prey_biomass_ind, sp)

                                    n_cells_sp2 = length(prey_cell_starts_arr)
                                    if 1 <= cell_idx_surv <= n_cells_sp2
                                        c_start2 = prey_cell_starts_arr[cell_idx_surv]
                                        c_end2   = prey_cell_ends_arr[cell_idx_surv]
                                        if c_start2 > 0
                                            sp_weight = 0.0f0
                                            energy_density_sp = sp <= length(agent_energy_densities) ? agent_energy_densities[sp] : 0.0f0
                                            for kk in c_start2:c_end2
                                                if kk >= 1 && kk <= length(prey_sorted_id_arr)
                                                    i = prey_sorted_id_arr[kk]
                                                    if i >= 1 && i <= length(prey_alive_arr) && prey_alive_arr[i] == 1.0f0
                                                        if i <= length(prey_length_arr) && i <= length(prey_biomass_school_arr) && i <= length(prey_biomass_ind_arr)
                                                            if prey_length_arr[i] >= min_prey_size && prey_length_arr[i] <= max_prey_size
                                                                sp_weight += prey_biomass_school_arr[i] * energy_density_sp * prey_biomass_ind_arr[i]
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                            if sp_weight > 0.0f0
                                                prop_sp = sp_weight / total_weighted_preference
                                                biomass_eaten_sp = (total_consumption_J * prop_sp) / energy_density_sp
                                                
                                                # Bound check before writing to multidimensional tracking grid
                                                if (x > 0 && x <= size(agent_biomass_eaten_grid, 1) &&
                                                    y > 0 && y <= size(agent_biomass_eaten_grid, 2) &&
                                                    z > 0 && z <= size(agent_biomass_eaten_grid, 3) &&
                                                    r > 0 && r <= size(agent_biomass_eaten_grid, 4) &&
                                                    pred_bin > 0 && pred_bin <= size(agent_biomass_eaten_grid, 5) &&
                                                    sp > 0 && sp <= size(agent_biomass_eaten_grid, 6))
                                                    @atomic agent_biomass_eaten_grid[x, y, z, r, pred_bin, sp] += biomass_eaten_sp
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                            
                            if total_resource_energy > 1.0f-9
                                for prey_r in 1:n_resources
                                    if prey_r != r && prey_r <= size(resource_biomass, 4)
                                        total_prey_biomass = resource_biomass[x, y, z, prey_r]
                                        if total_prey_biomass > 0.0f0 && prey_r <= size(resource_traits, 1) && size(resource_traits, 2) >= 9
                                            prey_μ = resource_traits[prey_r, 8]
                                            prey_σ = resource_traits[prey_r, 9]
                                            energy_density_r = resource_traits[prey_r, 7]
                                            total_consumed_from_this_prey_r::Float32 = 0.0f0
                                            for prey_bin in 1:n_resource_size_bins
                                                prey_dim_idx = n_species + prey_r
                                                if (prey_bin >= 1 && (prey_bin + 1) <= size(size_bin_thresholds, 1) &&
                                                    prey_dim_idx >= 1 && prey_dim_idx <= size(size_bin_thresholds, 2))

                                                    lower_bound_prey = size_bin_thresholds[prey_bin, prey_dim_idx]
                                                    upper_bound_prey = size_bin_thresholds[prey_bin+1, prey_dim_idx]
                                                    prey_mean_size = lower_bound_prey + (upper_bound_prey - lower_bound_prey) * 0.5f0

                                                    if prey_mean_size >= min_prey_size && prey_mean_size <= max_prey_size
                                                        proportion_in_prey_bin = calculate_proportion_in_bin(lower_bound_prey, upper_bound_prey, prey_μ, prey_σ)

                                                        biomass_in_prey_bin = total_prey_biomass * proportion_in_prey_bin
                                                        if biomass_in_prey_bin > 0.0f0
                                                            a_prey = resource_traits[prey_r, 5]
                                                            b_prey = resource_traits[prey_r, 6]
                                                            prey_ind_biom = a_prey * (prey_mean_size / 10f0)^b_prey
                                                            
                                                            energy = biomass_in_prey_bin * energy_density_r
                                                            bin_weight = energy * prey_ind_biom
                                                            
                                                            prop = bin_weight / total_weighted_preference
                                                            consumed_energy = total_consumption_J * prop
                                                            consumed_biomass = consumed_energy / energy_density_r
                                                            total_consumed_from_this_prey_r += consumed_biomass
                                                            
                                                            predator_dim_idx = n_species + r
                                                            pred_size_bin_for_output = find_species_size_bin(predator_mean_size, predator_dim_idx, size_bin_thresholds)
                                                            prey_size_bin_for_output = find_species_size_bin(prey_mean_size, prey_dim_idx, size_bin_thresholds)
                                                            
                                                            # Comprehensive bounds safety checks on write
                                                            if (x > 0 && x <= size(consumption_array, 1) &&
                                                                y > 0 && y <= size(consumption_array, 2) &&
                                                                z > 0 && z <= size(consumption_array, 3) &&
                                                                predator_dim_idx > 0 && predator_dim_idx <= size(consumption_array, 4) &&
                                                                prey_dim_idx > 0 && prey_dim_idx <= size(consumption_array, 5) &&
                                                                pred_size_bin_for_output > 0 && pred_size_bin_for_output <= size(consumption_array, 6) &&
                                                                prey_size_bin_for_output > 0 && prey_size_bin_for_output <= size(consumption_array, 7))
                                                                
                                                                @atomic consumption_array[x, y, z, predator_dim_idx, prey_dim_idx, pred_size_bin_for_output, prey_size_bin_for_output] += consumed_biomass
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                            if total_consumed_from_this_prey_r > 0.0f0 &&
                                               x > 0 && x <= size(resource_biomass, 1) &&
                                               y > 0 && y <= size(resource_biomass, 2) &&
                                               z > 0 && z <= size(resource_biomass, 3) &&
                                               prey_r > 0 && prey_r <= size(resource_biomass, 4)
                                               
                                                @atomic resource_biomass[x, y, z, prey_r] -= min(total_consumed_from_this_prey_r, total_prey_biomass)
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# --- STEP 4: Apply and Log Mortality ---
@kernel function apply_and_log_mortality!(
    alive, pool_x, pool_y, pool_z, biomass_ind, biomass_school, energy, abundance, length_arr,
    agent_biomass_eaten_grid,
    consumption_array,
    Pmort,
    size_bin_thresholds,
    sp_idx::Int32,
    n_species::Int32
)
    i = @index(Global)

    # Thorough array-length checks on global indexing bounds using nested conditional blocks instead of illegal returns
    if (i >= 1 && 
        i <= length(alive) && 
        i <= length(pool_x) && 
        i <= length(pool_y) && 
        i <= length(pool_z) && 
        i <= length(biomass_ind) && 
        i <= length(biomass_school) && 
        i <= length(energy) && 
        i <= length(abundance) && 
        i <= length(length_arr) &&
        alive[i] == 1.0f0)

        x, y, z = pool_x[i], pool_y[i], pool_z[i]

        lonres = size(agent_biomass_eaten_grid, 1)
        latres = size(agent_biomass_eaten_grid, 2)
        depthres = size(agent_biomass_eaten_grid, 3)

        if x > 0 && x <= lonres && y > 0 && y <= latres && z > 0 && z <= depthres
            total_biomass_eaten_potential::Float32 = 0.0f0
            for r in 1:size(agent_biomass_eaten_grid, 4)
                for pred_bin in 1:size(agent_biomass_eaten_grid, 5)
                    if sp_idx <= size(agent_biomass_eaten_grid, 6)
                        total_biomass_eaten_potential += agent_biomass_eaten_grid[x, y, z, r, pred_bin, sp_idx]
                    end
                end
            end
            
            if total_biomass_eaten_potential > 0.0f0
                ind_biomass = biomass_ind[i]
                if ind_biomass > 0.0f0
                    individuals_requested_float = total_biomass_eaten_potential / ind_biomass
                    individuals_available = abundance[i]
                    inds_removed_float = min(individuals_requested_float, individuals_available)
                    inds_removed = floor(Int32, inds_removed_float)

                    if inds_removed > 0
                        actual_biomass_removed = min(inds_removed * ind_biomass, biomass_school[i])
                        
                        if biomass_school[i] > 0
                            biomass_proportion_consumed = actual_biomass_removed / biomass_school[i]
                            energy_removed = energy[i] * biomass_proportion_consumed
                            @atomic energy[i] -= energy_removed
                            @atomic biomass_school[i] -= actual_biomass_removed
                            # `abundance` is an Int64 array; subtracting a Float32 here is an
                            # atomic operand/element type mismatch that can corrupt memory.
                            # Convert the decrement to the array's own element type.
                            @atomic abundance[i] -= eltype(abundance)(inds_removed)
                        end

                        # Strictly guard sp_idx column indexing bounds of the size bin thresholds
                        if sp_idx >= 1 && sp_idx <= size(size_bin_thresholds, 2)
                            prey_bin_P = find_species_size_bin(length_arr[i], sp_idx, size_bin_thresholds)
                            
                            if (x > 0 && x <= size(Pmort, 1) && 
                                y > 0 && y <= size(Pmort, 2) && 
                                z > 0 && z <= size(Pmort, 3) && 
                                sp_idx > 0 && sp_idx <= size(Pmort, 4) && 
                                prey_bin_P > 0 && prey_bin_P <= size(Pmort, 5))
                                
                                @atomic Pmort[x, y, z, sp_idx, prey_bin_P] += actual_biomass_removed
                            end
                        end
                        
                        for r in 1:size(agent_biomass_eaten_grid, 4)
                            for pred_bin in 1:size(agent_biomass_eaten_grid, 5)
                                potential_eaten_by_this_bin = 0.0f0
                                if sp_idx <= size(agent_biomass_eaten_grid, 6)
                                    potential_eaten_by_this_bin = agent_biomass_eaten_grid[x, y, z, r, pred_bin, sp_idx]
                                end
                                
                                if potential_eaten_by_this_bin > 0.0f0
                                    proportion_from_this_bin = potential_eaten_by_this_bin / total_biomass_eaten_potential
                                    logged_biomass = actual_biomass_removed * proportion_from_this_bin
                                    
                                    predator_dim_idx = n_species + r
                                    
                                    # Safe-protect lookup of size thresholds column dimensions on background resource index layouts
                                    if (pred_bin >= 1 && (pred_bin + 1) <= size(size_bin_thresholds, 1) &&
                                        predator_dim_idx >= 1 && predator_dim_idx <= size(size_bin_thresholds, 2))

                                        local predator_mean_size::Float32
                                        lower_bound = size_bin_thresholds[pred_bin, predator_dim_idx]
                                        upper_bound = size_bin_thresholds[pred_bin+1, predator_dim_idx]
                                        predator_mean_size = lower_bound + (upper_bound - lower_bound) * 0.5f0
                                        
                                        pred_size_bin_for_output = find_species_size_bin(predator_mean_size, predator_dim_idx, size_bin_thresholds)
                                        my_length = length_arr[i]
                                        prey_size_bin = find_species_size_bin(my_length, sp_idx, size_bin_thresholds)
                                        
                                        # Safety bounds check before final tracking matrix logging
                                        if (x > 0 && x <= size(consumption_array, 1) &&
                                            y > 0 && y <= size(consumption_array, 2) &&
                                            z > 0 && z <= size(consumption_array, 3) &&
                                            predator_dim_idx > 0 && predator_dim_idx <= size(consumption_array, 4) &&
                                            sp_idx > 0 && sp_idx <= size(consumption_array, 5) &&
                                            pred_size_bin_for_output > 0 && pred_size_bin_for_output <= size(consumption_array, 6) &&
                                            prey_size_bin > 0 && prey_size_bin <= size(consumption_array, 7))
                                            
                                            @atomic consumption_array[x, y, z, predator_dim_idx, sp_idx, pred_size_bin_for_output, prey_size_bin] += logged_biomass
                                        end
                                    end
                                end
                            end
                        end

                        if abundance[i] <= 0.0f0
                            alive[i] = 0.0f0
                        end
                    end
                end
            end
        end
    end
end

# --- STEP 5: Driver Function ---
function resource_predation!(model::MarineModel, output::MarineOutputs)
    arch = model.arch
    build_spatial_index!(model)
    g = model.depths.grid
    lonres = Int(g[g.Name .== "lonres", :Value][1])
    latres = Int(g[g.Name .== "latres", :Value][1])
    depthres = Int(g[g.Name .== "depthres", :Value][1])
    n_sp = model.n_species
    n_res = Int(model.n_resource)
    rt = model.resource_trait

    if n_res == 0; return; end

    grid_params = (
        cell_size_deg = Float32(g[g.Name .== "cellsize", :Value][1]),
        depth_res_m = Float32(g[g.Name .== "depthmax", :Value][1] / depthres)
    )

    size_bin_thresholds = model.size_bin_thresholds
    n_thresholds = Int32(size(size_bin_thresholds,1)-2)
    n_resource_size_bins = n_thresholds + 1
    
    resource_total_biomass = model.resources.biomass 
    agent_biomass_eaten_grid = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_res, n_resource_size_bins, n_sp))

    trait_order = [:Min_Prey, :Max_Prey, :Daily_Ration, :Handling_Time, :LWR_a, :LWR_b, :Energy_density]
    traits_cpu = zeros(Float32, n_res, length(trait_order) + 2)
    for (i, name) in enumerate(trait_order)
        traits_cpu[:, i] .= Float32.(rt[!, name])
    end
    for r in 1:n_res
        μ, σ = lognormal_params_from_minmax(rt.Min_Size[r], rt.Max_Size[r])
        traits_cpu[r, length(trait_order) + 1] = Float32(μ)
        traits_cpu[r, length(trait_order) + 2] = Float32(σ)
    end

    resource_traits_matrix = array_type(arch)(traits_cpu)
    agent_energy_densities_gpu = array_type(arch)([Float32(animal.p.Energy_density.second[sp]) for (sp, animal) in enumerate(model.individuals.animals)])

    # Convert all focal prey data struct properties to standard NTuples of standard 1D CuArrays
    all_prey_cell_starts = tuple((animal.data.cell_starts for animal in model.individuals.animals)...)
    all_prey_cell_ends   = tuple((animal.data.cell_ends for animal in model.individuals.animals)...)
    all_prey_sorted_id   = tuple((animal.data.sorted_id for animal in model.individuals.animals)...)
    all_prey_alive       = tuple((animal.data.alive for animal in model.individuals.animals)...)
    all_prey_length      = tuple((animal.data.length for animal in model.individuals.animals)...)
    all_prey_biomass_school = tuple((animal.data.biomass_school for animal in model.individuals.animals)...)
    all_prey_biomass_ind = tuple((animal.data.biomass_ind for animal in model.individuals.animals)...)

    kernel_calc = calculate_potential_mortality!(device(arch), (8,8,4,1), (lonres, latres, depthres, n_res))
    kernel_calc(
        agent_biomass_eaten_grid,
        resource_total_biomass,
        all_prey_cell_starts,
        all_prey_cell_ends,
        all_prey_sorted_id,
        all_prey_alive,
        all_prey_length,
        all_prey_biomass_school,
        all_prey_biomass_ind,
        output.consumption,
        size_bin_thresholds,
        Int32(n_sp),
        Int32(n_res),
        n_thresholds,
        resource_traits_matrix,
        agent_energy_densities_gpu,
        Float32(model.dt),
        grid_params.cell_size_deg,
        grid_params.depth_res_m,
        Int32(lonres),
        Int32(latres)
    )

    # `calculate_potential_mortality!` writes agent_biomass_eaten_grid, which the
    # per-species `apply_and_log_mortality!` launches below read. Force completion of
    # the producer kernel before launching the consumers so there is no read/write
    # hazard on that intermediate buffer.
    KernelAbstractions.synchronize(device(arch))

    for sp_idx in 1:n_sp
        agents = model.individuals.animals[sp_idx].data
        if length(agents.x) > 0
            kernel_apply = apply_and_log_mortality!(device(arch), 256, (length(agents.x),))
            kernel_apply(
                agents.alive, agents.pool_x, agents.pool_y, agents.pool_z, agents.biomass_ind, agents.biomass_school, agents.energy, agents.abundance, agents.length,
                agent_biomass_eaten_grid, 
                output.consumption,
                output.Pmort,
                size_bin_thresholds,
                Int32(sp_idx),
                Int32(n_sp)
            )
        end
    end

    KernelAbstractions.synchronize(device(arch))
end