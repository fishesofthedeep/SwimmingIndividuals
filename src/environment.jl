# ===================================================================
# Data Structures and Environment Loading
# ===================================================================

function generate_environment!(arch::Architecture, nc_file::String, plt_diags, files::DataFrame)
    ds = NCDataset(nc_file)
    envi_data = Dict{String, AbstractArray}()

    # Get the base results directory from the files DataFrame
    res_dir = files[files.File .== "res_dir", :Destination][1]
    # Construct the specific path for environment diagnostic plots
    output_dir = joinpath(res_dir, "diags", "Environment")

    # Only create/clear the directory if plotting is enabled
    if plt_diags == 1
        # Use isdir() to prevent errors if the directory doesn't exist yet
        isdir(output_dir) && rm(output_dir, recursive=true)
        mkpath(output_dir)
        @info "Loading environment parameters and saving plots to: $(abspath(output_dir))"
    else
        @info "Loading environment parameters..."
    end

    for var_name in keys(ds)
        if var_name in ["lon", "lat", "depth", "time"]
            continue
        end

        data_cpu_array = Array(ds[var_name])  # load into CPU
        envi_data[var_name] = array_type(arch)(data_cpu_array)  # send to CPU or GPU

        # Plotting logic
        if plt_diags == 1
            @info "Loading and plotting environmental variable: $var_name"

            local surface_slice
            if ndims(data_cpu_array) >= 3
                if ndims(data_cpu_array) == 4
                    surface_slice = data_cpu_array[:, :, 1, 1]
                else # 3D
                    surface_slice = data_cpu_array[:, :, 1]
                end
            elseif ndims(data_cpu_array) == 2
                surface_slice = data_cpu_array
            else
                @warn "Skipping plot for variable '$var_name' with unsupported dimensions."
                continue
            end

            p = heatmap(
                surface_slice',
                title = "Surface Layer: $var_name (Month 1)",
                xlabel = "Longitude Index",
                ylabel = "Latitude Index",
                c = :viridis
            )
            savefig(p, joinpath(output_dir, "$(var_name)_surface.png"))
        end
    end

    close(ds)
    return MarineEnvironment(envi_data, 1)
end

# ===================================================================
# Agent Temperature Sensing System
# ===================================================================

@kernel function calculate_temp_indices_kernel!(
    temp_idx_out, alive, pool_x, pool_y, pool_z,
    envi_ts, lonres, latres, depthres
)
    ind = @index(Global)
    @inbounds if alive[ind] == 1.0
        x, y, z = pool_x[ind], pool_y[ind], pool_z[ind]
        if 1 <= x <= lonres && 1 <= y <= latres && 1 <= z <= depthres
            linear_idx = x + (y-1)*lonres + (z-1)*lonres*latres + (envi_ts-1)*lonres*latres*depthres
            temp_idx_out[ind] = linear_idx
        else
            temp_idx_out[ind] = 0
        end
    else
        temp_idx_out[ind] = 0
    end
end

@kernel function gather_temps_kernel!(temps_out, temp_grid_4d, temp_indices)
    ind = @index(Global)
    idx = temp_indices[ind]
    if idx > 0
        @inbounds temps_out[ind] = temp_grid_4d[idx]
    end
end

function individual_temp!(model::MarineModel, sp::Int)
    arch = model.arch
    envi = model.environment
    files = model.files
    grid = model.depths.grid

    agent_data = model.individuals.animals[sp].data
    
    temp_grid_4d = envi.data["temp"]
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])

    kernel1 = calculate_temp_indices_kernel!(device(arch), 256, (length(agent_data.x),))
    kernel1(
        agent_data.temp_idx, agent_data.alive,
        agent_data.pool_x, agent_data.pool_y, agent_data.pool_z,
        envi.ts, lonres, latres, depthres
    )

    temps_out = array_type(arch)(zeros(eltype(temp_grid_4d), length(agent_data.x)))
    kernel2 = gather_temps_kernel!(device(arch), 256, (length(agent_data.x),))
    kernel2(temps_out, temp_grid_4d, agent_data.temp_idx)

    KernelAbstractions.synchronize(device(arch))

    return temps_out
end

# ===================================================================
# Habitat Capacity and Agent Placement
# ===================================================================

function initial_habitat_capacity(envi::MarineEnvironment, n_spec::Int32, n_resource::Int32, files::DataFrame, arch::Architecture, plt_diags)
    prefs_df = CSV.read(files[files.File .== "envi_pref",:Destination][1], DataFrame)
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait",:Destination][1], DataFrame))))
    resource = Dict(pairs(eachcol(CSV.read(files[files.File .== "resource_trait",:Destination][1], DataFrame))))

    spec_names = vcat(trait[:SpeciesLong][1:n_spec], resource[:SpeciesLong][1:n_resource])
    
    # --- 1. GATHER DATA TO CPU ---
    envi_data_cpu = Dict{String, Array}()
    for (name, grid) in envi.data
        envi_data_cpu[name] = Array(grid)
    end
    
    ref_var = envi_data_cpu["temp-surf"]
    lonres, latres, nmonths = size(ref_var)
    
    # --- 2. COMPUTE HABITAT ON CPU ---
    capacities_cpu = ones(Float32, lonres, latres, nmonths, n_spec + n_resource)

    Threads.@threads for i in 1:length(spec_names)
        sp_name = spec_names[i]
        sp_prefs = filter(row -> row.species == sp_name, prefs_df)
        
        if !isempty(sp_prefs)
            for month in 1:nmonths, lat in 1:latres, lon in 1:lonres
                
                suitability_total = 1.0
                
                for pref_row in eachrow(sp_prefs)
                    var_name = pref_row.variable
                    if !haskey(envi_data_cpu, var_name); continue; end
                    
                    env_var_grid = envi_data_cpu[var_name]
                    val = (ndims(env_var_grid) == 3) ? env_var_grid[lon, lat, month] : env_var_grid[lon, lat]

                    if ismissing(val)
                        suitability_total = 0.0
                        break
                    end
                    
                    pref_min = pref_row.pref_min
                    opt_min = pref_row.opt_min
                    opt_max = pref_row.opt_max
                    pref_max = pref_row.pref_max

                    suitability_i = 0.0
                    if any(ismissing, [pref_min, opt_min, opt_max, pref_max])
                        suitability_i = 1.0 
                    else
                        if val >= opt_min && val <= opt_max
                            suitability_i = 1.0
                        elseif val > pref_min && val < opt_min
                            suitability_i = (val - pref_min) / (opt_min - pref_min)
                        elseif val > opt_max && val < pref_max
                            suitability_i = (pref_max - val) / (pref_max - opt_max)
                        end
                    end
                    
                    suitability_total *= suitability_i
                end
                
                capacities_cpu[lon, lat, month, i] = suitability_total
            end
        end
    end
    
    # --- EXPORT MAPS OF HABITAT CAPACITY ---
    if plt_diags == 1
        # Get the base results directory
        res_dir = files[files.File .== "res_dir", :Destination][1]
        # Construct the specific output path for capacity plots
        output_dir = joinpath(res_dir, "diags", "Capacities")
        
        # Use isdir() to prevent errors if the directory doesn't exist yet
        isdir(output_dir) && rm(output_dir, recursive=true)
        mkpath(output_dir)

        for i in 1:length(spec_names)
            sp_name = spec_names[i]
            for month in 1:nmonths
                capacity_slice = capacities_cpu[:, :, month, i]
                p = heatmap(
                    capacity_slice', 
                    title = "",
                    xlabel = "Longitude Index",
                    ylabel = "Latitude Index",
                    c = :viridis,
                    clims = (0, 1)
                )
                savefig(p, joinpath(output_dir, "$(sp_name)_month_$(month)_capacity.png"))
            end
        end
        @info "Habitat capacity maps exported to $(abspath(output_dir))"
    end

    # --- 3. Copy the final CPU result back to the target device (GPU) ---
    return array_type(arch)(capacities_cpu)
end

function initial_ind_placement(df_cpu, sp, grid, n_selections, month, land_mask)
    latmax = grid[grid.Name .== "yulcorner", :Value][1]
    lonmin = grid[grid.Name .== "xllcorner", :Value][1]
    cell_size = grid[grid.Name .== "cellsize", :Value][1]

    lonres, latres = size(land_mask)

    capacity_slice = @view df_cpu[:, :, month, sp]
    
    valid_cells = DataFrame(x=Int[], y=Int[], value=Float32[])
    for lon in 1:lonres, lat in 1:latres
        # The definitive check: Must have capacity > 0
        if capacity_slice[lon, lat] > 0
            push!(valid_cells, (x=lon, y=lat, value=capacity_slice[lon, lat]))
        end
    end

    if nrow(valid_cells) > 0
        sort!(valid_cells, :value, rev=true)
        cumvals = cumsum(valid_cells.value)
        total = cumvals[end]

        x_values, y_values = Int[], Int[]
        for _ in 1:n_selections
            r = (rand()^2) * total
            selected_idx = findfirst(cumvals .>= r)
            if selected_idx === nothing; selected_idx = 1; end
            push!(x_values, valid_cells.x[selected_idx])
            push!(y_values, valid_cells.y[selected_idx])
        end
        flipped_y = (latres .- y_values) .+ 1

        actual_x = lonmin .+ (x_values .- 1) .* cell_size .+ rand(n_selections) .* cell_size
        actual_y = latmax .- (flipped_y .- 1) .* cell_size .- rand(n_selections) .* cell_size

        return (lons=actual_x, lats=actual_y, grid_x=x_values, grid_y=y_values)
    else
        @warn "No valid water-based habitat for species $sp in month $month. Placing randomly."
        x_values = rand(1:lonres, n_selections)
        y_values = rand(1:latres, n_selections)
        flipped_y = (latres .- y_values) .+ 1

        actual_x = lonmin .+ (x_values .- 1) .* cell_size .+ rand(n_selections) .* cell_size
        actual_y = latmax .- (flipped_y .- 1) .* cell_size .- rand(n_selections) .* cell_size
        return (lons=actual_x, lats=actual_y, grid_x=x_values, grid_y=y_values)
    end
end