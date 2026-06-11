# ===================================================================
# GPU-Compliant Visual Range Calculation
# ===================================================================

# Kernel to calculate the visual range for prey for each agent in parallel.
@kernel function visual_range_kernel!(vis_out, length, depth, min_prey, max_prey, surface_irradiance)
    ind = @index(Global)

    # Constants
    pred_contrast = 0.3f0
    salt = 30.0f0
    attenuation_coefficient = 0.32f0 - 0.016f0 * salt
    
    # Agent-specific calculations
    ind_length_m = length[ind] / 1000.0f0
    rmax = ind_length_m * 30.0f0
    
    # Simplified image size calculation for kernel
    pred_image = 0.75f0 * (ind_length_m / 0.01f0) * (ind_length_m / 4.0f0)
    eye_sensitivity = (rmax^2) / (pred_image * pred_contrast)
    
    # Light at depth
    I_z = surface_irradiance * exp(-attenuation_coefficient * depth[ind])
    
    # --- FIX: Calculate prey size factor based on actual prey size ---
    # The visual range is determined by the size of the target. We use the
    # smallest prey the predator will consider as the limiting factor.
    min_prey_length_m = ind_length_m * min_prey
    
    # Ensure prey size factor is not zero to avoid division by zero
    if min_prey_length_m > 0.0f0
        # The original formula for r_sq was physically incorrect. This is a more standard and stable formulation.
        # Reaction distance (r) is proportional to predator length and prey length.
        r_sq = (I_z * min_prey_length_m^2) / (eye_sensitivity * pred_contrast^2)
        r = r_sq > 0.0f0 ? sqrt(r_sq) : 0.0f0
        
        @inbounds vis_out[ind] = clamp(r, 0.0f0, rmax)
    else
        @inbounds vis_out[ind] = 0.0f0
    end
end

# Launcher for the visual range kernel.
function update_visual_range!(model::MarineModel, sp::Int)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p = model.individuals.animals[sp].p
    
    surface_irradiance = ipar_curve(model.t)
    
    kernel! = visual_range_kernel!(device(arch), 256, (length(data.x),))
    kernel!(
        data.vis_prey, data.length, data.z,
        p.Min_Prey[2][sp], p.Max_Prey[2][sp],
        surface_irradiance
    )
    KernelAbstractions.synchronize(device(arch))
end

# ===================================================================
# Core Behavioral Functions
# ===================================================================

# Main behavioral dispatcher.
function behavior(model::MarineModel, sp::Int64, ind::Vector{Int32}, outputs::MarineOutputs)
    behave_type = model.individuals.animals[sp].p.Type[2][sp]
    sp_dat = model.individuals.animals[sp].data
    
    # Update visual range for all agents of this species first
    update_visual_range!(model, sp)

    if behave_type == "dvm_strong"
        dvm_action!(model, sp, false) # false indicates it is NOT a weak migrator
        
        # Find which agents are NOT migrating to pass them to the decision function
        mig_status_cpu = Array(sp_dat.mig_status)
        not_migrating_indices = findall(x -> x <= 0.0, mig_status_cpu)
        decision_inds = Int32.(intersect(ind, not_migrating_indices))
        
        if !isempty(decision_inds)
            decision(model, sp, decision_inds, outputs)
        end
        
    elseif behave_type == "dvm_weak"
        dvm_action!(model, sp, true) # true indicates it IS a weak migrator

        mig_status_cpu = Array(sp_dat.mig_status)
        not_migrating_indices = findall(x -> x <= 0.0, mig_status_cpu)
        decision_inds = Int32.(intersect(ind, not_migrating_indices))

        if !isempty(decision_inds)
            decision(model, sp, decision_inds, outputs)
        end

    elseif behave_type == "pelagic_diver"
        dive_action!(model, sp)

        mig_status_cpu = Array(sp_dat.mig_status)
        # Statuses are: 0.0 (surface rest), -1.0 (deep rest)
        not_migrating_indices = findall(x -> x == 0.0 || x == -1.0, mig_status_cpu)
        decision_inds = Int32.(intersect(ind, not_migrating_indices))

        if !isempty(decision_inds)
            decision(model, sp, decision_inds, outputs)
        end
        
    elseif behave_type == "non_mig"
        decision(model, sp, ind, outputs)
    end
    
    return nothing
end

function find_foraging_indices!(
    time::AbstractVector,
    gut_fullness::AbstractVector,
    inds::AbstractVector{T}
) where {T <: Integer}
    still_forage_mask = (time .> 0f0) .& (gut_fullness .< 1)
    filtered_inds = inds[still_forage_mask]
    return filtered_inds
end

# Core decision-making function for foraging and movement.
function decision(model::MarineModel, sp::Int, ind::Vector{Int32}, outputs::MarineOutputs)
    sp_dat = model.individuals.animals[sp].data
    arch = model.arch

    grid = model.depths.grid
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])

    # ---- VALIDATION CHECK 1: Before Foraging ----
    # Check for corrupted coordinates before starting the predation logic.
    cpu_pool_x_pre = Array(sp_dat.pool_x)
    cpu_pool_y_pre = Array(sp_dat.pool_y)
    valid_indices = Int32[]
    for i in ind
        px, py = cpu_pool_x_pre[i], cpu_pool_y_pre[i]
        if 1 <= px <= lonres && 1 <= py <= latres
            push!(valid_indices, i)
        else
            @warn "PRE-FORAGE CORRUPTION DETECTED: Agent $(i) of species $sp has invalid coordinates ($px, $py). Excluding from foraging."
        end
    end

    # ---- Move relevant data to GPU once ----
    ind_gpu = array_type(arch)(Int32.(ind))
    n = length(ind_gpu)

    gut_fullness_gpu = array_type(arch)(Float32.(sp_dat.gut_fullness[ind]))

    rand_vals = array_type(arch)(rand(Float32, n))
    eat_mask_gpu = gut_fullness_gpu .<= rand_vals
    eating_gpu = ind_gpu[eat_mask_gpu]  # Initial foragers

    # Allocate a full-length time array on GPU (indexed by full predator index)
    time_gpu = array_type(arch)(fill(Float32(model.dt * 60.0), length(sp_dat.alive)))

    # -------- Foraging Loop --------
    num_foraging_attempts = model.foraging_attempts
    vprint("eat | ")

    for i in 1:num_foraging_attempts
        if isempty(eating_gpu); break; end

        # Convert GPU indices to CPU for CPU-only function
        eating = Array(eating_gpu)

        # CPU: Identify prey
        calculate_distances_prey!(model, sp, eating)

        # CPU: Resolve conflicts and allocate rations
        resolve_consumption!(model, sp, eating)

        # GPU: Apply consumption and update time
        apply_consumption!(model, sp, time_gpu, outputs)

        ## Debugging if necessary. Mainly for me.

        # Filter: Keep only individuals who ate something
        ration_vals = array_type(arch)(Float32.(sp_dat.successful_ration[eating]))
        nonzero_mask = ration_vals .> 0f0
        eating_gpu = eating_gpu[nonzero_mask]

        # Stop if none left
        isempty(eating_gpu) && break

        # Recalculate who should continue foraging
        gut_fullness_now = array_type(arch)(Float32.(sp_dat.gut_fullness[Array(eating_gpu)]))
        time_now = array_type(arch)(Float32.(time_gpu[Array(eating_gpu)]))

        eating_gpu = find_foraging_indices!(
            time_now, gut_fullness_now, eating_gpu
        )
    end

    cpu_pool_x_post = Array(sp_dat.pool_x)
    cpu_pool_y_post = Array(sp_dat.pool_y)
    for i in ind
        px, py = cpu_pool_x_post[i], cpu_pool_y_post[i]
        if !(1 <= px <= lonres && 1 <= py <= latres)
            @warn "POST-FORAGE CORRUPTION DETECTED: Agent $(i) of species $sp has invalid coordinates ($px, $py) after consumption."
        end
    end

    print("move | ")
    

    # Movement based on remaining time
    movement_toward_habitat!(model, sp, time_gpu)

    return nothing
end
# ===================================================================
# CPU-based Initialization Functions (used only during setup)
# ===================================================================

function visual_range_preds_init(length,depth,min_pred,max_pred,ind)
    pred_contrast = fill(0.3,ind)
    salt = fill(30, ind)
    attenuation_coefficient = 0.32 .- 0.016 .* salt
    ind_length = length ./ 1000
    pred_length = ind_length ./ 0.01
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(0)
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    pred_size_factor = 1+((min_pred+max_pred)/2)
    r = max.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* pred_size_factor)))
    return r
end

function visual_range_preys_init(length,depth,min_prey,max_prey,ind)
    pred_contrast = fill(0.3,ind)
    salt = fill(30, ind)
    attenuation_coefficient = 0.32 .- 0.016 .* salt
    ind_length = length ./ 1000
    pred_length = ind_length ./ 0.01
    pred_width = pred_length ./ 4
    pred_image = 0.75 .* pred_length .* pred_width
    rmax = ind_length .* 30
    eye_sensitivity = (rmax.^2) ./ (pred_image .* pred_contrast)
    surface_irradiance = ipar_curve(0)
    I_z = surface_irradiance .* exp.(-attenuation_coefficient .* depth)
    prey_size_factor = (min_prey+max_prey)/2
    r = min.(1,ind_length .* sqrt.(I_z ./ (pred_contrast .* eye_sensitivity .* prey_size_factor)))
    return r
end