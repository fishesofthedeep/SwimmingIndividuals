function dvm_action(model, sp, ind,outputs)
    animal = model.individuals.animals[sp]
    data = animal.data
    params = animal.p
    ΔT = params.t_resolution[2][sp]
    swim_speed = 2.64 #Meters per minute following Bianchi and Mislan 2016. Want to make this size based.
    z_dist_file_day = model.files[model.files.File .== "focal_z_dist_day", :Destination][1]
    z_dist_file_night = model.files[model.files.File .== "focal_z_dist_night", :Destination][1]
    grid_file = model.files[model.files.File .== "grid", :Destination][1]
    z_day_dist = CSV.read(z_dist_file_day, DataFrame)
    z_night_dist = CSV.read(z_dist_file_night, DataFrame)
    grid = CSV.read(grid_file, DataFrame)
    maxdepth = grid[grid.Name .== "depthmax", :Value][1]
    target_z_day = gaussmix(length(ind), z_day_dist[sp, "mu1"], z_day_dist[sp, "mu2"], z_day_dist[sp, "mu3"], z_day_dist[sp, "sigma1"], z_day_dist[sp, "sigma2"], z_day_dist[sp, "sigma3"], z_day_dist[sp, "lambda1"], z_day_dist[sp, "lambda2"])
    target_z_night = gaussmix(length(ind), z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"], z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"], z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"], z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])

    target_z_night .= clamp.(target_z_night, 1, maxdepth)

    for i in 1:length(ind)


        if (6*60 <= model.t < 18*60) && (any(data.mig_status[i] == 0.0)) # Start descent during daytime
            z_dist_file = model.files[model.files.File .=="focal_z_dist_day",:Destination][1]
            grid_file = model.files[model.files.File .=="grid",:Destination][1]
            z_day_dist = CSV.read(z_dist_file,DataFrame)
            grid = CSV.read(grid_file,DataFrame)
            maxdepth = grid[grid.Name .== "depthmax", :Value][1]

            data.mig_status[i] = 2
            data.target_z[i] = gaussmix(1,z_day_dist[sp,"mu1"],z_day_dist[sp,"mu2"],z_day_dist[sp,"mu3"],z_day_dist[sp,"sigma1"],z_day_dist[sp,"sigma2"],z_day_dist[sp,"sigma3"],z_day_dist[sp,"lambda1"],z_day_dist[sp,"lambda2"])[1]
            data.target_z[i] = clamp(data.target_z[i], 1, maxdepth)

            data.mig_rate[i] = swim_speed
            t_adjust = min(ΔT, abs((data.target_z[i] - data.z[i]) / data.mig_rate[i]))
            data.z[i] = min(data.target_z[i], data.z[i] + data.mig_rate[i] * ΔT)
            data.behavior[i] = 2
            if data.z[i] >= data.target_z[i]
                data.mig_status[i] = -1
            end
            
            data.feeding[i] = 0
        elseif (model.t >= 18*60) && (data.mig_status[i] == -1) # Start acent during nighttime
            z_dist_file = model.files[model.files.File .=="focal_z_dist_night",:Destination][1]
            grid_file = model.files[model.files.File .=="grid",:Destination][1]
            z_night_dist = CSV.read(z_dist_file,DataFrame)
            grid = CSV.read(grid_file,DataFrame)
            maxdepth = grid[grid.Name .== "depthmax", :Value][1]

            data.mig_status[i] = 1
            data.target_z[i] = gaussmix(1,z_night_dist[sp,"mu1"],z_night_dist[sp,"mu2"],z_night_dist[sp,"mu3"],z_night_dist[sp,"sigma1"],z_night_dist[sp,"sigma2"],z_night_dist[sp,"sigma3"],z_night_dist[sp,"lambda1"],z_night_dist[sp,"lambda2"])[1]
            data.target_z[i] = clamp(data.target_z[i], 1, maxdepth)

            data.behavior[i] = 2

            data.mig_rate[i] = swim_speed
            t_adjust = min(ΔT, abs((data.target_z[i] - data.z[i]) / data.mig_rate[i]))
            data.z[i] = max(data.target_z[i], data.z[i] - data.mig_rate[i] * ΔT)
            if data.z[i] == data.target_z[i]
                data.mig_status[i] = 0
                data.feeding[i] = 1
            end

        elseif data.mig_status[i] == 1 # Continue ascending
            target_z = data.target_z[i]
            t_adjust = min(ΔT, abs((target_z - data.z[i]) / data.mig_rate[i]))
            data.z[i] = max(target_z, data.z[i] - data.mig_rate[i] * ΔT)
            if (data.z[i] == target_z) | (model.t == 21*60)
                data.mig_status[i] = 0
                data.feeding[i] = 1
            end

        elseif data.mig_status[i] == 2 # Continue descending
            target_z = data.target_z[i]
            t_adjust = min(ΔT, abs((target_z - data.z[i]) / data.mig_rate[i]))
            data.z[i] = min(target_z, data.z[i] + data.mig_rate[i] * ΔT)
            if (data.z[i] == target_z) | (model.t == 9*60)
                data.mig_status[i] = -1
            end

        elseif data.mig_status[i] == 0
            data.behavior[i] = 1
        elseif data.mig_status[i] == -1
            data.behavior[i] = 0
        end
    end
    return nothing
end

function surface_dive(model, sp, ind)
    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp].data
        animal_p = model.individuals.animals[sp].p
        # Define behavioral parameters
        surface_interval = animal_p.Surface_Interval[2][sp]
        foraging_interval = animal_p.Dive_Interval[2][sp]
        dive_velocity = (animal.length[i]/1000) * 60 # meters per minute

        # Time resolution
        ΔT = animal_p.t_resolution[2][sp]
        # Progress the animal through the stages
        if animal.mig_status[i] == 0
            # Surface interval

            animal.interval[i] += ΔT
            max_fullness = 0.2 * animal.biomass[i]
            dive_trigger = animal.gut_fullness[i] / max_fullness
            dist = logistic(dive_trigger, 5, 0.5)
            #Add probability here to start diving

            if animal.interval[i] >= surface_interval && rand() > dist
                # Reset interval timer and move to the next stage (diving)
                animal.interval[i] = 0
                animal.mig_status[i] = 1
                # Randomly select dive depth
                animal.target_z[i] = sample_normal(animal_p.Dive_depth_min[2][sp], animal_p.Dive_depth_max[2][sp], std=20)[rand(1:end)]
            end

        elseif animal.mig_status[i] == 1
            # Continue diving

            #Change depth
            if animal.z[i] >= animal.target_z[i]
                # Reset active time and move to the next stage (foraging interval)
                animal.mig_status[i] = -1
                animal.feeding[i] = 1
            end

        elseif animal.mig_status[i] == -1
            # Foraging interval

            if animal.interval[i] >= foraging_interval
                # Reset interval timer and move to the next stage (ascending)
                animal.interval[i] = 0
                animal.mig_status[i] = 2
                animal.feeding[i] = 0
            else
                animal.interval[i] += ΔT
            end

        elseif animal.mig_status[i] == 2
            # Ascending

            if animal.z[i] <= 1
                # Reset active time and move to the next cycle (back to surface interval)
                animal.interval[i] = 0
                animal.mig_status[i] = 0

                # Increment dive count for the day
                animal.dives_remaining[i] -= 1
            end
        end

        # Check if the dive count for the day exceeds the maximum limit
        if animal.dives_remaining[i] <= 0
            # Prevent further diving
            animal.mig_status[i] = 0
        end

        # Update the position of the animal based on the dive velocity
        if animal.mig_status[i] == 1
            animal.z[i] = min(animal.target_z[i], animal.z[i] + dive_velocity * ΔT)
        elseif animal.mig_status[i] == 2
            animal.z[i] = max(1, animal.z[i] - dive_velocity * ΔT)
        end
    end

    return nothing
end

function pelagic_dive(model, sp, ind)

    for i in ind[1]:ind[1]
        animal = model.individuals.animals[sp].data
        animal_p = model.individuals.animals[sp].p
        # Define behavioral parameters
        surface_interval = animal_p.Surface_Interval[2][sp]
        foraging_interval = animal_p.Dive_Interval[2][sp]
        dive_velocity = (animal.length[i]/1000) * 60 # meters per minute

        # Time resolution
        ΔT = animal_p.t_resolution[2][sp]

        # Progress the animal through the stages
        if animal.mig_status[i] == 0
            # Surface interval
            animal.interval[i] += ΔT
            max_fullness = 0.2 * animal.biomass[i]
            dive_trigger = animal.gut_fullness[i] / max_fullness
            dist = logistic(dive_trigger, 5, 0.5)
            #Add probability here to start diving


            if animal.interval[i] >= surface_interval && rand() > dist
                # Reset interval timer and move to the next stage (diving)
                animal.interval[i] = 0
                animal.mig_status[i] = 1
                # Randomly select dive depth
                animal.target_z[i] = -50
                while animal.z[i] < animal.target_z[i]
                    if (6*60 <= model.t < 18*60)
                        animal.target_z[i] = gaussmix(1, z_day_dist[sp, "mu1"], z_day_dist[sp, "mu2"],z_day_dist[sp, "mu3"], z_day_dist[sp, "sigma1"],z_day_dist[sp, "sigma2"], z_day_dist[sp, "sigma3"],z_day_dist[sp, "lambda1"], z_day_dist[sp, "lambda2"])[1]
                    else
                        animal.target_z[i] = gaussmix(1, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])[1]
                    end
                end
            end

        elseif animal.mig_status[i] == 1
            # Continue diving

            #Change depth
            if animal.z[i] >= animal.target_z[i]
                # Reset active time and move to the next stage (foraging interval)
                animal.mig_status[i] = -1
                animal.feeding[i] = 1
            end

        elseif animal.mig_status[i] == -1
            # Foraging interval
            animal.interval[i] += ΔT

            if animal.interval[i] >= foraging_interval
                # Reset interval timer and move to the next stage (ascending)
                animal.interval[i] = 0
                animal.mig_status[i] = 2
                animal.feeding[i] = 0

                animal.target_z[i] = 5e6
                while animal.z[i] > animal.target_z[i]
                    if (6*60 <= model.t < 18*60)
                        animal.target_z[i] = gaussmix(1, z_day_dist[sp, "mu1"], z_day_dist[sp, "mu2"],z_day_dist[sp, "mu3"], z_day_dist[sp, "sigma1"],z_day_dist[sp, "sigma2"], z_day_dist[sp, "sigma3"],z_day_dist[sp, "lambda1"], z_day_dist[sp, "lambda2"])[1]
                    else
                        animal.target_z[i] = gaussmix(1, z_night_dist[sp, "mu1"], z_night_dist[sp, "mu2"],z_night_dist[sp, "mu3"], z_night_dist[sp, "sigma1"],z_night_dist[sp, "sigma2"], z_night_dist[sp, "sigma3"],z_night_dist[sp, "lambda1"], z_night_dist[sp, "lambda2"])[1]
                    end
                end
            end

        elseif animal.mig_status[i] == 2
            # Ascending

            if animal.z[i] <= animal.target_z[i]
                # Reset active time and move to the next cycle (back to surface interval)
                animal.interval[i] = 0
                animal.mig_status[i] = 0

                # Increment dive count for the day
                animal.dives_remaining[i] -= 1
            end
        end

        # Check if the dive count for the day exceeds the maximum limit
        if animal.dives_remaining[i] <= 0
            # Prevent further diving
            animal.mig_status[i] = 0
        end

        # Update the position of the animal based on the dive velocity
        if animal.mig_status[i] == 1
            animal.z[i] = min(animal.target_z[i], animal.z[i] + dive_velocity * ΔT)
        elseif animal.mig_status[i] == 2
            animal.z[i] = max(animal.target_z[i], animal.z[i] - dive_velocity * ΔT)
        end
    end

    return nothing
end

function cost_function_prey(position, predator_matrix)
    sum_distance = 0.0
    for predator in eachrow(predator_matrix)
        sum_distance += norm(position .- Vector(predator))
    end
    return -sum_distance  # We maximize distance, hence the negative
end

#Optimization function to find the ideal location for prey to go
function predator_avoidance(predator_locations, initial_prey_location, max_distance)
    initial_guess = initial_prey_location  # Initial guess for prey location
    predator_matrix = hcat(predator_locations.x, predator_locations.y, predator_locations.z)
    dist = [max_distance, max_distance, max_distance]
    lower_bound = initial_guess .- dist   # Lower bound for prey location
    upper_bound = initial_guess .+ dist   # Upper bound for prey location

    result = optimize(p -> cost_function_prey(p, predator_matrix), lower_bound, upper_bound, initial_guess, Fminbox())

    return Optim.minimizer(result)
end

function move_to_prey(model, sp, ind,eat_ind, time, prey_list)
    ddt = fill(model.individuals.animals[sp].p.t_resolution[2][sp] * 60.0,length(ind))
    ddt[eat_ind] .= time #Update times from feeding

    distances = model.individuals.animals[sp].p.Swim_velo[2][sp] .* (model.individuals.animals[sp].data.length[ind] / 1000) .* ddt  # meters the animal can swim

    Threads.@threads for ind_index in 1:length(ind)
        animal = ind[ind_index]
        prey_info = prey_list.preys[ind_index]
        
        if !isempty(prey_info)
            # Extract distances of all preys
            prey_distances = [prey.Distance for prey in prey_info]
            
            # Find the nearest prey
            min_distance, index = findmin(prey_distances)

            if !iszero(index)
                if min_distance <= distances[ind_index]
                    # Directly move to prey location
                    nearest_prey = prey_info[index]
                    model.individuals.animals[sp].data.x[animal] = nearest_prey.x
                    model.individuals.animals[sp].data.y[animal] = nearest_prey.y
                    model.individuals.animals[sp].data.z[animal] = nearest_prey.z
                else
                    # Move in a straight line towards the prey
                    nearest_prey = prey_info[index]
                    dx = nearest_prey.x - model.individuals.animals[sp].data.x[animal]
                    dy = nearest_prey.y - model.individuals.animals[sp].data.y[animal]
                    dz = nearest_prey.z - model.individuals.animals[sp].data.z[animal]

                    if !iszero(dx) && !iszero(dy) && !iszero(dz)
                        norm_factor = sqrt(dx^2 + dy^2 + dz^2)
                        direction = (dx, dy, dz) ./ norm_factor

                        model.individuals.animals[sp].data.x[animal] += direction[1] * distances[ind_index]
                        model.individuals.animals[sp].data.y[animal] += direction[2] * distances[ind_index]
                        model.individuals.animals[sp].data.z[animal] += direction[3] * distances[ind_index]
                    else
                        model.individuals.animals[sp].data.x[animal] = nearest_prey.x
                        model.individuals.animals[sp].data.y[animal] = nearest_prey.y
                        model.individuals.animals[sp].data.z[animal] = nearest_prey.z
                    end
                end
            end
        end
    end
end


