function resource_mortality!(model::MarineModel)
    arch = model.arch
    # Z rates from the trait table are treated as ANNUAL instantaneous rates
    annual_z_rates = model.resource_trait.Z
    
    # Convert annual instantaneous Z to a per-timestep proportional loss
    minutes_per_year = 365.0f0 * 1440.0f0
    per_minute_z = annual_z_rates ./ minutes_per_year
    
    # Proportion surviving one timestep = exp(-Z*dt)
    proportion_surviving = exp.(-per_minute_z .* model.dt)
    per_timestep_survival_rates = array_type(arch)(Float32.(proportion_surviving))

    # The kernel will now use the survival rates directly
    kernel! = resource_mortality_kernel!(device(arch), (8, 8, 4, 1), size(model.resources.biomass))
    kernel!(model.resources.biomass, per_timestep_survival_rates)
    KernelAbstractions.synchronize(device(arch))
end

@kernel function resource_mortality_kernel!(biomass_grid, per_timestep_survival_rates)
    lon, lat, depth, sp = @index(Global, NTuple)
    biomass = biomass_grid[lon, lat, depth, sp]

    if biomass > 0
        # Apply the pre-calculated per-timestep survival proportion
        @inbounds biomass_grid[lon, lat, depth, sp] = biomass * per_timestep_survival_rates[sp]
    end
end

"""
    remove_dead_agents!(model::MarineModel)

Resizes the agent StructArray for each species to remove all individuals
that are not alive (`alive == 0.0`). This should be called periodically
to manage memory and improve performance.
"""
function remove_dead_agents!(model::MarineModel)
    println("--- Starting Dead Agent Removal ---")
    for sp in 1:model.n_species
        # Bring data to the CPU to perform the resize operation
        animal_data_cpu = Array(model.individuals.animals[sp].data)
        total_before = length(animal_data_cpu)

        # Find the indices of all agents that are still alive
        alive_indices = findall(a -> a.alive == 1.0f0, animal_data_cpu)
        
        # Create a new, smaller StructArray containing only the living agents
        new_animal_data = animal_data_cpu[alive_indices]
        
        # Copy the resized data back to the GPU
        model.individuals.animals[sp].data = arch_array(model.arch, new_animal_data)

        # Print a summary of the operation
        total_after = length(new_animal_data)
        println("Species $sp: Removed $(total_before - total_after) dead agents. Total agents reduced from $total_before to $total_after.")
    end
    println("--- Dead Agent Removal Complete ---")
end

"""
    natural_mortality!(model::MarineModel, sp::Int, outputs::MarineOutputs)

Applies a baseline instantaneous natural mortality rate (M) to individuals 
within an agent. Updates school biomass, abundance, and energy, while 
logging the results to the global Nmort matrix.
"""
function natural_mortality!(model::MarineModel, sp::Int)
    arch = model.arch
    data = model.individuals.animals[sp].data
    p_cpu = model.individuals.animals[sp].p
    
    # 1. Gather Parameters
    m_annual = Float32(p_cpu.M.second[sp])
    minutes_per_year = 365.0f0 * 1440.0f0
    m_step = m_annual * (model.dt / minutes_per_year)
    
    # 2. Launch Kernel
    kernel! = natural_mortality_kernel!(device(arch), 256, (length(data.x),))
    kernel!(
        data.alive, 
        data.abundance, 
        data.biomass_school, 
        data.energy,
        m_step
    )
    
    KernelAbstractions.synchronize(device(arch))
end

@kernel function natural_mortality_kernel!(
    alive, abundance, biomass_school, energy, m_step
)
    ind = @index(Global)
    
    @inbounds if alive[ind] == 1.0f0
        # Calculate survival fraction: S = exp(-M*dt)
        survival_frac = exp(-m_step)
        
        # 1. Calculate new abundance with stochastic rounding
        old_n = Float32(abundance[ind])
        new_n_float = old_n * survival_frac
        new_n_int = Int64(floor(new_n_float + rand(Float32)))
        
        # 2. Update State Variables
        # There is no race condition for per-agent data.
        if biomass_school[ind] > 0.0f0
            energy[ind] *= survival_frac
        end
        
        abundance[ind] = new_n_int
        biomass_school[ind] *= survival_frac
        
        # 5. Handle Extinction
        if abundance[ind] <= 0
            alive[ind] = 0.0f0
            biomass_school[ind] = 0.0f0
            energy[ind] = 0.0f0
        end
    end
end