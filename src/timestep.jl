function TimeStep!(sim)
    start = now()
    model = sim.model
    temp = sim.temp
    outputs = sim.outputs

    model.iteration += 1
    model.t += sim.ΔT
    model.t %= 1440  #Reset the day at midnight

    chunk_size = 1000 #Process 1,000 individuals at a time.
    print(model.t)
    print(":   ")
    println("start")

    println("spec")
start = now()

    #Add the behavioral context for each species
    for (species_index, _) in enumerate(keys(model.individuals.animals))
        species = model.individuals.animals[species_index]
        t_resolution = species.p.t_resolution[2][species_index]
        if model.t % t_resolution == 0

            # Calculate n_ind outside the GPU loop
            alive = findall(x -> x == 1.0, species.data.ac) #Number of individuals per species that are active
            model.abund[species_index] = length(alive)
            model.bioms[species_index] = sum(model.individuals.animals[species_index].data.biomass[alive])
            if length(alive) > 0
                #Divide into chunks for quicker processing and eliminating memory allocation at one time.
                n = length(alive)
                num_chunks = ceil(Int,n/chunk_size)
                for chunk in 1:num_chunks
                    start_idx = (chunk-1) * chunk_size + 1
                    end_idx = min(chunk*chunk_size,n)
                    chunk_indices = alive[start_idx:end_idx]
                    behavior(model, species_index, chunk_indices, outputs)
                    ind_temp = individual_temp(model, species_index, chunk_indices, temp)
                    respire = respiration(model, species_index, chunk_indices, ind_temp)
                    egest, excrete = excretion(model, species_index,chunk_indices)
                    sda = specific_dynamic_action(species_index,egest,chunk_indices)
                    growth(model, species_index, sda, respire, egest, excrete,chunk_indices)
                    evacuate_gut(model, species_index, chunk_indices, ind_temp)
                    starvation(model,species, species_index, chunk_indices, outputs)
                    reproduce(model,species_index,chunk_indices)
                end
            end
        end 
        species.data.daily_ration[alive] .+= species.data.ration[alive]
        species.data.ration .= 0
        species.data.consumed .= 0
    end
    println(model.abund)
    stop = now()
    println(stop-start)

println("pool")
start = now()
    #Non-focal species processing
    for (pool_index,animal_index) in enumerate(keys(model.pools.pool))
        n = length(model.pools.pool[pool_index].data.length)
        num_chunks = ceil(Int,n/chunk_size)
        for chunk in 1:num_chunks
            start_idx = (chunk-1) * chunk_size + 1
            end_idx = min(chunk*chunk_size,n)
            chunk_indices = start_idx:end_idx
            pool_predation(model,pool_index,chunk_indices,outputs)
        end
    end
    stop = now()
    println(stop-start)

    if (model.t % model.output_dt == 0) & (model.iteration > model.spinup) #Only output results after the spinup is done.
        #timestep_results(sim) #Assign and output
        CSV.write("Mortality Counts.csv",Tables.table(outputs.mortalities))
    end 
    pool_growth(model) #Grow pool individuals back to carrying capacity (initial biomass)  
end