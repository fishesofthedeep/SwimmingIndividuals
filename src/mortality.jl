function predation_mortality(model::MarineModel,df,outputs)
    if model.iteration > model.spinup
        model.individuals.animals[df.Sp[1]].data.x[df.Ind[1]] = 5e6
        model.individuals.animals[df.Sp[1]].data.y[df.Ind[1]] = 5e6
        model.individuals.animals[df.Sp[1]].data.ac[df.Ind[1]] = 0.0
        model.individuals.animals[df.Sp[1]].data.energy[df.Ind[1]] = -500
        model.individuals.animals[df.Sp[1]].data.behavior[df.Ind[1]] = 4
        model.individuals.animals[df.Sp[1]].data.biomass[df.Ind[1]] = 0
        outputs.mortalities[df.Sp[1],1] += 1 #Add one to the predation mortality column
    end
    return nothing
end

function starvation(model,dead_sp, sp, i, outputs)
    starve = findall(x -> x < 0,dead_sp.data.energy[i])
    if model.iteration > model.spinup .& length(starve) > 0
        dead_sp.data.x[i[starve]] .= 5e6
        dead_sp.data.y[i[starve]] .= 5e6
        dead_sp.data.ac[i[starve]] .= 0.0
        dead_sp.data.biomass[i[starve]] .= 0.0
        dead_sp.data.behavior[i[starve]] .= 4
        outputs.mortalities[sp,2] += length(starve) #Add one to the starvation mortality column
        #outputs.production[model.iteration,sp] .+= model.individuals.animals[sp].data.weight[i] #For P/B iteration
    end
    return nothing
end

function reduce_pool(model,pool,ind,ration)
    model.pools.pool[pool].data.biomass[ind] -= ration[1]

    if model.pools.pool[pool].data.biomass[ind] == 0 #Make sure pool actually stays alive
        model.pools.pool[pool].data.biomass[ind] = 1e-9
    end
    return nothing
end