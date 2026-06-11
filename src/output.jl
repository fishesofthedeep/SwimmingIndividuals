# ===================================================================
# Data Structures and Functions for Model Output
# ===================================================================

"""
    generate_outputs(model::MarineModel)
This function creates the data structures for storing simulation results.
It is architecture-aware and will create arrays on the GPU if the model
is configured to run on a GPU.
"""
function generate_outputs(model::MarineModel, n_bins::Int32)
    arch = model.arch
    grid = model.depths.grid
    
    depthres = Int(grid[grid.Name .== "depthres", :Value][1])
    latres = Int(grid[grid.Name .== "latres", :Value][1])
    lonres = Int(grid[grid.Name .== "lonres", :Value][1])
    n_fish = length(model.fishing)
    n_total_species = model.n_species + model.n_resource

    # --- Create output arrays with the new PREDATOR size dimension ---
    # Dimensions are: [lon, lat, depth, pred_sp, prey_sp, pred_bin, prey_bin]
    
    consumption = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_total_species, n_total_species, n_bins, n_bins))
    
    # Mortalities track predation ON agents.
    # NOTE: these accumulate *biomass* (Float32 grams) during a step; they are
    # converted to instantaneous rates at output time. They MUST be Float32 to
    # match the MarineOutputs struct and to avoid truncating fractional grams.
    Smort = array_type(arch)(zeros(Float32, lonres, latres, depthres, model.n_species, n_bins))

    # Predation mortality ON agents (biomass g) and "other" mortality (senescence
    # and any non-starvation/non-predation/non-fishing death). Same shape as Smort.
    Pmort = array_type(arch)(zeros(Float32, lonres, latres, depthres, model.n_species, n_bins))
    Omort = array_type(arch)(zeros(Float32, lonres, latres, depthres, model.n_species, n_bins))

    # Fishing mortality is ON agents FROM fisheries (biomass caught, Float32 g)
    Fmort = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_fish, model.n_species, n_bins))
    
    abundance = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_total_species,n_bins))
    biomass = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_total_species,n_bins))
    # Standing biomass-by-size at the start of the current output interval; used as
    # the denominator when converting accumulated kill-biomass into instantaneous
    # rates, so F/S/P are deaths-over-interval / population-at-start-of-interval.
    biomass_ref = array_type(arch)(zeros(Float32, lonres, latres, depthres, n_total_species,n_bins))

    return MarineOutputs(Fmort, Smort, Pmort, Omort, consumption, abundance, biomass, biomass_ref)
end