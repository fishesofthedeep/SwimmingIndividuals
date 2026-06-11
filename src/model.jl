# ===================================================================
# High-Level Model Setup and Execution
# ===================================================================

# Look up an optional row in the params table; return `default` if absent.
function _opt_param(params::DataFrame, name::AbstractString, default)
    idx = findfirst(==(name), params.Name)
    idx === nothing && return default
    return params[idx, :Value]
end

"""
    setup_and_run_model(config_filename="files.csv")

The primary entry point for setting up the environment, initializing 
agents/resources, and driving the simulation loop.
"""
function setup_and_run_model(config_filename="files.csv")
    # Identify the path to the config file located in the root directory
    # (One level up from this file's location in src/)
    base_path = joinpath(@__DIR__, "..")
    config_path = joinpath(base_path, config_filename)

    if !isfile(config_path)
        error("Configuration file not found at: $config_path")
    end

    ## 1. Load configuration databases
    files = CSV.read(config_path, DataFrame)
    
    # Resolve scenario and results directories
    scen_dir_row = filter(row -> row.File == "scen_dir", files)
    scen_dir = scen_dir_row[1, :Destination]
    
    # Update destinations to be absolute or scenario-relative
    files.Destination = [
        row.File == "scen_dir" ? row.Destination : joinpath(scen_dir, row.Destination) 
        for row in eachrow(files)
    ]
    
    # Because we just updated files.Destination, res_dir ALREADY has the full path.
    # We just need to extract it and make the directory.
    res_dir_row = filter(row -> row.File == "res_dir", files)
    full_res_path = res_dir_row[1, :Destination] 
    mkpath(full_res_path)

    # Load traits, parameters, and grid settings
    trait = Dict(pairs(eachcol(CSV.read(files[files.File .== "focal_trait", :Destination][1], DataFrame))))
    resource_trait = CSV.read(files[files.File .== "resource_trait", :Destination][1], DataFrame)
    params = CSV.read(files[files.File .== "params", :Destination][1], DataFrame)
    grid = CSV.read(files[files.File .== "grid", :Destination][1], DataFrame)
    fisheries_df = CSV.read(files[files.File .== "fisheries", :Destination][1], DataFrame)
    # NetCDF environment file is OPTIONAL now: only needed when ASC forcing
    # (env_xml) is not configured. Look it up safely so a missing row / file does
    # not error when running purely from the .asc XML.
    envi_file = any(files.File .== "environment") ? files[files.File .== "environment", :Destination][1] : ""

    ## 2. Global simulation settings
    Nsp = parse(Int32, params[params.Name .== "numspec", :Value][1])
    Nresource = parse(Int32, params[params.Name .== "numresource", :Value][1])
    output_dt_minutes = parse(Int32, params[params.Name .== "output_dt", :Value][1])
    spinup = parse(Int32, params[params.Name .== "spinup", :Value][1])
    plt_diags = parse(Int32, params[params.Name .== "plt_diags", :Value][1])
    foraging_attempts = parse(Int32, params[params.Name .== "num_foraging_attempts", :Value][1])
    n_iteration = parse(Int32, params[params.Name .== "nts", :Value][1])
    dt = parse(Int32, params[params.Name .== "model_dt", :Value][1])
    n_iters = parse(Int16, params[params.Name .== "n_iter", :Value][1])
    
    # maxN is now configurable (param "maxN"); falls back to the historical default.
    maxN = Int64(parse(Int, string(_opt_param(params, "maxN", "500000"))))
    arch_str = params[params.Name .== "architecture", :Value][1]
    output_dt = Int32(max(1, round(output_dt_minutes / dt)))

    # --- Optional run controls (all default to previous behaviour if absent) ---
    # Reproducibility: seed host (and device) RNGs when a seed is provided.
    seed_val = _opt_param(params, "seed", "")
    if seed_val !== "" && seed_val !== missing && string(seed_val) != ""
        s = parse(Int, string(seed_val))
        Random.seed!(s)
        try
            CUDA.functional() && CUDA.seed!(s)
        catch err
            @warn "Could not seed CUDA RNG: $err"
        end
        @info "RNG seeded with $s for reproducibility."
    end

    # Performance/IO cadences (see RUNTIME_CONFIG in timestep.jl).
    RUNTIME_CONFIG[:merge_every]      = Int(parse(Int, string(_opt_param(params, "merge_every", "1"))))
    RUNTIME_CONFIG[:checkpoint_every] = Int(parse(Int, string(_opt_param(params, "checkpoint_every", "0"))))
    restart_requested = string(_opt_param(params, "restart", "0")) in ("1", "true", "TRUE")

    # Headless plotting backend for servers (no display). Only relevant if diagnostics on.
    if plt_diags > 0
        ENV["GKSwstype"] = "100"
    end

    # Handle Hardware Architecture
    if arch_str == "GPU"
        if CUDA.functional()
            arch = GPU()
            @info "✅ Architecture successfully set to GPU."
        else
            @warn "GPU specified but CUDA is not functional. Falling back to CPU."
            arch = CPU()
        end
    else
        arch = CPU()
        @info "✅ Architecture successfully set to CPU."
    end

    # --- Model start date (configurable via params.csv) ------------------------
    # `start_year` is the post-spinup "real" calendar year the simulation begins;
    # `start_month`/`start_day` are optional (default Jan 1). This date drives both
    # the calendar in TimeStep! (via MODEL_START_DATETIME) and initial agent
    # placement. Absent -> defaults to 2026-01-01 so older configs are unchanged.
    start_year  = Int(parse(Int, string(_opt_param(params, "start_year",  "2026"))))
    start_month = Int(parse(Int, string(_opt_param(params, "start_month", "1"))))
    start_day   = Int(parse(Int, string(_opt_param(params, "start_day",   "1"))))
    start_date  = Date(start_year, start_month, start_day)
    MODEL_START_DATETIME[] = DateTime(start_date)
    @info "Model start date set to $start_date (post-spinup)."

    ## 3. Environment and Infrastructure Initialization
    depths = generate_depths(files)

    # Environment source: time-varying ESRI .asc forcing from STConfig.xml when a
    # files.csv row File == "env_xml" is present (NetCDF is then NOT used at all),
    # otherwise the legacy NetCDF environment.nc path. The .asc path builds the
    # initial environment + habitat capacities directly from the rasters; from the
    # first timestep on, update_environment_from_asc! refreshes them each month.
    ENV_FORCING[] = nothing
    if any(files.File .== "env_xml")
        xml_path = files[files.File .== "env_xml", :Destination][1]
        base_dir = any(files.File .== "env_dir") ? files[files.File .== "env_dir", :Destination][1] : ""
        isfile(xml_path) || error("env_xml row present but file not found at: $xml_path")
        ENV_FORCING[] = load_env_forcing(xml_path, grid; base_dir=base_dir)
        @info "Environment source: time-varying ASC forcing from $xml_path (NetCDF disabled)."
        envi, capacities = bootstrap_environment_from_asc(ENV_FORCING[], files, grid, arch,
                                                          start_date, Nsp, Nresource, plt_diags)
    else
        envi = generate_environment!(arch, envi_file, plt_diags, files)
        capacities = initial_habitat_capacity(envi, Nsp, Nresource, files, arch, plt_diags)
    end

    ## 4. Multi-Run Simulation Loop
    for iter in 1:n_iters
        @info "--- Starting Simulation Run $iter ---"
        
        B = Float32.(trait[:Biomass][1:Nsp])

        # Initialize Agents and Resources
        inds, daily_birth_counters = generate_individuals(trait, arch, Nsp, B, maxN, depths, capacities, dt, envi, start_date)
        resources = initialize_resources(resource_trait, Nsp, Nresource, depths, capacities, arch)
        fishery_fleet = load_fisheries(fisheries_df, dt)

        # Pre-simulation summary
        init_abund = fill(0, Nsp)
        bioms = fill(0.0f0, Nsp) # FIXED: Strict 32-bit float array
        for sp in 1:Nsp
            init_abund[sp] = sum(inds.animals[sp].data.abundance)
            bioms[sp] = sum(inds.animals[sp].data.biomass_school)
        end

        # Generate size-bin thresholds for mortality/outputs.
        # Uses create_size_bin_matrix (utilities.jl): returns an (n_bins+1) x
        # n_total_species matrix of log-spaced thresholds covering BOTH the focal
        # species and the resource groups, matching what find_species_size_bin
        # expects. (Replaces the older focal-only (Nsp, n_bins) form.)
        n_bins = Int32(10)
        size_bin_thresholds = create_size_bin_matrix(trait, resource_trait, n_bins, arch)

        # Create the high-level model object
        # FIXED: Explicitly cast variables to strict Float32/Int32 to match constructor exactly
        model = MarineModel(
            arch, envi, depths, fishery_fleet, 0.0f0, Int32(0), Float32(dt), 
            inds, resources, resource_trait, capacities, 
            maxN, Nsp, Nresource, init_abund, bioms, init_abund, 
            files, output_dt, spinup, foraging_attempts, plt_diags,
            size_bin_thresholds, daily_birth_counters
        )

        # Setup outputs
        outputs = generate_outputs(model, n_bins)

        # Initialise biomass-linked quotas for the first year (mode-2 fisheries);
        # thereafter they are recomputed each Jan 1 in TimeStep!.
        update_dynamic_quotas!(model)

        # Setup and Run Simulation
        sim = MarineSimulation(model, dt, n_iteration, iter, outputs)

        # Resume from a checkpoint if requested and one exists for this run.
        if restart_requested
            ckpt = joinpath(full_res_path, "Checkpoint", "checkpoint_run$(iter).jls")
            if isfile(ckpt)
                load_checkpoint!(sim, ckpt)
            else
                @info "Restart requested but no checkpoint found for run $iter; starting fresh."
            end
        end

        runSI(sim)
    end
    
    @info "Simulation sequence complete."
    return nothing
end