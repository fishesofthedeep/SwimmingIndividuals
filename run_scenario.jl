# ===================================================================
# Scenario Execution Script
# ===================================================================
# Usage from the terminal (in the root directory of your project):
#   julia run_scenario.jl
#   OR
#   julia run_scenario.jl examples/Mackerel
# ===================================================================

# 1. Activate the local package environment
using Pkg
Pkg.activate(".")
Pkg.build("CUDA")
# 2. Load the main package
using SwimmingIndividuals

"""
    run_scenario(target_dir::String)

Checks the targeted directory (or the root directory) for the configuration 
file and executes the model.
"""
function run_scenario(target_dir::String)
    @info "Initializing simulation setup..."
    
    # Check if a files.csv exists specifically inside the targeted directory
    dir_config = joinpath(target_dir, "files.csv")
    
    # Check if there is a global files.csv in the root directory (your current setup)
    root_config = "files.csv"
    
    config_to_use = ""

    if isfile(dir_config)
        @info "Found configuration inside scenario directory: $dir_config"
        config_to_use = dir_config
    elseif isfile(root_config)
        @info "Found root configuration file: $root_config. Using this to point to $target_dir."
        config_to_use = root_config
    else
        @error "Could not find a 'files.csv' configuration file in '$target_dir' or the root directory."
        return
    end

    try
        @info "Executing setup_and_run_model with $config_to_use..."
        setup_and_run_model(config_to_use)
        @info "Simulation completed successfully for scenario: $target_dir"
    catch e
        @error "An error occurred during the simulation:" exception=(e, catch_backtrace())
    end
end

# --- Script Execution ---
# Check if the user passed a directory argument from the command line.
# If not, default to the "examples/Mackerel" directory.
target_directory = length(ARGS) > 0 ? ARGS[1] : "examples/Mackerel"

run_scenario(target_directory)