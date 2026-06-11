"""
    SwimmingIndividuals

A high-performance agent-based model (ABM) for simulating individual-based marine 
ecosystem dynamics. This module encapsulates life histories, behaviors, 
bioenergetics, predation, and fishing impacts within a modular, 
architecture-aware framework.
"""
module SwimmingIndividuals

# --- 1. EXTERNAL DEPENDENCIES ---
# All dependencies listed in Project.toml must be loaded here.
using PlanktonIndividuals
using Distributions
using Random
using CSV
using DataFrames
using StructArrays
using Statistics
using Dates
using Optim
using LinearAlgebra
using Tables
using CUDA
using LoopVectorization
using NCDatasets
using StaticArrays
using Interpolations
using DelimitedFiles
using StatsBase
using Plots
using NearestNeighbors
using QuadGK
using Printf
using HDF5
using Serialization

# Specific sub-components and hardware abstraction tools
using PlanktonIndividuals.Grids
using PlanktonIndividuals.Architectures: device, Architecture, GPU, CPU, rng_type, array_type
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using CUDA: @atomic, atomic_cas!, atomic_sub!, @cuprintf

# --- 2. SOURCE FILE INCLUSIONS ---
# The order of inclusion is critical to ensure types and utilities are 
# defined before they are used by high-level logic.
include("utilities.jl")   # General helper functions and math
include("asc_environment.jl") # Model-native XML / ESRI .asc time-varying forcing (defines ENV_FORCING)
include("create.jl")      # Agent and Resource construction/initialization
include("environment.jl") # NetCDF loading and habitat suitability
include("simulation.jl")  # Simulation and Model structs
include("output.jl")      # Data structures for simulation results
include("behavior.jl")    # Perception and decision-making logic
include("movement.jl")    # Pathfinding and vertical migration
include("predation.jl")   # Foraging and consumption resolution
include("mortality.jl")   # Survival and dead agent management
include("fisheries.jl")   # Fleet dynamics and harvest logic
include("energy.jl")      # Metabolic costs and growth partitioning
include("timestep.jl")    # Timestep orchestration
include("analysis.jl")    # Post-processing and data export
include("update.jl")      # State updates and maintenance
include("model.jl")       # High-level entry point (setup_and_run_model)

# --- 3. PUBLIC API EXPORTS ---
# These functions and types are available to the user upon `using SwimmingIndividuals`
export 
    # Core Objects
    MarineModel, 
    MarineSimulation, 
    MarineOutputs,

    # Primary Entry Point
    setup_and_run_model,

    # Component Launchers
    runSI,
    generate_environment!,
    generate_individuals,
    initialize_resources,
    load_fisheries,

    # Checkpoint / Restart
    save_checkpoint,
    load_checkpoint!,

    # Time-varying ASC environmental forcing
    load_env_forcing,
    update_environment_from_asc!,
    parse_env_config

end # module SwimmingIndividuals