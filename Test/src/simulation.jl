mutable struct MarineOutputWriter
    filepath::String
    write_log::Bool
    save_diags::Bool
    save_plankton::Bool
    diags_file::String
    plankton_file::String
    plankton_include::Tuple
    plankton_iteration_interval::Int
    max_filesize::Number # in Bytes
    part_diags::Int
    part_plankton::Int
end

mutable struct MarineOutputs
    #depths::Vector{Float64}
    #daily_ration::Vector{Float64}
    #trophiclevel::Vector{Float64}
    mortalities::Matrix{Float64}
    biomass::Array{Float64,3}
    #consumption_biomass::Matrix{Float64}
    consumption::Array{Float64,4}
    production::Matrix{Float64}
    #production_biomass::Matrix{Float64}
    pool_density::Matrix{Float64}
    diet_composition::Matrix{Float64}
    timestep_array::Array{Float64,3}
end

mutable struct MarineInput
    #salt::AbstractArray{Float64,4}      # salinity
    #disox::AbstractArray{Float64,4}     # dissolved oxygen
    #PARF::AbstractArray{Float64,3}      # PARF
    #vels::NamedTuple                    # velocity fields for nutrients and individuals
    #ΔT_vel::Float64                     # time step of velocities provided
    #ΔT_PAR::Float64                     # time step of surface PAR provided
    #ΔT_temp::Float64                    # time step of temperature provided
    #ΔT_salt::Float64                    # time step of salinity provided
    #ΔT_disox::Float64                   # time step of dissolved oxygen provided
end

mutable struct MarineSimulation
    model::MarineModel                       # Model object
    #input::MarineInput                       # model input, temp, PAR, and velocities
    #diags::Union{PlanktonDiagnostics,Nothing}  # diagnostics
    ΔT::Float64                                  # model time step
    iterations::Int64                          # run the simulation for this number of iterations
    temp::Vector{Float64}                       # temperature
    outputs::MarineOutputs
    #output_writer::Union{MarineOutputWriter,Nothing} # Output writer
end

function simulation(model::MarineModel, ΔT::Float64, iterations::Int64,temp::Vector,outputs::MarineOutputs)

#input = MarineInput(temp, PARF, vels, ΔT_vel, ΔT_PAR, ΔT_temp)

#validate_bcs(model.nutrients, model.grid, iterations)

#if diags == nothing
#diags = MarineDiagnostics(model)
#end

sim = MarineSimulation(model, ΔT, iterations,temp,outputs)

#validate_velocity(sim, model.grid)
#validate_PARF(sim, model.grid)
#validate_temp(sim, model.grid)

return sim
end

function MarineOutputWriter(;dir = "./results",
    diags_prefix = "diags",
    plankton_prefix = "plankton",
    write_log = false,
    save_diags = false,
    save_plankton = false,
    plankton_include = (:x, :y, :z, :length),
    plankton_iteration_interval = 1,
    max_filesize = Inf
    )

isdir(dir) && rm(dir, recursive=true)
mkdir(dir)

diags_file = ""
plankton_file = ""

if save_diags
diags_file = joinpath(dir, diags_prefix*".jld2")
end
if save_plankton
plankton_file = joinpath(dir, plankton_prefix*".jld2")
end

return MarineOutputWriter(dir, write_log, save_diags, save_plankton, diags_file, plankton_file,
     plankton_include, plankton_iteration_interval, max_filesize, 1, 1)
end