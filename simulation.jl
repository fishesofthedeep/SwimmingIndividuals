mutable struct MarineModel
    arch::Architecture          # architecture on which models will run
    t::Float64                  # time in second
    iteration::Int64            # model interation
    individuals::individuals    # initial individuals generated by `setup_agents`
    grid::AbstractGrid          # grid information
    #timestepper::timestepper    # Add back in once environmental parameters get involved
end

mutable struct MarineInput
    temp::AbstractArray{Float64,4}      # temperature
    salt::AbstractArray{Float64,4}      # salinity
    disox::AbstractArray{Float64,4}     # dissolved oxygen
    PARF::AbstractArray{Float64,3}      # PARF
    vels::NamedTuple                    # velocity fields for nutrients and individuals
    ΔT_vel::Float64                     # time step of velocities provided
    ΔT_PAR::Float64                     # time step of surface PAR provided
    ΔT_temp::Float64                    # time step of temperature provided
    ΔT_salt::Float64                    # time step of salinity provided
    ΔT_disox::Float64                   # time step of dissolved oxygen provided
end

mutable struct MarineSimulation
    model::MarineModel                       # Model object
    #input::MarineInput                       # model input, temp, PAR, and velocities
    #diags::Union{PlanktonDiagnostics,Nothing}  # diagnostics
    ΔT::Float64                                  # model time step
    iterations::Int64                          # run the simulation for this number of iterations
    output_writer::Union{MarineOutputWriter,Nothing} # Output writer
end


function simulation(model::MarineModel; ΔT::Float64, iterations::Int64,
    PARF = default_PARF(model.grid, ΔT, iterations),
    temp = default_temperature(model.grid, ΔT, iterations),
    diags = nothing,
    vels = (;),
    ΔT_vel::Float64 = ΔT,
    ΔT_PAR::Float64 = 3600.0,
    ΔT_temp::Float64 = 3600.0,
    output_writer = nothing
    )

#input = MarineInput(temp, PARF, vels, ΔT_vel, ΔT_PAR, ΔT_temp)

#validate_bcs(model.nutrients, model.grid, iterations)

#if diags == nothing
#diags = MarineDiagnostics(model)
#end

sim = MarineSimulation(model, ΔT, iterations, output_writer)

#validate_velocity(sim, model.grid)
#validate_PARF(sim, model.grid)
#validate_temp(sim, model.grid)

return sim
end
