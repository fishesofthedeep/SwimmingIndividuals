# ===================================================================
# Main Simulation Driver
# ===================================================================

"""
    runSI(sim::MarineSimulation)

A simple wrapper function to start the simulation. This can be used as the
main entry point in your `model.jl` script.
"""
function runSI(sim::MarineSimulation)
    println("✅ Model Initialized. Starting simulation run...")
    if sim.model.iteration > 0
        println("   Resuming from iteration $(sim.model.iteration).")
    end
    while sim.model.iteration < sim.iterations
        TimeStep!(sim)
    end
    println("✅ Simulation run complete.")
end
