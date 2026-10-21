using LinearAlgebra
using MPCCModels
using NLPModels
using SparseArrays

include("aula_config.jl")
include("saddle_layout.jl")
include("complementarity_projection.jl")
include("aula_subproblem.jl")
include("gauss_newton_solver.jl")
include("aula_solver.jl")

export ImpactSolver, solve!

"""
    ImpactSolver(mpcc::MPCCModels.AbstractMPCCModel; kwargs...)

`kwargs` accepts keyword arguments corresponding to fields of `AulaConfig`.
"""
mutable struct ImpactSolver
    aula_solver::AulaSolver
    aula_subproblem::AulaSubproblem
    aula_config::AulaConfig
end

function ImpactSolver(mpcc::MPCCModels.AbstractMPCCModel; kwargs...)
    aula_config = validate(AulaConfig(; kwargs...))
    aula_subproblem = AulaSubproblem(mpcc, aula_config)
    return ImpactSolver(AulaSolver(), aula_subproblem, aula_config)
end

"""
    solve!(solver::ImpactSolver)

Solve the MPCC stored in `solver`, mutating its AuLa and Gauss-Newton state.
Returns the `AulaResult` from this solve.
"""
function solve!(solver::ImpactSolver)
    s = solver
    x_init = NLPModels.get_x0(s.aula_subproblem._mpcc)
    return solve!(s.aula_solver, s.aula_subproblem, s.aula_config, x_init)
end
