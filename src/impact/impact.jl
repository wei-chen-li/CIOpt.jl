using LinearAlgebra
using MPCCModels
using NLPModels
using SparseArrays

include("aula_config.jl")
include("dual_block.jl")
include("saddle_layout.jl")
include("complementarity_projection.jl")
include("aula_subproblem.jl")
include("gauss_newton_solver.jl")
include("aula_solver.jl")

export ImpactSolver, solve!

"""
    ImpactSolver(mpcc::MPCCModels.AbstractMPCCModel; kwargs...)

A non-vertical MPCC is converted to vertical form. `AulaConfig` is setup from
`kwargs`.
"""
mutable struct ImpactSolver{M,T<:AbstractFloat}
    mpcc::M
    config::AulaConfig
    subproblem::AulaSubproblem{M,T}
    aula_solver::AulaSolver
end

function ImpactSolver(mpcc::MPCCModels.AbstractMPCCModel; kwargs...)
    config = validate(AulaConfig(; kwargs...))
    mpcc = MPCCModels.is_vertical(mpcc) ? mpcc : MPCCModels.vertical_form(mpcc)
    subproblem = AulaSubproblem(mpcc, config)
    return ImpactSolver(mpcc, config, subproblem, AulaSolver())
end

"""
    solve!(solver::ImpactSolver)

Solve the MPCC stored in `solver`, mutating its AuLa and Gauss-Newton state.
Returns the `AulaResult` from this solve.
"""
function solve!(solver::ImpactSolver)
    x_init = MPCCModels.get_x0(solver.mpcc)
    return solve!(solver.aula_solver, solver.subproblem, solver.config, x_init)
end
