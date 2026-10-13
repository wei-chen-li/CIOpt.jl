using LinearAlgebra
using SparseArrays
import MPCCModels
import NLPModels

include("aula_config.jl")
include("dual_block.jl")
include("saddle_layout.jl")
include("complementarity_projection.jl")
include("aula_subproblem.jl")
include("mpcc_subproblem.jl")
include("gauss_newton_solver.jl")
include("aula_solver.jl")

const ImpactOptions = AulaConfig
const ImpactResult = AulaResult

function solve(model::MPCCModels.AbstractMPCCModel;
               options::ImpactOptions=ImpactOptions())
    sub = buildMPCC(model, options)
    x0 = model.nlp.meta.x0
    return solve!(AulaSolver(), sub, options, x0)
end
