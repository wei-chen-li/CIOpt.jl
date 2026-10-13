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

function solve(mpcc::MPCCModels.AbstractMPCCModel;
               options::ImpactOptions=ImpactOptions())
    vertical = MPCCModels.is_vertical(mpcc)
    if !vertical
        mpcc = MPCCModels.vertical_form(mpcc)
    end
    x0 = if vertical; mpcc.nlp.meta.x0 else mpcc.nlp.meta.parent.x0 end
    return solve!(AulaSolver(), buildMPCC(mpcc, options), options, x0)
end
