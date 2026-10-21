using CIOpt
using ADNLPModels
using MPCCModels
using LinearAlgebra
using SparseArrays
using Test

# Generic MPCC models: min ||z-target||² with variable complementarity pairs.
function make_mpcc(target, x0)
    n = length(target)
    nlp = ADNLPModels.ADNLPModel(x->sum(abs2,x-target), x0, zeros(n), fill(Inf,n))
    mpcc = MPCCModels.MPCCModel(nlp, collect(1:2:n), collect(2:2:n))
    return mpcc
end

function make_config(; use_saddle=true)
    return (; max_outer_iters=300, max_inner_iters=50,
              rho_scale=1.5, rho_max=1e8,
              outer_tol_h=1e-7, outer_tol_comp=1e-7,
              newton_max_iter=50, use_saddle=use_saddle,
              print_level=0)
end

# A generic, non-trajectory MPCC solved directly with buildMPCC().
#   minimize (z1 - 1)^2 + (z2 - 1)^2   s.t.   0 <= z1 ⊥ z2 >= 0
# The complementarity forces one of z1, z2 to zero, so the optimum is (1, 0) or
# (0, 1) with objective 1.
@testset "Generic MPCC" begin
    for use_saddle in (true, false)
        z0 = [0.5 ,0.5]
        mpcc = make_mpcc(ones(2), z0)
        config = make_config(; use_saddle)
        solver = CIOpt.ImpactSolver(mpcc; config...)
        result = CIOpt.solve!(solver)

        comp = abs(result.z[1] * result.z[2])

        @test result.converged
        @test result.status == :converged
        @test comp < 1e-5
        @test result.objective_value ≈ 1.0 atol=1e-2
    end
end

@testset "Generic MPCC Multi-Comp" begin
    target = [0.8, 0.2, 0.25, 0.75]
    z0 = zeros(4)
    mpcc = make_mpcc(target, z0)
    config = make_config()
    solver = CIOpt.ImpactSolver(mpcc; config...)
    result = CIOpt.solve!(solver)
    comp0 = abs(result.z[1] * result.z[2])
    comp1 = abs(result.z[3] * result.z[4])

    @test result.converged
        @test result.status == :converged
    @test comp0 < 1e-5
    @test comp1 < 1e-5
    @test result.objective_value ≈ 0.1025 atol=1e-2
end

@testset "AulaSolver reuse across solve" begin
    solver = CIOpt.AulaSolver()
    gn = nothing
    for (target,expected,use_saddle) in (([1.0,0.2],[1.0,0.0],true),
                                         ([0.1,2.0],[0.0,2.0],true),
                                         ([3.0,0.1],[3.0,0.0],false))
        z0 = [0.5,0.5]
        mpcc = make_mpcc(target, z0)
        config = CIOpt.AulaConfig(; make_config(;use_saddle)...)
        sub = CIOpt.AulaSubproblem(mpcc, config)
        result = CIOpt.solve!(solver, sub, config, z0)

        @test result.converged
        @test result.z ≈ expected atol=1e-4
        @test solver.gn.config.use_saddle == use_saddle
        if gn !== nothing
            @test solver.gn === gn
        end
        gn = solver.gn
    end
end
