using ADNLPModels
using CIOpt
using MPCCModels
using SparseArrays
using Test

@testset "solve a simple vertical form MPCC" begin
    objective(x) = (x[1] - 1)^2 + (x[2] - 0.2)^2
    nlp = ADNLPModels.ADNLPModel(
        objective,  # f
        [2.0, 2.0], # x0
        [0.0, 0.0], # lvar
        [Inf, Inf], # uvar
    )
    mpcc = MPCCModels.MPCCModel(nlp, [1], [2])
    @assert MPCCModels.is_vertical(mpcc)

    solver = CIOpt.ImpactSolver(mpcc; print_level=0)
    result = CIOpt.solve!(solver)

    tol = 1e-4
    @test result.converged
    @test result.status == :converged
    @test result.x[1] >= -tol
    @test result.x[2] >= -tol
    @test result.x[1] * result.x[2] ≈ 0.0 atol=tol
    @test result.x ≈ [1.0, 0.0] atol=tol
    @test result.objective_value ≈ 0.04 atol=tol
end

@testset "solve a simple non-vertical form MPCC" begin
    objective(x) = (x[1] - 1)^2 + (x[2] - 0.2)^2
    nlp = ADNLPModels.ADNLPModel(
        objective,  # f
        [2.0, 2.0], # x0
        sparse([1.0 0.0; 0.0 1.0]), # A
        [0.0, 0.0],         # lcon
        [Inf, Inf],         # ucon
    )
    mpcc = MPCCModels.MPCCModelConCon(nlp, [1], [2])
    @assert !MPCCModels.is_vertical(mpcc)

    solver = CIOpt.ImpactSolver(mpcc; print_level=0)
    result = CIOpt.solve!(solver)

    tol = 1e-4
    @test result.converged
    @test result.status == :converged
    @test result.x[1] >= -tol
    @test result.x[2] >= -tol
    @test result.x[1] * result.x[2] ≈ 0.0 atol=tol
    @test result.x[1:2] ≈ [1.0, 0.0] atol=tol
    @test result.objective_value ≈ 0.04 atol=tol
end
