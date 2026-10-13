using ADNLPModels
using CIOpt
using MPCCModels
using Test

@testset "solve a simple vertical MPCC" begin
    objective(x) = (x[1] - 1)^2 + (x[2] - 0.2)^2
    nlp = ADNLPModels.ADNLPModel(
        objective,
        [2.0, 2.0],
        [0.0, 0.0],
        [Inf, Inf],
    )
    mpcc = MPCCModels.MPCCModel(nlp, [1], [2])
    @assert MPCCModels.is_vertical(mpcc)

    result = CIOpt.solve(
        mpcc;
        options=CIOpt.ImpactOptions(print_level=0)
    )

    tol = 1e-4
    @test result.converged
    @test result.status == :converged
    @test result.x[1] >= -tol
    @test result.x[2] >= -tol
    @test result.x[1] * result.x[2] ≈ 0.0 atol=tol
    @test result.x ≈ [1.0, 0.0] atol=tol
    @test result.objective_value ≈ 0.04 atol=tol
end
