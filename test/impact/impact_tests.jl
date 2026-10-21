using Test

@testset "IMPACT Tests" begin
include("aula_config_tests.jl")
include("aula_solver_tests.jl")
include("aula_subproblem_tests.jl")
include("complementarity_projection_tests.jl")
include("gauss_newton_solver_tests.jl")
end
