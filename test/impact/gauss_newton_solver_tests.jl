using LinearAlgebra
using SparseArrays
using Test
include("../../src/impact/gauss_newton_solver.jl")

# Saddle vs normal-equations Gauss-Newton.
#   ℓ(x; p) = ½‖x - a‖²,  r(x; p) = √rho * (x - b),  p = [a; b; rho].
# The minimiser is (a + rho*b) / (1 + rho).
function solve_ls(use_saddle, a, b, rho)
    n = length(a)
    ℓ(x, p) = sum(abs2, x - p[1:n]) / 2
    ∇ℓ(x, p) = x - p[1:n]
    ∇²ℓ(x, p) = spdiagm(0 => ones(n))
    residual(x, p) = sqrt(p[end]) * (x - p[n+1:2n])
    jacobian(x, p) = spdiagm(0 => fill(sqrt(p[end]), n))

    config = GaussNewtonConfig(use_saddle=use_saddle, max_iter=50, grad_tol=1e-12)
    solver = GaussNewtonSolver(ℓ, ∇ℓ, ∇²ℓ, residual, jacobian, config)
    params = vcat(a, b, rho)
    x = zeros(n)
    solve!(solver, params, x)

    x_star = (a + rho * b) / (1 + rho)
    return norm(x - x_star, Inf)
end

@testset "Gauss-Newton normal vs saddle" begin
    a = [1.0, -2.0, 0.5, 3.0]
    b = [0.0, 1.0, -1.0, 2.0]
    for rho in (1.0, 100.0, 1e4)
        @testset "rho = $rho" begin
            err_normal = solve_ls(false, a, b, rho)
            err_saddle = solve_ls(true, a, b, rho)
            @test err_normal < 1e-8
            @test err_saddle < 1e-8
        end
    end
end
