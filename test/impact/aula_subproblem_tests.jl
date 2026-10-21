module AulaSubproblemTests

using ADNLPModels
using CIOpt
using LinearAlgebra
using MPCCModels
using NLPModels
using SparseArrays
using Test

# Deliberately has no underlying NLP, metadata, or private work buffers. An
# AbstractMPCCModel implementation may provide the public API directly.
struct PublicAPIMPCC{T} <: MPCCModels.AbstractMPCCModel{T,Vector{T}}
    start::Vector{T}
end

NLPModels.get_nvar(::PublicAPIMPCC) = 4
NLPModels.get_ncon(::PublicAPIMPCC) = 3
NLPModels.get_x0(model::PublicAPIMPCC) = model.start
NLPModels.get_minimize(::PublicAPIMPCC) = false
NLPModels.get_lvar(::PublicAPIMPCC{T}) where {T} = T[2, 3, -1, 4]
NLPModels.get_uvar(::PublicAPIMPCC{T}) where {T} = T[8, Inf, 5, 4]
NLPModels.get_lcon(::PublicAPIMPCC{T}) where {T} = T[1, 2, -Inf]
NLPModels.get_ucon(::PublicAPIMPCC{T}) where {T} = T[1, 6, 3]
MPCCModels.get_ncc(::PublicAPIMPCC) = 2
MPCCModels.get_cc_var_indices(::PublicAPIMPCC) = [1, 2]
MPCCModels.lcomp_left(::PublicAPIMPCC{T}) where {T} = T[2, 5]
MPCCModels.lcomp_right(::PublicAPIMPCC{T}) where {T} = T[3, 7]

function NLPModels.obj(::PublicAPIMPCC, x::AbstractVector)
    return x[1]^2 + 2x[2]^2 + 3x[3]^2 + 4x[4]^2 + x[1] * x[2]
end

function NLPModels.grad(::PublicAPIMPCC{T}, x::AbstractVector{T}) where {T}
    return T[2x[1] + x[2], 4x[2] + x[1], 6x[3], 8x[4]]
end

function NLPModels.hess(::PublicAPIMPCC{T}, x::AbstractVector;
                        obj_weight::Real=one(T)) where {T}
    return obj_weight * sparse(T[2 1 0 0; 1 4 0 0; 0 0 6 0; 0 0 0 8])
end

function NLPModels.cons(::PublicAPIMPCC{T}, x::AbstractVector) where {T}
    return T[x[1] + x[3], x[2] * x[4], x[3] - 2x[4]]
end

function NLPModels.jac(::PublicAPIMPCC{T}, x::AbstractVector) where {T}
    return sparse(T[1 0 1 0; 0 x[4] 0 x[2]; 0 0 1 -2])
end

function MPCCModels.comp_res_left!(::PublicAPIMPCC{T}, x::AbstractVector{T},
                                   out::AbstractVector{T}) where {T}
    out[1] = x[1] - 2
    out[2] = x[2] * x[3] - 5
    return out
end

function MPCCModels.comp_res_right!(::PublicAPIMPCC{T}, x::AbstractVector{T},
                                    out::AbstractVector{T}) where {T}
    out[1] = x[2] - 3
    out[2] = x[4] + 2x[1] - 7
    return out
end

function MPCCModels.jac_comp_left(::PublicAPIMPCC{T}, x::AbstractVector{T}) where {T}
    return sparse(T[1 0 0 0; 0 x[3] x[2] 0])
end

function MPCCModels.jac_comp_right(::PublicAPIMPCC{T}, x::AbstractVector{T}) where {T}
    return sparse(T[0 1 0 0; 2 0 0 1])
end

# The order of bound rows is an adapter implementation detail. Match complete
# value/Jacobian rows, consuming each actual row once to detect duplicates.
function check_bound_rows(values, jacobian, expected_values, expected_jacobian)
    @test size(jacobian) == size(expected_jacobian)
    @test length(values) == length(expected_values)
    unmatched = collect(eachindex(values))
    for (value, row) in zip(expected_values, eachrow(expected_jacobian))
        match = findfirst(i -> isapprox(values[i], value) &&
                              isapprox(Vector(jacobian[i, :]), row), unmatched)
        @test match !== nothing
        match === nothing || deleteat!(unmatched, match)
    end
end

function finite_difference_jacobian(callback, x, params)
    step = 1e-6
    columns = map(eachindex(x)) do j
        plus, minus = copy(x), copy(x)
        plus[j] += step
        minus[j] -= step
        (callback(plus, params) - callback(minus, params)) / (2step)
    end
    return hcat(columns...)
end

@testset "AulaSubproblem public MPCC API" begin
    model = PublicAPIMPCC([2.5, 3.5, 0.25, 4.5])
    config = CIOpt.AulaConfig(rho_eq_init=2, rho_ineq_init=3, rho_comp_init=5,
                              print_level=0)
    sub = CIOpt.AulaSubproblem(model, config)
    x = copy(NLPModels.get_x0(model))
    f, p = sub.functions, CIOpt.make_params(sub)

    @test sub.model === model
    @test sub.nvar == 4
    @test !hasproperty(sub, :params)
    @test sub.params_info isa NamedTuple
    @test (sub.params_info.neq, sub.params_info.nineq, sub.params_info.ncomp) == (2, 6, 2)
    @test p isa NamedTuple
    @test !hasproperty(p,:data)
    @test p.rho_eq == fill(2.0, 2)
    @test p.rho_ineq == fill(3.0, 6)
    @test p.rho_comp == fill(5.0, 2)
    for (name, dimension) in ((:kappa_eq, 2), (:kappa_ineq, 6), (:kappaG, 2),
                              (:kappaH, 2), (:sG, 2), (:sH, 2))
        @test getproperty(p, name) == zeros(dimension)
    end

    # Maximization changes the inner objective's sign while objective reports
    # the model's original objective value.
    @test f.objective(x, p) == NLPModels.obj(model, x)
    @test f.ℓ(x, p) == -NLPModels.obj(model, x)
    @test f.∇ℓ(x, p) == -NLPModels.grad(model, x)
    @test f.∇²ℓ(x, p) == -NLPModels.hess(model, x)

    check_bound_rows(f.h(x, p), f.jac_h(x, p),
                     [x[4] - 4, x[1] + x[3] - 1],
                     [0.0 0 0 1; 1 0 1 0])
    # The lower bounds of complementarity variables x1/x2 are represented by
    # G/H. Their finite upper bounds, other bounds, and fixed x4 are retained.
    check_bound_rows(f.g(x, p), f.jac_g(x, p),
                     [x[1] - 8, -1 - x[3], x[3] - 5,
                      2 - x[2] * x[4], x[2] * x[4] - 6, x[3] - 2x[4] - 3],
                     [1.0 0 0 0; 0 0 -1 0; 0 0 1 0;
                      0 -x[4] 0 -x[2]; 0 x[4] 0 x[2]; 0 0 1 -2])
    @test f.G(x, p) == [x[1] - 2, x[2] * x[3] - 5]
    @test f.H(x, p) == [x[2] - 3, x[4] + 2x[1] - 7]
    @test f.jac_G(x, p) == [1.0 0 0 0; 0 x[3] x[2] 0]
    @test f.jac_H(x, p) == [0.0 1 0 0; 2 0 0 1]

    # Constraint callbacks stay unscaled when the shared penalty state changes.
    p.rho_eq .= 7
    p.rho_ineq .= 11
    p.rho_comp .= [13, 17]
    @test f.G(x, p) == [x[1] - 2, x[2] * x[3] - 5]
    @test f.jac_G(x, p) == [1.0 0 0 0; 0 x[3] x[2] 0]
end

@testset "AulaSubproblem fresh parameter state" begin
    model = PublicAPIMPCC([2.5,3.5,0.25,4.5])
    config = CIOpt.AulaConfig(rho_eq_init=4, rho_ineq_init=9, rho_comp_init=16,
                              print_level=0)
    sub = CIOpt.AulaSubproblem(model, config)
    p1, p2 = CIOpt.make_params(sub), CIOpt.make_params(sub)
    names = (:rho_eq, :kappa_eq, :rho_ineq, :kappa_ineq, :rho_comp,
              :kappaG, :kappaH, :sG, :sH)
    @test keys(p1) == names
    @test all(getproperty(p1, name) !== getproperty(p2, name) for name in names)
    @test all(getproperty(p1, names[i]) !== getproperty(p1, names[j])
              for i in eachindex(names) for j in (i+1):length(names))

    x = NLPModels.get_x0(model)
    initial_residual = CIOpt.eval_residual(sub, x, p2)
    initial_jacobian = CIOpt.eval_jacobian(sub, x, p2)
    p1.kappa_eq .= [2, -2]
    p1.sG[1] = 0.25
    residual = CIOpt.eval_residual(sub, x, p1)
    @test residual[1:2] ≈ initial_residual[1:2] + [1, -1]
    @test residual[9:10] ≈ initial_residual[9:10] + [-1, 0]
    @test CIOpt.eval_residual(sub, x, p2) == initial_residual
    @test p2.kappa_eq == zeros(2)
    @test p2.sG == zeros(2)

    expected_value = sub.functions.ℓ(x, p1) + dot(residual, residual) / 2
    @test CIOpt.eval_augmented_objective(sub, x, p1) ≈ expected_value
    @test CIOpt.eval_task_objective(sub, x, p1) == NLPModels.obj(model, x)
    @test CIOpt.eval_task_gradient(sub, x, p1) == -NLPModels.grad(model, x)
    gradient = sub.functions.∇ℓ(x, p1) + initial_jacobian' * residual
    @test CIOpt.eval_augmented_gradient_inf(sub, x, p1) ≈ norm(gradient, Inf)

    p1.rho_eq .= 16
    @test CIOpt.eval_jacobian(sub, x, p1)[1:2, :] ≈ 2initial_jacobian[1:2, :]
    @test CIOpt.eval_jacobian(sub, x, p2) == initial_jacobian

    fresh = CIOpt.make_params(sub)
    @test fresh.rho_eq == fill(4.0, 2)
    @test fresh.kappa_eq == zeros(2)
    @test fresh.sG == zeros(2)
end

@testset "AulaSubproblem Float32 parameters" begin
    model = PublicAPIMPCC(Float32[2.5, 3.5, 0.25, 4.5])
    config = CIOpt.AulaConfig(rho_eq_init=2.5, rho_ineq_init=3.5, rho_comp_init=5.5,
                              print_level=0)
    sub = CIOpt.AulaSubproblem(model, config)
    p = CIOpt.make_params(sub)
    @test all(state -> state isa Vector{Float32}, values(p))
    @test p.rho_eq == fill(2.5f0, 2)
    @test p.rho_ineq == fill(3.5f0, 6)
    @test p.rho_comp == fill(5.5f0, 2)
    for (name, dimension) in ((:kappa_eq, 2), (:kappa_ineq, 6), (:kappaG, 2),
                              (:kappaH, 2), (:sG, 2), (:sH, 2))
        @test getproperty(p, name) == zeros(Float32, dimension)
    end
    # The Float64 config values are valid, but unusable after conversion to T.
    overflow = CIOpt.AulaConfig(rho_eq_init=1e100)
    underflow = CIOpt.AulaConfig(rho_comp_init=1e-100)
    @test_throws ArgumentError CIOpt.make_params(CIOpt.AulaSubproblem(model, overflow))
    @test_throws ArgumentError CIOpt.make_params(CIOpt.AulaSubproblem(model, underflow))
end

@testset "AulaSubproblem compacted ordinary constraint rows" begin
    objective(x) = sum(abs2, x)
    constraints(x) = [x[1]^2 + x[2] + 3,
                      x[1] + 2x[2] - x[3],
                      x[2] * x[3] + 4,
                      x[3]^2 + x[4] + 2,
                      x[1] * x[4] + x[2]^2,
                      x[1] + x[3] * x[4] + 5]
    x = [1.2, -0.4, 1.6, 0.75]
    nlp = ADNLPModels.ADNLPModel(objective, copy(x),
                                 [-Inf, -2.0, 1.0, 0.5], [3.0, Inf, 4.0, 0.5],
                                 constraints, [2.0, 1.0, 3.0, 1.0, -1.0, 4.0],
                                 [Inf, 1.0, Inf, Inf, 5.0, Inf])
    # Source rows 2 and 5 become ordinary MPCC rows 1 and 2.
    model = MPCCModels.MPCCModelConCon(nlp, [1, 4], [3, 6])
    sub = CIOpt.AulaSubproblem(model)
    f, p = sub.functions, CIOpt.make_params(sub)
    c = constraints(x)

    @test NLPModels.get_ncon(model) == 2
    @test NLPModels.cons(model, x) ≈ c[[2, 5]]
    @test f.ℓ(x, p) ≈ objective(x)
    @test f.∇ℓ(x, p) ≈ 2x
    @test f.∇²ℓ(x, p) ≈ 2Matrix{Float64}(I, 4, 4)
    check_bound_rows(f.h(x, p), f.jac_h(x, p),
                     [x[4] - 0.5, c[2] - 1],
                     [0.0 0 0 1; 1 2 -1 0])
    check_bound_rows(f.g(x, p), f.jac_g(x, p),
                     [x[1] - 3, -2 - x[2], 1 - x[3], x[3] - 4,
                      -1 - c[5], c[5] - 5],
                     [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 1 0;
                      -x[4] -2x[2] 0 -x[1]; x[4] 2x[2] 0 x[1]])
    @test f.G(x, p) ≈ c[[1, 4]] - [2, 1]
    @test f.H(x, p) ≈ c[[3, 6]] - [3, 4]
    @test f.jac_G(x, p) ≈ [2x[1] 1 0 0; 0 0 2x[3] 1]
    @test f.jac_H(x, p) ≈ [0.0 x[3] x[2] 0; 1 0 x[4] x[3]]

    for (callback, jacobian) in ((f.h, f.jac_h), (f.g, f.jac_g),
                                  (f.G, f.jac_G), (f.H, f.jac_H))
        @test Matrix(jacobian(x, p)) ≈ finite_difference_jacobian(callback, x, p) atol=2e-7 rtol=2e-7
    end
end

end # module AulaSubproblemTests
