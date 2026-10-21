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

# The default allocating method accesses a private MPCC work buffer.
function MPCCModels.comp_res_right(model::PublicAPIMPCC{T},
                                   x::AbstractVector{T}) where {T}
    out = Vector{T}(undef, MPCCModels.get_ncc(model))
    return MPCCModels.comp_res_right!(model, x, out)
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

function finite_difference_jacobian(callback, x)
    step = 1e-6
    columns = map(eachindex(x)) do j
        plus, minus = copy(x), copy(x)
        plus[j] += step
        minus[j] -= step
        (callback(plus) - callback(minus)) / (2step)
    end
    return hcat(columns...)
end

@testset "AulaSubproblem public MPCC API" begin
    model = PublicAPIMPCC([2.5, 3.5, 0.25, 4.5])
    config = CIOpt.AulaConfig(rho_eq_init=2, rho_ineq_init=3, rho_comp_init=5,
                              print_level=0)
    sub = CIOpt.AulaSubproblem(model, config)
    x = copy(NLPModels.get_x0(model))
    p = CIOpt.make_params(sub)

    @test CIOpt.get_nvar(sub) == 4
    @test p isa NamedTuple
    @test !hasproperty(p,:data)
    @test p.rho_eq == fill(2.0, 2)
    @test p.rho_ineq == fill(3.0, 6)
    @test p.rho_comp == fill(5.0, 2)
    for (name, dimension) in ((:kappa_eq, 2), (:kappa_ineq, 6), (:kappaG, 2),
                              (:kappaH, 2), (:sG, 2), (:sH, 2))
        @test getproperty(p, name) == zeros(dimension)
    end

    # Maximization negates the objective and its derivatives for minimization.
    @test CIOpt.eval_obj(sub, x) == -NLPModels.obj(model, x)
    @test CIOpt.eval_obj_grad(sub, x) == -NLPModels.grad(model, x)
    @test CIOpt.eval_obj_hess(sub, x) == -NLPModels.hess(model, x)

    J = CIOpt.eval_al_residual_jacobian(sub, x, p)
    check_bound_rows(CIOpt.eval_eq(sub, x), J[1:2, :] / sqrt(2.0),
                     [x[4] - 4, x[1] + x[3] - 1],
                     [0.0 0 0 1; 1 0 1 0])
    # The lower bounds of complementarity variables x1/x2 are represented by
    # G/H. Their finite upper bounds, other bounds, and fixed x4 are retained.
    inequalities = [x[1] - 8, -1 - x[3], x[3] - 5,
                    2 - x[2] * x[4], x[2] * x[4] - 6, x[3] - 2x[4] - 3]
    inequality_jacobian = [1.0 0 0 0; 0 0 -1 0; 0 0 1 0;
                           0 -x[4] 0 -x[2]; 0 x[4] 0 x[2]; 0 0 1 -2]
    masked_jacobian = Diagonal(inequalities .> 0) * inequality_jacobian
    check_bound_rows(CIOpt.eval_ineq(sub, x), J[3:8, :] / sqrt(3.0),
                     inequalities, masked_jacobian)
    G = [x[1] - 2, x[2] * x[3] - 5]
    H = [x[2] - 3, x[4] + 2x[1] - 7]
    @test CIOpt.eval_GH(sub, x) == (G, H)
    @test J[9:10, :] / sqrt(5.0) ≈ [1.0 0 0 0; 0 x[3] x[2] 0]
    @test J[11:12, :] / sqrt(5.0) ≈ [0.0 1 0 0; 2 0 0 1]
    numerical = finite_difference_jacobian(y -> CIOpt.eval_al_residual(sub, y, p), x)
    @test Matrix(J) ≈ numerical atol=2e-7 rtol=2e-7
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
    initial_residual = CIOpt.eval_al_residual(sub, x, p2)
    initial_jacobian = CIOpt.eval_al_residual_jacobian(sub, x, p2)
    p1.kappa_eq .= [2, -2]
    p1.sG[1] = 0.25
    residual = CIOpt.eval_al_residual(sub, x, p1)
    @test residual[1:2] ≈ initial_residual[1:2] + [1, -1]
    @test residual[9:10] ≈ initial_residual[9:10] + [-1, 0]
    @test CIOpt.eval_al_residual(sub, x, p2) == initial_residual
    @test p2.kappa_eq == zeros(2)
    @test p2.sG == zeros(2)

    expected_value = CIOpt.eval_obj(sub, x) + dot(residual, residual) / 2
    @test CIOpt.eval_al_objective(sub, x, p1) ≈ expected_value
    gradient = CIOpt.eval_obj_grad(sub, x) + initial_jacobian' * residual
    @test CIOpt.eval_al_objective_grad_inf(sub, x, p1) ≈ norm(gradient, Inf)

    p1.rho_eq .= 16
    @test CIOpt.eval_al_residual_jacobian(sub, x, p1)[1:2, :] ≈ 2initial_jacobian[1:2, :]
    @test CIOpt.eval_al_residual_jacobian(sub, x, p2) == initial_jacobian

    # A positive multiplier can activate a satisfied inequality. Its shifted
    # residual, rather than the sign of the raw constraint, selects the row.
    inequalities = CIOpt.eval_ineq(sub, x)
    inactive = findfirst(value -> value < 0, inequalities)
    p1.kappa_ineq[inactive] = p1.rho_ineq[inactive] * (1 - inequalities[inactive])
    row = length(p1.kappa_eq) + inactive
    @test initial_residual[row] == 0
    @test CIOpt.eval_al_residual(sub, x, p1)[row] ≈ 3
    @test all(iszero, initial_jacobian[row, :])
    active_jacobian = CIOpt.eval_al_residual_jacobian(sub, x, p1)
    @test norm(active_jacobian[row, :]) > 0
    numerical = finite_difference_jacobian(y -> CIOpt.eval_al_residual(sub, y, p1), x)
    @test Matrix(active_jacobian) ≈ numerical atol=2e-7 rtol=2e-7
    @test CIOpt.eval_al_residual(sub, x, p2) == initial_residual

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
    p = CIOpt.make_params(sub)
    c = constraints(x)

    @test NLPModels.get_ncon(model) == 2
    @test NLPModels.cons(model, x) ≈ c[[2, 5]]
    @test CIOpt.eval_obj(sub, x) ≈ objective(x)
    @test CIOpt.eval_obj_grad(sub, x) ≈ 2x
    @test CIOpt.eval_obj_hess(sub, x) ≈ 2Matrix{Float64}(I, 4, 4)
    # Activate all inequality residual rows to inspect both signs of every
    # bound Jacobian, including bounds that are satisfied at this sample point.
    p.kappa_ineq .= p.rho_ineq .* (1 .+ abs.(CIOpt.eval_ineq(sub, x)))
    J = CIOpt.eval_al_residual_jacobian(sub, x, p)
    check_bound_rows(CIOpt.eval_eq(sub, x), J[1:2, :],
                     [x[4] - 0.5, c[2] - 1],
                     [0.0 0 0 1; 1 2 -1 0])
    check_bound_rows(CIOpt.eval_ineq(sub, x), J[3:8, :],
                     [x[1] - 3, -2 - x[2], 1 - x[3], x[3] - 4,
                      -1 - c[5], c[5] - 5],
                     [1.0 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 1 0;
                      -x[4] -2x[2] 0 -x[1]; x[4] 2x[2] 0 x[1]])
    G, H = CIOpt.eval_GH(sub, x)
    @test G ≈ c[[1, 4]] - [2, 1]
    @test H ≈ c[[3, 6]] - [3, 4]
    @test J[9:10, :] ≈ [2x[1] 1 0 0; 0 0 2x[3] 1]
    @test J[11:12, :] ≈ [0.0 x[3] x[2] 0; 1 0 x[4] x[3]]
    numerical = finite_difference_jacobian(y -> CIOpt.eval_al_residual(sub, y, p), x)
    @test Matrix(J) ≈ numerical atol=2e-7 rtol=2e-7
end

end # module AulaSubproblemTests
