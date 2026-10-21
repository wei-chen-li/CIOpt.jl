"""Configuration for the adaptive Levenberg-Marquardt least-squares solver.

Damping is adjusted automatically by a gain-ratio (trust-region) rule: a step
that reduces the objective is accepted and the damping is lowered; a step that
does not is rejected and the damping is raised. The damping always starts from
`lambda_init` at the beginning of each minimize() call, so a large value from a
previous (harder) solve never carries over into the next one.
"""
Base.@kwdef struct GaussNewtonConfig
    max_iter::Int = 50                    # Maximum outer Gauss-Newton iterations
    grad_tol::Float64 = 1e-6              # Stop when ||∇ℓ + Jᵀr||_inf below this
    step_tol::Float64 = 1e-8              # Stop when ||step||_2 below this
    lambda_init::Float64 = 1e-5           # Initial LM damping, reset on every minimize()
    lambda_min::Float64 = 1e-12           # Lower clamp for damping
    lambda_max::Float64 = 1e12            # Upper clamp for damping
    max_damping_tries::Int = 20           # Reject/raise-damping attempts per iteration
    use_saddle::Bool = true
    sigma_primal::Float64 = 1e-8          # Proximal floor added to the (1,1) block
    saddle_equilibrate_dual::Bool = true  # Congruence scaling; preserves dz and sparsity
    saddle_refinement_steps::Int = 0      # Optional residual-correction solves
    print_level::Int = 0                  # 0 silent, >=1 per-iteration trace
end

GaussNewtonConfig(c::AulaConfig) = GaussNewtonConfig(
    max_iter=c.newton_max_iter, grad_tol=c.newton_tol, step_tol=c.newton_step_tol,
    lambda_init=c.newton_regularization, lambda_min=c.newton_lambda_min,
    lambda_max=c.newton_lambda_max, max_damping_tries=c.newton_max_damping_tries,
    use_saddle=c.use_saddle, sigma_primal=c.saddle_sigma_primal,
    saddle_equilibrate_dual=c.saddle_equilibrate_dual,
    saddle_refinement_steps=c.saddle_refinement_steps,
    print_level=c.print_level >= 3 ? 1 : 0)


"""Standalone damped Gauss-Newton solver for nonlinear least squares.

Minimises  f(x) = ℓ(x; p) + ½‖r(x; p)‖²  for a fixed parameter vector p,
where ℓ, its gradient ∇ℓ and hessian ∇²ℓ; r and its Jacobian J := dr/dx are
supplied as functions with arguments (x, p).

The Levenberg-Marquardt step solves the damped normal equations

    (∇²ℓ + JᵀJ + λI) Δx = -∇ℓ - Jᵀr

or the damped saddle equations

    ⌈ ∇²ℓ + (λ+σ)I   Jᵀ ⌉ ⌈ Δx ⌉ = ⌈ -∇ℓ ⌉
    ⌊ J              -I ⌋ ⌊  y ⌋   ⌊  -r ⌋

via sparse Cholesky (normal) or pivoted LU (saddle), and a gain-ratio rule accepts or rejects the
step while adapting λ. Symbolic factorisations are reused across
calls while the assembled matrix sparsity pattern is unchanged.

This module knows nothing about the MPCC/AuLa structure: callers bake all
multipliers, penalties and slack variables into the parameter vector p.
"""
mutable struct GaussNewtonSolver
    functions::NamedTuple
    config::GaussNewtonConfig
    factor::Any
    pattern::Union{Nothing,Tuple{Vector{Int},Vector{Int}}}
    last_iterations::Int
    last_x_update_failed::Bool
    eval_seconds::Float64
    factor_seconds::Float64
end

GaussNewtonSolver() = GaussNewtonSolver((;), GaussNewtonConfig(), nothing, nothing,
                                        0, false, 0.0, 0.0)

"""
    init(solver::GaussNewtonSolver, ℓ, ∇ℓ, ∇²ℓ, residual, jacobian,
         config=GaussNewtonConfig())

Bind or replace callbacks `(x, p) -> value` and configure the solver. Return
`solver`. Reuse the factorisation cache when its matrix pattern is unchanged;
changing the linear-algebra backend clears the cache. Call before `solve!`.
"""
function init(solver::GaussNewtonSolver, ℓ, ∇ℓ, ∇²ℓ, residual, jacobian,
              config::GaussNewtonConfig=GaussNewtonConfig())
    config.max_iter > 0 && config.max_damping_tries > 0 ||
        throw(ArgumentError("iteration and damping retry limits must be positive"))
    for name in (:grad_tol, :step_tol, :lambda_init, :lambda_min, :lambda_max,
                 :sigma_primal)
        v = getfield(config, name)
        isfinite(v) && v > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    config.lambda_min <= config.lambda_max ||
        throw(ArgumentError("lambda_min must not exceed lambda_max"))
    config.saddle_refinement_steps >= 0 ||
        throw(ArgumentError("saddle_refinement_steps must be nonnegative"))
    if solver.config.use_saddle != config.use_saddle
        solver.factor = nothing
        solver.pattern = nothing
    end
    solver.functions = (; ℓ, ∇ℓ, ∇²ℓ, residual, jacobian)
    solver.config = config
    solver.last_iterations = 0
    solver.last_x_update_failed = false
    solver.eval_seconds = 0.0
    solver.factor_seconds = 0.0
    return solver
end

"""
    solve!(solver::GaussNewtonSolver, params, x; grad_tol=-1)

Minimise `ℓ(x; p) + ½‖r(x; p)‖²`, modifying `x` in place, and return
the final objective. A positive `grad_tol` overrides the configured tolerance.
Damping resets on every call. `last_iterations` counts accepted steps;
`last_x_update_failed` is true only when a linear solve failed and no step was
accepted. Evaluation and factorisation timers accumulate across calls.
"""
function solve!(solver::GaussNewtonSolver, params, x::AbstractVector{T};
                grad_tol::Real=-1) where {T<:AbstractFloat}
    isempty(solver.functions) && throw(ArgumentError("call init before solve!"))
    c = solver.config
    gtol = grad_tol > 0 ? grad_tol : c.grad_tol

    function eval_values(x, p)
        start = time_ns()
        try
            f = solver.functions
            r = copy(f.residual(x, p))
            return f.ℓ(x, p) + dot(r, r) / 2, r
        finally
            solver.eval_seconds += (time_ns() - start) * 1e-9
        end
    end

    function eval_derivatives(x, p)
        start = time_ns()
        try
            f = solver.functions
            gℓ = copy(f.∇ℓ(x, p))
            Hℓ = sparse(f.∇²ℓ(x, p))
            J = sparse(f.jacobian(x, p))
            return gℓ, sparse(Symmetric(Hℓ, :L)), J
        finally
            solver.eval_seconds += (time_ns() - start) * 1e-9
        end
    end

    f_val, r = eval_values(x, params)
    gℓ, Hℓ, J = eval_derivatives(x, params)
    n, m = length(x), length(r)
    length(gℓ) == n && size(Hℓ) == (n, n) && size(J) == (m, n) ||
        throw(DimensionMismatch("objective derivatives and residual Jacobian must match x and r"))

    λ = clamp(c.lambda_init, c.lambda_min, c.lambda_max)
    ν = 2.0  # Geometric raise factor on rejection (Nielsen).

    # X-update failure tracking for reporting; it does not change control flow.
    solver.last_iterations = 0
    solver.last_x_update_failed = false
    had_la_failure = false  # some LM attempt's factorisation reported info != Success
    any_accept = false      # at least one LM step was accepted this call

    for iter in 0:(c.max_iter - 1)
        grad = gℓ + J' * r
        grad_inf = norm(grad, Inf)
        if c.print_level >= 1
            println("    GN $iter: f=$f_val, ||grad||_inf=$grad_inf, lambda=$λ",
                    c.use_saddle ? " [saddle]" : "")
        end
        grad_inf < gtol && break

        accepted = false
        step_norm = zero(T)
        la_failed_this_iter = false

        # The undamped model and RHS stay fixed across damping retries.
        if c.use_saddle
            # Congruence scaling diag(I, S) preserves the primal step.
            scales = c.saddle_equilibrate_dual ?
                inv.(sqrt.(vec(sum(abs2, J; dims=2)) .+ one(T))) :
                ones(T, m)
            S = spdiagm(0 => scales)
            C = S * J
            base = [Hℓ C'; C -spdiagm(0 => scales .^ 2)]
            rhs = vcat(-gℓ, -scales .* r)
        else
            base = Hℓ + J' * J
            rhs = -grad
        end

        diag_add = c.use_saddle ?
            vcat(fill(λ + c.sigma_primal, n), zeros(T, m)) : fill(λ, n)
        A = base + spdiagm(0 => diag_add)
        λ_in_A = λ

        for _ in 1:c.max_damping_tries
            if λ != λ_in_A
                for k in 1:n
                    A[k, k] += λ - λ_in_A
                end
                λ_in_A = λ
            end

            sol = _solve_linear_system!(solver, A, rhs)
            if sol === nothing
                had_la_failure = true
                la_failed_this_iter = true
            else
                dx = sol[1:n]
                x_new = x + dx
                f_new, r_trial = eval_values(x_new, params)
                Jdx = J * dx
                predicted = -dot(grad, dx) -
                            (dot(dx, Hℓ * dx) + dot(Jdx, Jdx)) / 2
                gain = predicted > 0 ? (f_val - f_new) / predicted : -1.0

                if isfinite(f_new) && f_new < f_val && gain > 0
                    x .= x_new
                    step_norm = norm(dx)
                    r = r_trial
                    f_val = f_new
                    gℓ, Hℓ, J = eval_derivatives(x, params)
                    shrink = max(1 / 3, 1 - (2 * gain - 1)^3)
                    λ = max(λ * shrink, c.lambda_min)
                    ν = 2.0
                    accepted = true
                    break
                end
            end
            λ = min(λ * ν, c.lambda_max)
            ν *= 2
        end

        # Steepest-descent fallback for a failed LM factorisation. If every LM
        # retry fails or is rejected, take a backtracking step along -grad so the
        # X-update can still make progress before trying LM again.
        if !accepted && la_failed_this_iter && grad_inf >= gtol
            gg = dot(grad, grad)
            Jg = J * grad
            curv = dot(grad, Hℓ * grad) + dot(Jg, Jg)
            α = curv > 0 ? gg / curv : 1 / (1 + grad_inf)
            if !isfinite(α) || α <= 0
                α = 1 / (1 + grad_inf)
            end
            for _ in 1:c.max_damping_tries
                x_new = x - α * grad
                f_new, r_trial = eval_values(x_new, params)
                if isfinite(f_new) && f_new < f_val - 1e-4 * α * gg
                    x .= x_new
                    step_norm = α * sqrt(gg)
                    r = r_trial
                    f_val = f_new
                    gℓ, Hℓ, J = eval_derivatives(x, params)
                    accepted = true
                    break
                end
                α *= 0.5
            end
        end

        accepted && (solver.last_iterations += 1)
        any_accept |= accepted
        accepted || break  # No descent even along -grad (e.g. non-finite Jacobian).
        step_norm < c.step_tol && break
    end

    # Treat the X-update as stuck only when a factorisation failed and no step was
    # accepted. Transient failures recovered by damping are not terminal.
    solver.last_x_update_failed = had_la_failure && !any_accept
    return f_val
end

function _solve_linear_system!(solver, A, rhs)
    all(isfinite, nonzeros(A)) && all(isfinite, rhs) || return nothing
    start = time_ns()
    try
        same_pattern = solver.pattern !== nothing &&
            solver.pattern[1] == A.colptr && solver.pattern[2] == A.rowval
        if solver.config.use_saddle
            solver.factor = same_pattern && solver.factor !== nothing ?
                lu!(solver.factor, A; check=true) : lu(A; check=true)
        else
            solver.factor = same_pattern && solver.factor !== nothing ?
                cholesky!(solver.factor, Symmetric(A); check=true) :
                cholesky(Symmetric(A); check=true)
        end
        solver.pattern = (copy(A.colptr), copy(A.rowval))
        sol = solver.factor \ rhs
        if solver.config.use_saddle
            for _ in 1:solver.config.saddle_refinement_steps
                residual = rhs - A * sol
                norm(residual) <= 1e-10 * max(1, norm(rhs)) && break
                correction = solver.factor \ residual
                all(isfinite, correction) || return nothing
                sol .+= correction
            end
        end
        return all(isfinite, sol) ? sol : nothing
    catch err
        if err isa PosDefException || err isa SingularException ||
           err isa ZeroPivotException || err isa LinearAlgebra.LAPACKException
            solver.factor = nothing
            solver.pattern = nothing
            return nothing
        end
        rethrow()
    finally
        solver.factor_seconds += (time_ns() - start) * 1e-9
    end
end
