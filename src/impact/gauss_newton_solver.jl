"""
State for damped Gauss-Newton/Newton X updates. `iterations` counts accepted
primal steps and `factor_failures` counts failed sparse factorizations across
the most recent `minimize!` call. The timing fields accumulate over the current
outer solve; `solve!` resets them at its start. A transient factorization failure
only marks `x_update_failed` if that X solve accepted no step.
"""
mutable struct GaussNewtonSolver
    iterations::Int
    factor_failures::Int
    x_update_failed::Bool
    eval_seconds::Float64
    factor_seconds::Float64
end
GaussNewtonSolver() = GaussNewtonSolver(0, 0, false, 0.0, 0.0)

"""
X-update controls for the damped primal solve. `lambda_init` is restored for
each `minimize!` call; failed or rejected steps raise it, while accepted steps
lower it according to the gain ratio. The saddle backend avoids explicitly
forming the penalty normal-equation term; `sigma_primal` adds a small proximal
diagonal to its primal block.
"""
Base.@kwdef struct GaussNewtonConfig
    max_iter::Int = 50
    grad_tol::Float64 = 1e-6
    step_tol::Float64 = 1e-8
    lambda_init::Float64 = 1e-5
    lambda_min::Float64 = 1e-12
    lambda_max::Float64 = 1e12
    max_damping_tries::Int = 20
    use_saddle::Bool = true
    sigma_primal::Float64 = 1e-8
    equilibrate_dual::Bool = true
    refinement_steps::Int = 0
end

GaussNewtonConfig(c::AulaConfig) = GaussNewtonConfig(
    max_iter=c.newton_max_iter, grad_tol=c.newton_tol, step_tol=c.newton_step_tol,
    lambda_init=c.newton_regularization, lambda_min=c.newton_lambda_min,
    lambda_max=c.newton_lambda_max, max_damping_tries=c.newton_max_damping_tries,
    use_saddle=c.use_saddle, sigma_primal=c.saddle_sigma_primal,
    equilibrate_dual=c.saddle_equilibrate_dual, refinement_steps=c.saddle_refinement_steps)

function _saddle_step(Hlag, C, rhs_penalty, gradient, λ, config::GaussNewtonConfig)
    n, nd = length(gradient), size(C, 1)
    gbase = gradient - C' * rhs_penalty
    σ = config.sigma_primal
    sdual = ones(eltype(gradient), nd)
    if config.equilibrate_dual
        for i in 1:nd
            rownorm2 = sum(abs2, C[i, :])
            sdual[i] = max(1.0, inv(sqrt(rownorm2 + 1.0)))
        end
    end
    S = spdiagm(0 => sdual)
    # Solve the primal-dual system for the same damped Gauss-Newton step as
    # (Hlag + C'*C + lambda*I)\dx = -gradient. Since gradient already includes
    # C'*rhs, define gbase = gradient - C'*rhs and introduce a dual variable:
    #
    #   [Hlag+(lambda+sigma)I   C'*S ] [dx] = [-gbase]
    #   [S*C                   -S^2 ] [ y]   [-S*rhs ]
    #
    # Eliminating y recovers the penalty normal equations (with the small sigma
    # proximal term). The diagonal congruence scale S improves row conditioning.
    M = [sparse(Hlag + (λ + σ) * I) C' * S;
         S * C -spdiagm(0 => sdual .^ 2)]
    b = vcat(-gbase, -sdual .* rhs_penalty)
    F = try
        lu(sparse(M); check=false)
    catch
        return nothing, 0.0
    end
    issuccess(F) || return nothing, 0.0
    t0 = time_ns()
    sol = try
        F \ b
    catch
        return nothing, (time_ns() - t0) / 1e9
    end
    elapsed = (time_ns() - t0) / 1e9
    all(isfinite, sol) || return nothing, elapsed
    for _ in 1:config.refinement_steps
        r = b - M * sol
        norm(r) <= 1e-10 * max(1.0, norm(b)) && break
        δ = try
            F \ r
        catch
            return nothing, elapsed
        end
        all(isfinite, δ) || return nothing, elapsed
        sol .+= δ
    end
    return sol[1:n], elapsed
end

function _normal_step(Hlag, C, gradient, λ)
    # C already includes each active residual's sqrt(2*rho)*scale factor, so
    # C'*C is the Gauss-Newton penalty curvature and `gradient` is the full
    # augmented gradient.
    H = sparse(Hlag + C' * C + λ * I)
    F = try
        cholesky(Symmetric(H); check=false)
    catch
        return nothing
    end
    issuccess(F) || return nothing
    dx = try
        -(F \ gradient)
    catch
        return nothing
    end
    return all(isfinite, dx) ? dx : nothing
end

"""
    minimize!(solver, sub, x, config; grad_tol=-1)

Minimize the current augmented objective in-place with damped sparse X steps.
At each iteration, solve the saddle system or normal equations, evaluate the
trial point, and accept it only when the objective decreases and the gain ratio
is positive. Rejections increase the LM damping geometrically; accepted steps
reduce it. Damping always restarts at `lambda_init` on the next call.

`grad_tol` can override the configured gradient tolerance for one call, which
lets the AuLa outer loop use an inexact forcing sequence. If a factorization
fails, backtracking steepest descent is tried after the LM retries; a failed
factorization is considered terminal only when no step was accepted.
"""
function minimize!(solver::GaussNewtonSolver, sub::AulaSubproblem, x::Vector,
                   config::AulaConfig; grad_tol::Real=-1)
    gn = GaussNewtonConfig(config)
    gtol = grad_tol > 0 ? grad_tol : gn.grad_tol
    solver.iterations = 0
    solver.factor_failures = 0
    solver.x_update_failed = false
    λ, ν = gn.lambda_init, 2.0
    any_accept = false
    any_factor_failure = false

    for _ in 1:gn.max_iter
        t0 = time_ns()
        φ, g, Hlag, C, b, _, _, _, _ = evaluate_augmented(sub, x)
        solver.eval_seconds += (time_ns() - t0) / 1e9
        ginf = isempty(g) ? 0.0 : norm(g, Inf)
        ginf < gtol && break
        # This is the Hessian of the local quadratic model. The full model
        # objective used below also includes the first-order term `g`.
        model_hess = sparse(Hlag + C' * C)
        accepted = false
        la_failed_this_iter = false
        step_norm = 0.0

        for _ in 1:gn.max_damping_tries
            t_factor = time_ns()
            if gn.use_saddle && size(C, 1) > 0
                dx, _ = _saddle_step(Hlag, C, b, g, λ, gn)
                solver.factor_seconds += (time_ns() - t_factor) / 1e9
            else
                dx = _normal_step(Hlag, C, g, λ)
                solver.factor_seconds += (time_ns() - t_factor) / 1e9
            end
            if dx === nothing
                any_factor_failure = true
                la_failed_this_iter = true
                solver.factor_failures += 1
                λ = min(λ * ν, gn.lambda_max)
                ν *= 2
                continue
            end

            xtrial = x + dx
            t_eval = time_ns()
            φtrial = evaluate_augmented(sub, xtrial)[1]
            solver.eval_seconds += (time_ns() - t_eval) / 1e9
            predicted = -dot(g, dx) - dot(dx, model_hess * dx) / 2
            # Gain ratio compares actual reduction to the local quadratic
            # prediction. It controls whether LM damping grows or shrinks.
            gain = predicted > 0 ? (φ - φtrial) / predicted : -1.0
            if isfinite(φtrial) && φtrial < φ && gain > 0
                x .= xtrial
                step_norm = norm(dx)
                λ = max(λ * max(1 / 3, 1 - (2 * gain - 1)^3), gn.lambda_min)
                ν = 2.0
                accepted = true
                break
            end
            λ = min(λ * ν, gn.lambda_max)
            ν *= 2
        end

        if !accepted && la_failed_this_iter && ginf >= gtol
            gg = dot(g, g)
            curv = 2 * sum(abs2, C * g)
            α = curv > 0 ? gg / curv : 1 / (1 + ginf)
            (!isfinite(α) || α <= 0) && (α = 1 / (1 + ginf))
            for _ in 1:gn.max_damping_tries
                xtrial = x - α * g
                t_eval = time_ns()
                φtrial = evaluate_augmented(sub, xtrial)[1]
                solver.eval_seconds += (time_ns() - t_eval) / 1e9
                if isfinite(φtrial) && φtrial < φ - 1e-4 * α * gg
                    x .= xtrial
                    step_norm = α * sqrt(gg)
                    accepted = true
                    break
                end
                α /= 2
            end
        end
        accepted || break
        any_accept = true
        solver.iterations += 1
        step_norm < gn.step_tol && break
    end
    solver.x_update_failed = any_factor_failure && !any_accept
    return first(evaluate_augmented(sub, x))
end
