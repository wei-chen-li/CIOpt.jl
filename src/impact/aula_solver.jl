"""
Violation record for one final constraint channel. Standard constraint
violations are reported unscaled; for complementarity, `violation` is the raw
`||G .* H||_∞` product and `residual_violation` measures distance from `G,H` to
the projected complementary slacks.
"""
struct ConstraintViolation{T<:AbstractFloat}
    name::String
    violation::T
    residual_violation::T
end

"""
Result of an AuLa solve. `x` and `objective_value` are the final primal point and
original model objective. Feasibility fields use raw, unscaled units. The four
`comp_*` fields report nonnegativity and multiplier-support certificates for
the complementarity pairs. `total_gn_iterations` counts accepted X steps.
`eval_time` and `factor_time` count work inside X updates; `solve_time` also
includes the outer loop and its bookkeeping. `stationarity_violation` is zero
when stationarity checking is disabled. `converged` is true exactly when
`status == :converged`.
"""
struct AulaResult{T<:AbstractFloat}
    x::Vector{T}
    objective_value::T
    converged::Bool
    status::Symbol
    status_message::String
    outer_iterations::Int
    total_inner_iterations::Int
    total_gn_iterations::Int
    equality_violation::T
    inequality_violation::T
    complementarity_violation::T
    stationarity_violation::T
    comp_neg_G::T
    comp_neg_H::T
    comp_support_G::T
    comp_support_H::T
    constraint_violations::Vector{ConstraintViolation{T}}
    solve_time::Float64
    eval_time::Float64
    factor_time::Float64
end

"""
    AulaSolver()

AuLa driver with a reusable damped Gauss-Newton X-step state. One solver object
can be reused for successive `solve!` calls; each outer solve resets its timers
and each X solve starts with the configured LM damping.
"""
mutable struct AulaSolver
    gn::GaussNewtonSolver
end
AulaSolver() = AulaSolver(GaussNewtonSolver())

_block_raw(sub, x, b) = eval_block_source(sub, x, b)
_inf(v) = isempty(v) ? 0.0 : norm(v, Inf)
_sqscale(s) = s > 0 ? s^2 : 1.0

function _balanced_rho_eff(d2, f_abs, config)
    # Choose an effective penalty so rho_eff*||c||^2 is comparable to the
    # objective magnitude. d2 is expressed in unscaled residual units.
    clamp(2 * max(1, f_abs) / d2, config.auto_rho_clip_min, config.auto_rho_clip_max)
end

function _comp_distance2(comp, G, H)
    # Squared distance from each (G_i,H_i) to the union of the two
    # complementarity branches. This gives auto-rho initialization without
    # needing rho first, so it avoids a rho -> projection -> rho cycle.
    d2 = 0.0
    for i in 1:comp.dim
        nG, nH = min(G[i], 0), min(H[i], 0)
        d2 += min(nG^2 + H[i]^2, G[i]^2 + nH^2)
    end
    return d2
end

function _safe_penalty(rho, vnew, vold, tol, config, scale)
    # Increase rho only when feasibility is still outside tolerance and the
    # violation did not improve by the configured safeguard ratio. Auto-rho
    # mode caps effective stiffness rho*scale^2, not nominal rho alone. The
    # per-block cap lets a small scale reach the effective ceiling; max(rho_max,
    # ...) preserves the configured nominal headroom for larger scales.
    if vnew >= tol && vnew > config.penalty_decrease_ratio * vold
        cap = config.auto_rho_init ? max(config.rho_max, config.auto_rho_clip_max / _sqscale(scale)) : config.rho_max
        return min(rho * config.rho_scale, cap)
    end
    return rho
end

function _current_comp(sub, x)
    isempty(sub.comp_blocks) && return Vector{Float64}(), Vector{Float64}()
    return MPCCModels.comp_res_left(sub.model, x), MPCCModels.comp_res_right(sub.model, x)
end

function _reset_penalties!(sub, config)
    # A stagnation restart discards stale multipliers, slacks, and grown
    # penalties, returning every channel to its build-time seed.
    for b in sub.dual_blocks
        fill!(b.kappa, 0)
        b.rho = b.rho_init
    end
    for c in sub.comp_blocks
        fill!(c.kappaG, 0); fill!(c.kappaH, 0)
        fill!(c.sG, 0); fill!(c.sH, 0)
        c.rho = c.rho_init
    end
    sync_params!(sub)
end

"""
    solve!(solver, subproblem, config, x_init)

Run impact's safeguarded AuLa method on a model-backed subproblem. Each outer
iteration clips multipliers, approximately minimizes the augmented objective,
ascends standard and split complementarity multipliers, then increases only
those penalties whose violations have stalled. The inner BCD loop alternates a
damped primal X update with the exact closed-form complementarity slack update.

Convergence requires every standard block to meet its unscaled feasibility
tolerance. Each complementarity block must meet both its product test and its
split-slack residual test. If requested, stationarity of the augmented
Lagrangian is checked after feasibility is reached.
"""
function solve!(solver::AulaSolver, sub::AulaSubproblem, config::AulaConfig,
                x_init::AbstractVector)
    validate(config)
    T = eltype(sub.parameter_buffer)
    check_stationarity = sub.stationarity_enabled
    conditioned_comp = sub.conditioned_complementarity
    stationarity_tol = sub.stationarity_tol
    x = T.(x_init)
    length(x) == sub.nvar && all(isfinite, x) || throw(ArgumentError("invalid initial point"))
    total_inner = 0
    total_gn = 0
    solver.gn.eval_seconds = 0.0
    solver.gn.factor_seconds = 0.0
    converged = false
    la_failure = false
    ρbalanced = fill(!config.auto_rho_init, length(sub.dual_blocks))
    ρcompbalanced = fill(!config.auto_rho_init, length(sub.comp_blocks))
    # Previous violations start at +Inf. The safeguard therefore leaves rho
    # unchanged on the first outer pass, before it can compare progress.
    vprev = fill(Inf, length(sub.dual_blocks))
    vcprev = fill(Inf, length(sub.comp_blocks))
    outer_viol_prev = 1e3
    stagnant_outer = 0
    restart_count = 0
    nouter = 0

    if config.auto_rho_init
        # Balance initially infeasible blocks against the objective in effective
        # units. Blocks starting close to feasible keep their configured seed;
        # their first informative residual is handled by the deferred pass below.
        f0 = abs(NLPModels.obj(sub.model, x))
        for (i, b) in enumerate(sub.dual_blocks)
            c = b.scale .* _block_raw(sub, x, b)
            b.kind === :inequality && (c = max.(c, 0))
            d2 = sum(abs2, c) / _sqscale(b.scale)
            ρbalanced[i] = d2 > 1
            b.rho = ρbalanced[i] ? _balanced_rho_eff(d2, f0, config) / _sqscale(b.scale) : b.rho_init
        end
        if !isempty(sub.comp_blocks)
            G, H = _current_comp(sub, x)
            for (i, c) in enumerate(sub.comp_blocks)
                d2 = _comp_distance2(c, G, H)
                ρcompbalanced[i] = d2 > 1
                c.rho = ρcompbalanced[i] ? _balanced_rho_eff(d2, f0, config) / _sqscale(c.scale) : c.rho_init
            end
        end
    end

    previous_point = copy(x)
    start_time = time_ns()
    for outer in 1:config.max_outer_iters
        nouter = outer
        previous_point .= x
        # Bound multipliers before each inexact inner solve: equality and split
        # complementarity multipliers may have either sign, while inequality
        # multipliers must remain nonnegative.
        for b in sub.dual_blocks
            if b.kind === :inequality
                b.kappa .= clamp.(b.kappa, 0, config.safeguard_factor)
            else
                b.kappa .= clamp.(b.kappa, -config.safeguard_factor, config.safeguard_factor)
            end
        end
        for c in sub.comp_blocks
            c.kappaG .= clamp.(c.kappaG, -config.safeguard_factor, config.safeguard_factor)
            c.kappaH .= clamp.(c.kappaH, -config.safeguard_factor, config.safeguard_factor)
        end
        sync_params!(sub)

        outer_index = outer - 1
        # A loose early stopping threshold avoids oversolving poor subproblems;
        # later outer iterations use the tighter threshold needed for progress.
        inner_tol = outer_index < config.inner_tol_ramp_start ? config.inner_tol_init :
                    outer_index < config.inner_tol_ramp_end ? (config.inner_tol_init + config.inner_tol_final) / 2 :
                    config.inner_tol_final
        # Compare complete BCD sweeps (X step plus slack projection), not just
        # the change from the X step, when checking inner stagnation.
        ϕprev = evaluate_augmented(sub, x)[1]
        for _ in 1:config.max_inner_iters
            total_inner += 1
            # Inexact Newton forcing: relax the gradient target in proportion to
            # the previous outer violation, but never above the configured cap.
            gn_tol = config.use_forcing_sequence ?
                min(config.forcing_cap, max(config.newton_tol, config.forcing_factor * outer_viol_prev)) : -1.0
            minimize!(solver.gn, sub, x, config; grad_tol=gn_tol)
            la_failure |= solver.gn.x_update_failed
            total_gn += solver.gn.iterations
            if !isempty(sub.comp_blocks)
                G, H = _current_comp(sub, x)
                for c in sub.comp_blocks
                    project_complementarity!(c, G, H)
                end
                sync_params!(sub)
            end
            ϕ = evaluate_augmented(sub, x)[1]
            abs(ϕprev - ϕ) < inner_tol && break
            ϕprev = ϕ
        end

        G, H = _current_comp(sub, x)
        scaled_violations = zeros(T, length(sub.dual_blocks))
        unscaled_violations = zeros(T, length(sub.dual_blocks))
        # Dual ascent uses scaled residuals. Inequality multipliers are projected
        # back onto the nonnegative orthant after each ascent.
        for (i, b) in enumerate(sub.dual_blocks)
            raw = _block_raw(sub, x, b)
            scaled = b.scale .* raw
            if b.kind === :inequality
                b.kappa .= max.(b.kappa .+ b.rho .* scaled, 0)
                scaled_violations[i] = _inf(max.(scaled, 0))
                unscaled_violations[i] = _inf(max.(raw, 0))
            else
                b.kappa .+= b.rho .* scaled
                scaled_violations[i] = _inf(scaled)
                unscaled_violations[i] = _inf(raw)
            end
        end
        comp_prod_blocks = zeros(T, length(sub.comp_blocks))
        comp_split_blocks = zeros(T, length(sub.comp_blocks))
        comp_term_blocks = zeros(T, length(sub.comp_blocks))
        comp_res2 = zeros(T, length(sub.comp_blocks))
        # The complementarity channels ascend on both split-slack residuals;
        # the product and split residual are kept separately for termination.
        for (i, c) in enumerate(sub.comp_blocks)
            rG = c.scale .* (G .- c.sG)
            rH = c.scale .* (H .- c.sH)
            c.kappaG .+= c.rho .* rG
            c.kappaH .+= c.rho .* rH
            comp_prod_blocks[i] = _inf(G .* H)
            comp_split_blocks[i] = max(_inf(rG), _inf(rH))
            comp_term_blocks[i] = conditioned_comp ? c.scale^2 * comp_prod_blocks[i] : comp_prod_blocks[i]
            comp_res2[i] = sum(abs2, G .- c.sG) + sum(abs2, H .- c.sH)
        end
        sync_params!(sub)

        # Penalty safeguards and convergence use unscaled violations, matching
        # the caller's tolerances even when a residual conditioning scale is used.
        for (i, b) in enumerate(sub.dual_blocks)
            b.rho = _safe_penalty(b.rho, unscaled_violations[i], vprev[i], b.tol, config, b.scale)
            vprev[i] = unscaled_violations[i]
        end
        for (i, c) in enumerate(sub.comp_blocks)
            v = conditioned_comp ? comp_split_blocks[i] : comp_split_blocks[i] / c.scale
            c.rho = _safe_penalty(c.rho, v, vcprev[i], c.tol, config, c.scale)
            vcprev[i] = v
        end

        # A block that started nearly feasible carries no useful scale
        # information at x0. Balance it once its residual becomes measurable,
        # preserving any penalty growth already made by the safeguard.
        if config.auto_rho_init && (!all(ρbalanced) || !all(ρcompbalanced))
            fnow = abs(NLPModels.obj(sub.model, x))
            for (i, b) in enumerate(sub.dual_blocks)
                ρbalanced[i] && continue
                c = b.scale .* _block_raw(sub, x, b)
                b.kind === :inequality && (c = max.(c, 0))
                d2 = sum(abs2, c) / _sqscale(b.scale)
                if d2 > 1
                    b.rho = max(b.rho, _balanced_rho_eff(d2, fnow, config) / _sqscale(b.scale))
                    ρbalanced[i] = true
                end
            end
            for (i, c) in enumerate(sub.comp_blocks)
                ρcompbalanced[i] && continue
                if comp_res2[i] > 1
                    c.rho = max(c.rho, _balanced_rho_eff(comp_res2[i], fnow, config) / _sqscale(c.scale))
                    ρcompbalanced[i] = true
                end
            end
        end

        # Complementarity has two independent feasibility conditions: the raw
        # product certifies G*H≈0, while the split residual certifies that the
        # original G,H values agree with their nonnegative complementary slacks.
        feasible = true
        max_v = 0.0
        for (i, b) in enumerate(sub.dual_blocks)
            max_v = max(max_v, unscaled_violations[i])
            unscaled_violations[i] < b.tol || (feasible = false)
        end
        comp_res_u = 0.0
        for (i, c) in enumerate(sub.comp_blocks)
            split_test = conditioned_comp ? comp_split_blocks[i] : comp_split_blocks[i] / c.scale
            comp_res_u = max(comp_res_u, split_test)
            (comp_term_blocks[i] < c.tol && split_test < c.tol) || (feasible = false)
        end
        outer_viol_prev = max(max_v, comp_res_u, maximum(comp_term_blocks; init=0.0))

        stationarity = Inf
        if feasible && check_stationarity
            stationarity = _inf(evaluate_augmented(sub, x)[2])
        end
        stationary = !check_stationarity || (feasible && stationarity < stationarity_tol)
        if config.print_level >= 1
            @info "AuLa outer iteration" iteration=outer objective=NLPModels.obj(sub.model, x) max_constraint_violation=max_v complementarity=maximum(comp_prod_blocks; init=0.0) complementarity_split=comp_res_u stationarity=stationarity
        end
        if feasible && stationary
            converged = true
            break
        end

        # If repeated outer iterations make no primal movement while severe
        # infeasibility or nonstationarity remains, perturb along -grad(f),
        # reset dual state, and restart from the configured penalty seeds.
        if restart_count < sub.max_stagnation_restarts
            step_inf = _inf(x - previous_point)
            step_floor = 1e-10 * (1 + _inf(x))
            stagnant_outer = step_inf <= step_floor ? stagnant_outer + 1 : 0
            severe_infeas = any(unscaled_violations .> 10 .* [b.tol for b in sub.dual_blocks]) ||
                any(comp_term_blocks .> 10 .* [c.tol for c in sub.comp_blocks]) ||
                any(comp_split_blocks .> 10 .* [c.tol for c in sub.comp_blocks])
            severe_nonstat = check_stationarity && feasible && stationarity > 10 * stationarity_tol
            if stagnant_outer >= 3 && (severe_infeas || severe_nonstat)
                d = -NLPModels.grad(sub.model, x)
                dinf = _inf(d)
                if !all(isfinite, d) || dinf <= 1e-12
                    d = [isodd(k + restart_count) ? 1.0 : -1.0 for k in eachindex(x)]
                else
                    d ./= dinf
                end
                x .+= 1e-2 .* max.(1, abs.(x)) .* d
                _reset_penalties!(sub, config)
                if !isempty(sub.comp_blocks)
                    G, H = _current_comp(sub, x)
                    foreach(c -> project_complementarity!(c, G, H), sub.comp_blocks)
                end
                fill!(vprev, Inf); fill!(vcprev, Inf)
                outer_viol_prev = 1e3
                stagnant_outer = 0
                restart_count += 1
                sync_params!(sub)
            end
        end
    end

    elapsed = (time_ns() - start_time) / 1e9
    # Return raw, unscaled feasibility measures and MPCC certificates. The
    # support values pair positive complementarity legs with their split
    # multipliers; unlike active-set classifications, these need no threshold.
    G, H = _current_comp(sub, x)
    comp_prod = _inf(G .* H)
    negG = _inf(min.(G, 0)); negH = _inf(min.(H, 0))
    supportG = 0.0; supportH = 0.0
    for c in sub.comp_blocks
        supportG = max(supportG, _inf(min.(max.(G, 0), abs.(c.kappaG))))
        supportH = max(supportH, _inf(min.(max.(H, 0), abs.(c.kappaH))))
    end
    raw_violations = T[]
    for b in sub.dual_blocks
        raw = _block_raw(sub, x, b)
        push!(raw_violations, b.kind === :equality ? _inf(raw) : _inf(max.(raw, 0)))
    end
    eqviol = maximum([raw_violations[i] for (i,b) in enumerate(sub.dual_blocks) if b.kind === :equality]; init=0.0)
    ineqviol = maximum([raw_violations[i] for (i,b) in enumerate(sub.dual_blocks) if b.kind === :inequality]; init=0.0)
    stat = check_stationarity ? _inf(evaluate_augmented(sub, x)[2]) : 0.0
    cv = ConstraintViolation{T}[]
    for (i,b) in enumerate(sub.dual_blocks)
        push!(cv, ConstraintViolation(b.name, T(raw_violations[i]), T(raw_violations[i])))
    end
    for c in sub.comp_blocks
        split = max(_inf(G - c.sG), _inf(H - c.sH))
        push!(cv, ConstraintViolation(c.name, T(comp_prod), T(split)))
    end
    # Preserve whether any inner X update got stuck, even if later outer
    # iterations recovered. Convergence always takes precedence in the status.
    status = converged ? :converged : la_failure ? :linear_algebra_failure : :max_iterations
    msg = converged ? "Converged" : la_failure ? "Linear-algebra failure in X update" :
          check_stationarity ? "Max outer iterations reached (feasibility or stationarity)" :
          "Max outer iterations reached"
    return AulaResult(T.(x), T(NLPModels.obj(sub.model, x)), converged, status, msg,
        nouter, total_inner, total_gn, T(eqviol), T(ineqviol), T(comp_prod), T(stat),
        T(negG), T(negH), T(supportG), T(supportH), cv, elapsed,
        solver.gn.eval_seconds, solver.gn.factor_seconds)
end
