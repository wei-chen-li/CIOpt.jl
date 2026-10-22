"""AuLa result. `z` is the raw decision vector.
Status is `:converged`, `:max_iterations`, or `:linear_algebra_failure`.
"""
Base.@kwdef mutable struct AulaResult{T<:AbstractFloat}
    z::Vector{T}

    objective_value::T = zero(T)            # f(z)
    equality_violation::T = zero(T)         # ‖eq(z)‖_inf
    inequality_violation::T = zero(T)       # ‖max(ineq(z), 0)‖_inf
    complementarity_violation::T = zero(T)  # ‖G(z) ∘ H(z)‖_inf
    stationarity_violation::T = zero(T)     # ‖∇ Lᵨ(z)‖_inf

    comp_neg_G::T = zero(T)                 # ‖min(G(z), 0)‖_inf
    comp_neg_H::T = zero(T)                 # ‖min(H(z), 0)‖_inf
    comp_support_G::T = zero(T)             # ‖min(max(G(z), 0), kappaG)‖_inf
    comp_support_H::T = zero(T)             # ‖min(max(H(z), 0), kappaH)‖_inf

    converged::Bool = false
    status::Symbol = :max_iterations
    status_message::String = ""

    outer_iterations::Int = 0
    total_inner_iterations::Int = 0
    total_gn_iterations::Int = 0

    solve_time::Float64 = 0.0
    eval_time::Float64 = 0.0
    factor_time::Float64 = 0.0
end


"""Safeguarded Augmented-Lagrangian solver with an inner BCD loop.

The solver consumes an AulaSubproblem, independent of the task that produced
it. In broad strokes it runs:

   Outer (Algorithm 1): clip every dual block's multiplier to the safeguard box,
   solve the AuLa subproblem inexactly, dual-ascend each multiplier, then apply a
   per-block *safeguarded* penalty increase (rho grows only when a block's
   violation fails to decrease enough between outer iterations).

   Inner (Algorithm 2): block coordinate descent alternating a damped
   Gauss-Newton X-update (GaussNewtonSolver) with the closed-form complementarity
   (Y,Z)-update (projectComplementarity), stopping on objective stagnation.
"""
mutable struct AulaSolver
    gn::GaussNewtonSolver
end
AulaSolver() = AulaSolver(GaussNewtonSolver())

function solve!(solver::AulaSolver, sub::AulaSubproblem, config::AulaConfig,
                z_init::AbstractVector)
    t_start = time_ns()
    length(z_init) == get_nvar(sub) && all(isfinite, z_init) ||
        throw(ArgumentError("AulaSolver: invalid initial decision vector"))

    # Bind the Gauss-Newton X-solver to this subproblem.
    init(solver.gn;
         ℓ=(x, _)->eval_obj(sub, x),
         ∇ℓ=(x, _)->eval_obj_grad(sub, x),
         ∇²ℓ=(x, _)->eval_obj_hess(sub, x),
         residual=(x, p)->eval_al_residual(sub, x, p),
         jacobian=(x, p)->eval_al_residual_jacobian(sub, x, p),
         config=GaussNewtonConfig(config))
    gn = solver.gn

    p = make_params(sub)
    T = eltype(p.rho_eq)
    z = Vector{T}(z_init)
    neq, nineq, ncomp = length(p.kappa_eq), length(p.kappa_ineq), length(p.kappaG)

    # Previous violations for the penalty safeguards and forcing sequence.
    INF = T(Inf)
    v_prev = (eq=fill(INF,neq), ineq=fill(INF,nineq), comp=fill(INF,ncomp))
    outer_viol_prev = T(1e3)

    s, eta, gamma = config.safeguard_factor, config.penalty_decrease_ratio, config.rho_scale
    G, H = zeros(T,ncomp), zeros(T,ncomp)

    # Balanced initial penalties (Birgin-Martinez): pick each block's rho_0 so that
    # at z_init the penalty term rho * ||c||^2 matches the objective's magnitude,
    # instead of trusting the build-time rho_*_init seeds.
    rho_balanced = (eq=trues(neq), ineq=trues(nineq), comp=trues(ncomp))
    balanced_rho(d2,f_abs) = clamp(2 * max(1, f_abs) / d2,
                                  config.auto_rho_clip_min, config.auto_rho_clip_max)
    if config.auto_rho_init
        f0 = abs(eval_obj(sub, z))
        eq, ineq = eval_eq(sub, z), eval_ineq(sub, z)
        G, H = eval_GH(sub, z)
        d2_comp = min.(abs2.(min.(G,0)) + abs2.(H), abs2.(G) + abs2.(min.(H,0)))
        for (rho, d2 ,done) in ((p.rho_eq,   abs2.(eq),           rho_balanced.eq),
                                (p.rho_ineq, abs2.(max.(ineq,0)), rho_balanced.ineq),
                                (p.rho_comp, d2_comp,             rho_balanced.comp))
            for i in eachindex(rho)
                done[i] = d2[i] > 1
                done[i] && (rho[i] = balanced_rho(d2[i], f0))
            end
        end
    end

    if config.print_level >= 1
        println(repeat("=",60),"\nBCD-AuLa (IMPACT) Solver\n",repeat("=",60))
        println("Variables: ", get_nvar(sub), ", equality: ", neq,", inequality: ", nineq,
                ", complementarity pairs: ", ncomp)
    end

    converged = false
    any_la_failure = false
    stagnant_outer = 0
    stagnation_restarts = 0
    outer_iters = 0
    total_inner = 0
    total_gn = 0

    for outer_iter in 0:(config.max_outer_iters - 1)
        outer_iters = outer_iter

        z_outer_start = copy(z)
        # Clip multipliers before the inner solve.
        p.kappa_eq .= clamp.(p.kappa_eq, -s, s)
        p.kappa_ineq .= clamp.(p.kappa_ineq, 0, s)
        p.kappaG .= clamp.(p.kappaG, -s, s)
        p.kappaH .= clamp.(p.kappaH, -s, s)

        inner_tol = (
            (outer_iter < config.inner_tol_ramp_start)
                ? config.inner_tol_init
                : (outer_iter < config.inner_tol_ramp_end)
                    ? 0.5 * (config.inner_tol_init + config.inner_tol_final)
                    : config.inner_tol_final)

        # Inner BCD loop: alternate the GN X-update and the closed-form slack update.
        phi_prev = eval_al_objective(sub, z, p)
        for inner in 0:(config.max_inner_iters-1)
            total_inner += 1

            gn_tol = (
                config.use_forcing_sequence
                    ? min(config.forcing_cap,
                          max(config.newton_tol,
                              config.forcing_factor * outer_viol_prev))
                    : -1.0)
            solve!(gn, p, z; grad_tol=gn_tol)
            any_la_failure |= gn.last_x_update_failed
            total_gn += gn.last_iterations

            if ncomp > 0
                G, H = eval_GH(sub, z)
                project_complementarity!(G, H, p.kappaG, p.kappaH, p.rho_comp, p.sG, p.sH)
            end

            phi_S = eval_al_objective(sub, z, p)
            config.print_level >= 2 && println("  Inner $inner: phi=$phi_S")
            abs(phi_prev - phi_S) < inner_tol && break
            phi_prev = phi_S
        end

        # Dual ascent and scaled violation measurement for equalities and inequalities.
        eq, ineq = eval_eq(sub,z), eval_ineq(sub,z)
        p.kappa_eq .= p.kappa_eq + p.rho_eq .* eq
        p.kappa_ineq .= max.(p.kappa_ineq + p.rho_ineq .* ineq, 0)
        v_eq = abs.(eq)
        v_ineq = max.(ineq,0)

        # Complementarity dual ascent on the split G/H residuals.
        comp_res_G, comp_res_H = G - p.sG, H - p.sH
        p.kappaG .+= p.rho_comp .* comp_res_G
        p.kappaH .+= p.rho_comp .* comp_res_H
        v_comp = max.(abs.(comp_res_G), abs.(comp_res_H))
        comp_products = abs.(G .* H)
        comp_prod = inf_norm(comp_products)

        # Update per-constraint penalties from violations.
        rho_cap = (config.auto_rho_init
                    ? max(config.rho_max, config.auto_rho_clip_max)
                    : config.rho_max)
        for (rho, v, old, tol) in ((p.rho_eq,   v_eq,   v_prev.eq,   config.outer_tol_h),
                                   (p.rho_ineq, v_ineq, v_prev.ineq, config.outer_tol_g),
                                   (p.rho_comp, v_comp, v_prev.comp, config.outer_tol_comp))
            for i in eachindex(rho)
                rho[i] = safeguarded_penalty(rho[i], v[i], old[i], tol, eta, gamma, rho_cap)
                old[i] = v[i]
            end
        end

        # Deferred balancing (auto_rho_init): a block that started (near-)feasible
        # was seeded with the scale-free neutral value; the first outer iteration
        # that hands it a measurable residual supplies the scale the balance formula
        # needs, and the block is balanced once, here. max() keeps any safeguard
        # growth that already happened. Costs one residual eval per still-unbalanced
        # block per outer iteration, and nothing once every block is balanced.
        if config.auto_rho_init && any(!all(done) for done in rho_balanced)
            f_cur = abs(eval_obj(sub, z))
            d2_comp = abs2.(comp_res_G) + abs2.(comp_res_H)
            for (rho, d2, done) in ((p.rho_eq,   abs2.(eq),           rho_balanced.eq),
                                    (p.rho_ineq, abs2.(max.(ineq,0)), rho_balanced.ineq),
                                    (p.rho_comp, d2_comp,             rho_balanced.comp))
                for i in eachindex(rho)
                    if !done[i] && d2[i] > 1
                        rho[i] = max(rho[i], balanced_rho(d2[i], f_cur))
                        done[i] = true
                    end
                end
            end
        end

        feasible = all(<(config.outer_tol_h), v_eq) &&
                   all(<(config.outer_tol_g), v_ineq) &&
                   all(<(config.outer_tol_comp), v_comp) &&
                   all(<(config.outer_tol_comp), comp_products)

        outer_viol_prev = max(inf_norm(v_eq), inf_norm(v_ineq), inf_norm(v_comp), comp_prod)

        stationarity = config.check_stationarity ? INF : zero(T)
        if config.check_stationarity && feasible
            stationarity = eval_al_objective_grad_inf(sub, z, p)
        end
        stationary = !config.check_stationarity ||
                     (feasible && stationarity < config.stationarity_tol)

        if config.print_level >= 1
            print("Outer ", outer_iter, ": equality=", inf_norm(v_eq), " inequality=", inf_norm(v_ineq))
            ncomp > 0 && print(" comp=", comp_prod, " comp_res=", inf_norm(v_comp))
            config.check_stationarity && print(" stationarity=",stationarity)
            println()
        end

        if feasible && stationary
            converged = true
            break
        end

        total_inner >= config.max_total_inner_iters && break

        if stagnation_restarts < config.max_stagnation_restarts
            step_inf = inf_norm(z - z_outer_start)
            step_floor = 1e-10 * (1.0 + inf_norm(z))
            stagnant_outer = step_inf <= step_floor ? stagnant_outer + 1 : 0
            severely_infeasible = any(>(10config.outer_tol_h), v_eq) ||
                                  any(>(10config.outer_tol_g), v_ineq) ||
                                  any(>(10config.outer_tol_comp), v_comp) ||
                                  any(>(10config.outer_tol_comp), comp_products)
            severely_nonstationary = config.check_stationarity && feasible &&
                                     stationarity > 10config.stationarity_tol
            if stagnant_outer >= 3 && (severely_infeasible || severely_nonstationary)
                direction = -eval_obj_grad(sub, z)
                d_inf = inf_norm(direction)
                if !all(isfinite,direction) || d_inf <= 1e-12
                    direction = T[iseven(i-1+stagnation_restarts) ? 1 : -1 for i in eachindex(z)]
                else
                    direction /= d_inf
                end
                z .+= 1e-2 .* max.(one(T), abs.(z)) .* direction
                p = make_params(sub)
                if ncomp > 0
                    G, H = eval_GH(sub, z)
                    project_complementarity!(G, H, p.kappaG, p.kappaH, p.rho_comp, p.sG, p.sH)
                end
                for old in v_prev
                    fill!(old, INF)
                end
                outer_viol_prev = T(1e3)
                stagnant_outer = 0
                stagnation_restarts += 1
            end
        end
    end

    # Fill the public result fields.
    res = AulaResult{T}(;
        z=z,
        converged=converged,
        outer_iterations=outer_iters,
        total_inner_iterations=total_inner,
        total_gn_iterations=total_gn)
    res.objective_value = eval_obj(sub, z; original=true)
    config.check_stationarity && (res.stationarity_violation = eval_al_objective_grad_inf(sub, z, p))

    G, H = eval_GH(sub, z)
    eq, ineq = eval_eq(sub, z), eval_ineq(sub, z)
    res.equality_violation = inf_norm(eq)
    res.inequality_violation = inf_norm(max.(ineq, 0))
    res.complementarity_violation = inf_norm(G .* H)
    res.comp_neg_G = inf_norm(min.(G, 0))
    res.comp_neg_H = inf_norm(min.(H, 0))
    res.comp_support_G = inf_norm(min.(max.(G, 0), abs.(p.kappaG)))
    res.comp_support_H = inf_norm(min.(max.(H, 0), abs.(p.kappaH)))

    res.solve_time = (time_ns() - t_start) * 1e-9
    res.eval_time = gn.eval_seconds
    res.factor_time = gn.factor_seconds
    if converged
        res.status = :converged
        res.status_message = "Converged"
    elseif any_la_failure
        res.status = :linear_algebra_failure
        res.status_message = "Linear-algebra failure: X-update factorisation stuck"
    else
        res.status = :max_iterations
        res.status_message = config.check_stationarity && isfinite(res.stationarity_violation) ?
            "Max iterations reached (feasibility or stationarity)" : "Max iterations reached"
    end

    if config.print_level >= 1
        println(repeat("=",60),"\n",converged ? "CONVERGED" : "STOPPED ($(res.status))",
                " in ",res.solve_time,"s, ",res.outer_iterations," outer / ",total_inner," inner\n",repeat("=",60))
    end
    return res
end


"""Per-constraint safeguarded penalty update.
rho is raised only when a constraint is still above tolerance and did not
decrease enough since the previous outer iteration.
"""
function safeguarded_penalty(rho, v_new, v_old, tol, eta, gamma, rho_max)
    violating = v_new >= tol
    insufficient_decrease = v_new > eta * v_old
    return violating && insufficient_decrease ? min(rho * gamma, rho_max) : rho
end


inf_norm(v) = isempty(v) ? zero(eltype(v)) : norm(v, Inf)
