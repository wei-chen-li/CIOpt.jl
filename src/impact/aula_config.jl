"""
    AulaConfig(; kwargs...)

Configuration used by the augmented-Lagrangian solver.

The inner loop is block-coordinate descent: the complementarity slacks are frozen
at their closed-form projection while the trajectory variables take a
Gauss-Newton step, both inside the AuLa outer loop.
"""
Base.@kwdef struct AulaConfig
    # Initial AuLa penalties, grouped by constraint type.
    rho_eq_init::Float64 = 1
    rho_ineq_init::Float64 = 1
    rho_comp_init::Float64 = 1

    # Penalty schedule
    rho_max::Float64 = 1e6                 # Penalty cap
    rho_scale::Float64 = 1.1               # Penalty increase factor gamma > 1
    penalty_decrease_ratio::Float64 = 0.5  # Safeguard ratio eta in [0,1]

    # Balanced initial penalties (Birgin & Martinez, Practical Augmented Lagrangian
    # Methods, sec. 12.1). When on, the rho_*_init seeds above are ignored and each
    # dual block's initial EFFECTIVE penalty rho_eff = rho * scale^2 is computed at
    # z_init as
    #
    #     rho_eff = clip(2 max(1, |f(z0)|) / ||c(z0)||^2, clip_min, clip_max)
    #
    # with c the block's UNSCALED residual (violating part only for inequalities;
    # distance to the complementarity set for comp blocks). A block whose initial
    # residual is below 1 carries no scale information; it keeps its build-time
    # rho_*_init seed -- the only prior there is -- and is balanced at the first
    # outer iteration that hands it a measurable residual. Working in effective units makes the
    # whole scheme invariant under both the objective's scaling and the per-type
    # conditioning scales -- a nominal-rho window under-enforces any block with
    # scale << 1 (rho_max * scale^2 can sit orders below every other channel),
    # which is what produced mode-averaged chattering plans at small comp_scale.
    # In auto mode the safeguarded growth cap is per-block for the same reason:
    # max(rho_max, clip_max / scale^2). The safeguarded update still grows rho,
    # so this changes where the penalty path starts, never where it can end.
    #
    # Off by default, because it is a *replacement* for per-task penalty tuning and
    # the drivers in experiments/ are tuned. Measured A/B over all ten drivers with
    # nothing else changed: every task still converges, but the objective is 3.2x
    # worse in geometric mean and the wall time 1.6x higher, with the worst cases on
    # cart_transporter (33x objective, 14x time). It reaches a solution in fewer
    # outer iterations on several tasks -- it just reaches a worse one.
    #
    # Turn it on for a problem you have *not* tuned: it balances each block's
    # effective penalty against the objective instead of asking for a rho_max and a
    # rho_scale that only a sweep would find.
    auto_rho_init::Bool = false
    auto_rho_clip_min::Float64 = 1e-2  # clip floor for rho_eff = rho * scale^2
    auto_rho_clip_max::Float64 = 1e2   # clip ceiling for rho_eff

    # Multiplier safeguard, applied before each inner solve.
    safeguard_factor::Float64 = 1e6  # kappa in [-s, s], mu in [0, s]


    # Outer loop.
    max_outer_iters::Int = 1000
    outer_tol_h::Float64 = 1e-5     # Equality feasibility tolerance
    outer_tol_g::Float64 = 1e-5     # Inequality feasibility tolerance
    outer_tol_comp::Float64 = 1e-5  # Complementarity tolerance

    # Inner loop (BCD sweeps).
    max_inner_iters::Int = 50
    inner_tol_init::Float64 = 1e-2   # Stagnation tolerance, early outer iters
    inner_tol_final::Float64 = 1e-3  # Stagnation tolerance, later outer iters
    # Outer iteration at which the stagnation tolerance steps from inner_tol_init to
    # their midpoint, and then to inner_tol_final. The schedule is a ramp, not a
    # cliff: an early subproblem is far from the solution and solving it tightly
    # wastes inner work, while a late one needs the tight tolerance for the outer
    # loop to make progress at all.
    inner_tol_ramp_start::Int = 3
    inner_tol_ramp_end::Int = 8

    # Optional termination test on the infinity norm of the augmented gradient.
    check_stationarity::Bool = false
    stationarity_tol::Float64 = 1e-5
    # Restart after this many consecutive stagnant outer iterations if severe
    # infeasibility or nonstationarity remains. Zero disables restarts.
    max_stagnation_restarts::Int = 0

    # Solve the GN X-step loosely while feasibility is far off, then tighten as
    # the outer loop improves: grad_tol = min(cap, max(newton_tol, factor * viol)).
    use_forcing_sequence::Bool = true
    forcing_cap::Float64 = 1e-2
    forcing_factor::Float64 = 0.1

    # Damped Gauss-Newton/Newton X update. Damping resets to
    # newton_regularization at each X solve and is adjusted by a gain-ratio rule.
    newton_max_iter::Int = 50
    newton_tol::Float64 = 1e-6             # Gradient inf-norm tolerance
    newton_step_tol::Float64 = 1e-8        # Step-norm tolerance
    newton_regularization::Float64 = 1e-5  # Levenberg-Marquardt damping
    # Damping bounds and retry budget for the gain-ratio Levenberg-Marquardt
    # update. A new minimize! call always starts from newton_regularization.
    newton_lambda_min::Float64 = 1e-12
    newton_lambda_max::Float64 = 1e12
    newton_max_damping_tries::Int = 20

    # X-update linear-algebra backend (see SaddleLayout). The saddle form is the
    # default because it is much better conditioned than the normal equations on
    # dense single-shooting problems.
    use_saddle::Bool = true
    saddle_sigma_primal::Float64 = 1e-8   # Primal proximal floor on the (1,1) block
    saddle_equilibrate_dual::Bool = true  # Diagonal dual-row scaling
    saddle_refinement_steps::Int = 0      # Optional iterative refinement steps

    # 0 silent, 1 outer summary, 2 inner, 3 inner GN
    print_level::Int = 1
end


function validate(config::AulaConfig)
    for name in (:rho_eq_init, :rho_ineq_init, :rho_comp_init, :rho_max,
                 :auto_rho_clip_min, :auto_rho_clip_max, :safeguard_factor,
                 :outer_tol_h, :outer_tol_g, :outer_tol_comp,
                 :inner_tol_init, :inner_tol_final, :stationarity_tol,
                 :forcing_cap, :forcing_factor, :newton_tol, :newton_step_tol,
                 :newton_regularization, :newton_lambda_min, :newton_lambda_max,
                 :saddle_sigma_primal)
        value = getfield(config, name)
        isfinite(value) && value > 0 ||
            throw(ArgumentError("$name must be finite and positive"))
    end
    isfinite(config.rho_scale) && config.rho_scale > 1 ||
        throw(ArgumentError("rho_scale must be finite and greater than 1"))
    isfinite(config.penalty_decrease_ratio) && 0 <= config.penalty_decrease_ratio <= 1 ||
        throw(ArgumentError("penalty_decrease_ratio must be finite and in [0, 1]"))

    for name in (:max_outer_iters, :max_inner_iters, :newton_max_iter,
                 :newton_max_damping_tries)
        getfield(config, name) > 0 || throw(ArgumentError("$name must be positive"))
    end
    for name in (:inner_tol_ramp_start, :inner_tol_ramp_end,
                 :max_stagnation_restarts, :saddle_refinement_steps)
        getfield(config, name) >= 0 || throw(ArgumentError("$name must be nonnegative"))
    end
    0 <= config.print_level <= 3 ||
        throw(ArgumentError("print_level must be in 0:3"))
    config.auto_rho_clip_min <= config.auto_rho_clip_max ||
        throw(ArgumentError("auto_rho_clip_min must not exceed auto_rho_clip_max"))
    config.inner_tol_ramp_start <= config.inner_tol_ramp_end ||
        throw(ArgumentError("inner_tol_ramp_start must not exceed inner_tol_ramp_end"))
    config.newton_lambda_min <= config.newton_lambda_max ||
        throw(ArgumentError("newton_lambda_min must not exceed newton_lambda_max"))
    return config
end
