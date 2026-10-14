"""
    AulaConfig(; kwargs...)

Controls the safeguarded augmented-Lagrangian (AuLa) outer loop and its inner
block-coordinate-descent (BCD) solve. Each outer iteration updates the dual
multipliers and penalties. Each inner sweep alternates a damped primal X step
with an exact projection of the complementarity slacks.

The `*_scale` fields condition residuals; `rho_*_init` fields seed their
penalties; and the outer tolerances are applied to unscaled feasibility
residuals (except when `conditioned_complementarity` is enabled). With a direct
`MPCCModels` input, ordinary equality and inequality bounds form shared blocks.
There is no separately identified dynamics block, so `dynamics_scale` and
`rho_dynamics_init` are retained for config compatibility but are not used by
this model adapter.
"""
Base.@kwdef struct AulaConfig
    # Initial penalty for each kind of residual block. Penalties grow by
    # rho_scale, subject to rho_max, only when safeguarded progress is too small.
    rho_fix_point_init::Float64 = 1
    rho_dynamics_init::Float64 = 1
    rho_eq_init::Float64 = 1
    rho_ineq_init::Float64 = 1
    rho_comp_init::Float64 = 1
    # rho_scale is gamma in the geometric increase; rho_max is its nominal cap.
    # eta=penalty_decrease_ratio prevents an increase once v_new <= eta*v_old.
    rho_max::Float64 = 1e6
    rho_scale::Float64 = 1.1
    penalty_decrease_ratio::Float64 = 0.5

    # Optional balanced initialization (off by default). For a block with an
    # informative initial residual c, choose effective stiffness
    # rho_eff = rho*scale^2 = clamp(2*max(1,abs(f(x0)))/||c||^2, clip_min, clip_max).
    # Inequality c contains only its violating part; complementarity c is the
    # distance to the union of its two feasible branches. A block starting with
    # ||c||^2 <= 1 keeps its configured seed until a later outer iterate gives
    # it a measurable residual. The effective units make the initialization
    # independent of both objective units and residual conditioning scales.
    auto_rho_init::Bool = false
    auto_rho_clip_min::Float64 = 1e-2
    auto_rho_clip_max::Float64 = 1e2
    # In auto mode, safeguarded growth is capped at
    # max(rho_max, auto_rho_clip_max/scale^2) for each block. This lets a
    # small conditioning scale reach the requested effective penalty ceiling.

    # Clip equality and complementarity multipliers to [-s,s], and inequality
    # multipliers to [0,s], before each inner solve.
    safeguard_factor::Float64 = 1e6

    # Positive residual multipliers used for conditioning. They affect the
    # effective penalty rho*scale^2 as well as the residual sent to the X step.
    fix_point_scale::Float64 = 1
    dynamics_scale::Float64 = 1
    eq_scale::Float64 = 1
    ineq_scale::Float64 = 1
    comp_scale::Float64 = 1
    outer_tol_h::Float64 = 1e-5    # Equality and fixed-variable feasibility.
    outer_tol_g::Float64 = 1e-5    # Positive part of one-sided inequalities.
    outer_tol_comp::Float64 = 1e-5 # Complementarity product and split residual.

    max_outer_iters::Int = 1000
    # The inner loop checks the change in the full augmented objective after
    # each X/slack sweep. It uses inner_tol_init before ramp_start, their
    # midpoint until ramp_end, then inner_tol_final. Ramp indices are zero-based.
    max_inner_iters::Int = 50
    inner_tol_init::Float64 = 1e-2
    inner_tol_final::Float64 = 1e-3
    inner_tol_ramp_start::Int = 3
    inner_tol_ramp_end::Int = 8

    # Optional termination test on the infinity norm of the augmented gradient.
    check_stationarity::Bool = false
    # If true, compare scale^2*|G_i*H_i| with outer_tol_comp; otherwise compare
    # the raw product. The split-residual feasibility check uses matching units.
    conditioned_complementarity::Bool = false
    stationarity_tol::Float64 = 1e-5
    # Restart after this many consecutive stagnant outer iterations if severe
    # infeasibility or nonstationarity remains. Zero disables restarts.
    max_stagnation_restarts::Int = 0

    # Inexact Newton forcing: grad_tol = min(forcing_cap,
    # max(newton_tol, forcing_factor*previous_outer_violation)).
    use_forcing_sequence::Bool = true
    forcing_cap::Float64 = 1e-2
    forcing_factor::Float64 = 0.1
    # Damped Gauss-Newton/Newton X update. Damping resets to
    # newton_regularization at each X solve and is adjusted by a gain-ratio rule.
    newton_max_iter::Int = 50
    newton_tol::Float64 = 1e-6
    newton_step_tol::Float64 = 1e-8
    newton_regularization::Float64 = 1e-5
    # Damping bounds and retry budget for the gain-ratio Levenberg-Marquardt
    # update. A new minimize! call always starts from newton_regularization.
    newton_lambda_min::Float64 = 1e-12
    newton_lambda_max::Float64 = 1e12
    newton_max_damping_tries::Int = 20

    # Select the saddle KKT step or normal equations for the X update. The saddle
    # form avoids explicitly building C'*C and is less sensitive to large penalty
    # values; this Julia path assembles a fresh system from the current Jacobian.
    use_saddle::Bool = true
    saddle_sigma_primal::Float64 = 1e-8
    # Congruence-scale each dual row using its Jacobian norm to reduce badly
    # scaled pivots while leaving the primal step unchanged in exact arithmetic.
    saddle_equilibrate_dual::Bool = true
    # Optional iterative refinement of the saddle solve, using the same factors.
    saddle_refinement_steps::Int = 0
    # 0 is quiet; values >=1 log the outer-iteration summary.
    print_level::Int = 1
end

function validate(config::AulaConfig)
    config.max_outer_iters > 0 && config.max_inner_iters > 0 && config.newton_max_iter > 0 ||
        throw(ArgumentError("iteration limits must be positive"))
    config.rho_scale > 1 && config.rho_max > 0 &&
        0 <= config.penalty_decrease_ratio <= 1 ||
        throw(ArgumentError("invalid safeguarded penalty configuration"))
    config.newton_regularization > 0 && config.newton_lambda_min > 0 &&
        config.newton_lambda_max >= config.newton_lambda_min ||
        throw(ArgumentError("invalid Levenberg-Marquardt damping configuration"))
    config.rho_fix_point_init > 0 && config.rho_dynamics_init > 0 &&
        config.rho_eq_init > 0 && config.rho_ineq_init > 0 && config.rho_comp_init > 0 ||
        throw(ArgumentError("initial penalty values must be positive"))
    config.outer_tol_h > 0 && config.outer_tol_g > 0 && config.outer_tol_comp > 0 ||
        throw(ArgumentError("constraint tolerances must be positive"))
    config.max_stagnation_restarts >= 0 ||
        throw(ArgumentError("max_stagnation_restarts must be nonnegative"))
    if config.check_stationarity
        isfinite(config.stationarity_tol) && config.stationarity_tol > 0 ||
            throw(ArgumentError("stationarity_tol must be finite and positive"))
    end
    config.auto_rho_clip_min > 0 && config.auto_rho_clip_max >= config.auto_rho_clip_min ||
        throw(ArgumentError("invalid automatic penalty initialization bounds"))
    for (name, s) in ((:fix_point_scale, config.fix_point_scale),
                      (:dynamics_scale, config.dynamics_scale),
                      (:eq_scale, config.eq_scale), (:ineq_scale, config.ineq_scale),
                      (:comp_scale, config.comp_scale))
        isfinite(s) && s > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    return config
end
