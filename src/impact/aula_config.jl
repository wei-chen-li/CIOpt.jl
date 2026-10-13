"""Solver settings corresponding to `impact_solver`'s `AulaConfig`."""
Base.@kwdef struct AulaConfig
    # Per-block penalty seeds and safeguarded updates.
    rho_fix_point_init::Float64 = 1
    rho_dynamics_init::Float64 = 1
    rho_eq_init::Float64 = 1
    rho_ineq_init::Float64 = 1
    rho_comp_init::Float64 = 1
    rho_max::Float64 = 1e6
    rho_scale::Float64 = 1.1
    penalty_decrease_ratio::Float64 = 0.5
    auto_rho_init::Bool = false
    auto_rho_clip_min::Float64 = 1e-2
    auto_rho_clip_max::Float64 = 1e2

    safeguard_factor::Float64 = 1e6
    fix_point_scale::Float64 = 1
    dynamics_scale::Float64 = 1
    eq_scale::Float64 = 1
    ineq_scale::Float64 = 1
    comp_scale::Float64 = 1
    outer_tol_h::Float64 = 1e-5
    outer_tol_g::Float64 = 1e-5
    outer_tol_comp::Float64 = 1e-5

    max_outer_iters::Int = 1000
    max_inner_iters::Int = 50
    inner_tol_init::Float64 = 1e-2
    inner_tol_final::Float64 = 1e-3
    inner_tol_ramp_start::Int = 3
    inner_tol_ramp_end::Int = 8

    check_stationarity::Bool = false
    conditioned_complementarity::Bool = false
    stationarity_tol::Float64 = 1e-5
    max_stagnation_restarts::Int = 0

    use_forcing_sequence::Bool = true
    forcing_cap::Float64 = 1e-2
    forcing_factor::Float64 = 0.1
    newton_max_iter::Int = 50
    newton_tol::Float64 = 1e-6
    newton_step_tol::Float64 = 1e-8
    newton_regularization::Float64 = 1e-5
    newton_lambda_min::Float64 = 1e-12
    newton_lambda_max::Float64 = 1e12
    newton_max_damping_tries::Int = 20

    use_saddle::Bool = true
    saddle_sigma_primal::Float64 = 1e-8
    saddle_equilibrate_dual::Bool = true
    saddle_refinement_steps::Int = 0
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
