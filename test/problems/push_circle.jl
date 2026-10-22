using ADNLPModels
using LinearAlgebra
using MPCCModels
using NLPModels
using Test

# export push_circle_model, test_push_circle_solution

const MASS = 1.0
const GRAVITY = 9.81
const GROUND_FRICTION = 0.1
const CONTACT_FRICTION = 0.1
const RADIUS = 0.3
const DT = 0.05
const TRANSLATION_SCALE = 1 / (GROUND_FRICTION * MASS * GRAVITY)

function dynamics(x, u)
    dx, dy = x[3] - x[1], x[4] - x[2]
    distance = sqrt(dx^2 + dy^2 + 1e-9)
    fn, ft, vx, vy = u
    Fx = (-fn * dx - ft * dy) / distance
    Fy = (-fn * dy + ft * dx) / distance
    return [TRANSLATION_SCALE * Fx, TRANSLATION_SCALE * Fy, vx, vy]
end

gap(x) = sqrt((x[3] - x[1])^2 + (x[4] - x[2])^2 + 1e-9) - RADIUS

"""
    push_circle_model()

Build a fresh horizon-100 MPCC for the reference circumnavigation scenario.
Returns the MPCC model, initial guess, initial state, goal, horizon, and state
block length. Decision variables are `[vec(X); vec(U)]`, with state columns
`[qx, qy, sx, sy]` and control columns `[fn, ft, vx, vy]`.
"""
function push_circle_model()
    # Reference scenario: the goal is the pusher's starting position, so the
    # pusher must get around the disk before it can push toward the goal.
    horizon = 100
    angle = 225 * pi / 180
    goal = 1.5 .* [cos(angle), sin(angle)]
    initial_state = vcat(zeros(2), goal)
    goal_state = vcat(goal, goal - RADIUS * goal / norm(goal))
    nstate = 4 * (horizon + 1)
    nvar = nstate + 4 * horizon

    # Multiple shooting: z = [vec(X); vec(U)], as in the C++ builder. Start
    # from a constant state trajectory and zero controls, rather than a solution.
    initial_guess = vcat(repeat(initial_state, horizon + 1), zeros(4 * horizon))
    lvar, uvar = fill(-Inf, nvar), fill(Inf, nvar)
    lvar[1:4] .= initial_state
    uvar[1:4] .= initial_state
    force_indices = collect(nstate + 1:4:nvar)
    lvar[force_indices] .= 0

    function objective(z)
        terminal = @view z[4 * horizon + 1:nstate]
        # Reference driver weights: stage ||u||² and terminal ||x - goal||².
        return 1e-2 * sum(abs2, @view(z[nstate + 1:end])) +
               100 * sum(abs2, terminal - goal_state)
    end

    function constraints(z)
        X = reshape(@view(z[1:nstate]), 4, horizon + 1)
        U = reshape(@view(z[nstate + 1:end]), 4, horizon)
        c = Vector{eltype(z)}(undef, 7 * horizon)
        for k in 1:horizon
            x, u = X[:, k], U[:, k]
            # Forward Euler, matching IMPACT/impact_solver/src/mpcc_stage.cpp.
            c[4 * k - 3:4 * k] .= X[:, k + 1] - x - DT * dynamics(x, u)
            c[4 * horizon + 2 * k - 1] = u[2] - CONTACT_FRICTION * u[1]
            c[4 * horizon + 2 * k] = -u[2] - CONTACT_FRICTION * u[1]
            c[6 * horizon + k] = gap(x)
        end
        return c
    end

    lcon = vcat(zeros(4 * horizon), fill(-Inf, 2 * horizon), zeros(horizon))
    ucon = vcat(zeros(4 * horizon), zeros(2 * horizon), fill(Inf, horizon))
    nlp = ADNLPModel(objective, initial_guess, lvar, uvar, constraints, lcon, ucon;
                     name="Push circle")
    gap_indices = collect(6 * horizon + 1:7 * horizon)
    # 0 ≤ normal force ⊥ disk gap ≥ 0, with the friction cone as ordinary inequalities.
    mpcc = MPCCModel(nlp, force_indices, gap_indices, fill(MPCCModels.VarCon, horizon))
    return (; mpcc, initial_guess, initial_state, goal, horizon, nstate)
end

function segment_distance(a, b)
    step = b - a
    step_squared = dot(step, step)
    fraction = iszero(step_squared) ? 0.0 : clamp(-dot(a, step) / step_squared, 0.0, 1.0)
    return norm(a + fraction * step)
end

"""
    test_push_circle_solution(problem, z; tolerance=1e-5, goal_tolerance=2e-3)

Check objective improvement, feasibility, contact, and circumnavigation for a
solver's decision vector `z` on a problem returned by `push_circle_model()`.
Solver status and reported diagnostics can be checked by each solver's tests.
"""
function test_push_circle_solution(problem, z::AbstractVector;
                                   tolerance=1e-4, goal_tolerance=2e-3)
    @testset "Push circle trajectory" begin
        @test all(isfinite, z)
        @test NLPModels.obj(problem.mpcc, z) <
              0.1 * NLPModels.obj(problem.mpcc, problem.initial_guess)

        X = reshape(@view(z[1:problem.nstate]), 4, problem.horizon + 1)
        U = reshape(@view(z[problem.nstate + 1:end]), 4, problem.horizon)
        @test X[:, 1] ≈ problem.initial_state atol=tolerance
        @test norm(X[1:2, end] - problem.goal, Inf) < goal_tolerance
        @test maximum(norm(X[:, k + 1] - X[:, k] - DT * dynamics(X[:, k], U[:, k]), Inf)
                      for k in 1:problem.horizon) < tolerance

        forces = U[1, :]
        gaps = [gap(X[:, k]) for k in 1:problem.horizon]
        @test minimum(forces) >= -tolerance
        @test minimum(gaps) >= -tolerance
        @test maximum(abs.(forces .* gaps)) < tolerance
        @test maximum(abs.(U[2, :]) - CONTACT_FRICTION .* forces) < tolerance

        # Initially the pusher is on the goal side, where pushing would move the
        # disk away from its target. It approaches without applying contact force.
        @test abs(forces[1]) < tolerance
        @test norm(X[1:2, 2] - X[1:2, 1], Inf) < tolerance
        @test norm(X[3:4, 2] - X[3:4, 1]) > 1e-2

        direction = problem.goal / norm(problem.goal)
        perpendicular = [-direction[2], direction[1]]
        relative = X[3:4, :] - X[1:2, :]
        along = vec(direction' * relative)
        lateral = vec(perpendicular' * relative)

        # Either direction around the disk is valid. The pusher must detour
        # laterally and reach the far side before most of the disk motion.
        @test maximum(abs, lateral) > 0.9 * RADIUS
        far_side = findfirst(<(-RADIUS / 2), along)
        @test far_side !== nothing
        if far_side !== nothing
            @test norm(X[1:2, far_side] - X[1:2, 1]) < 0.1 * norm(problem.goal)
        end

        # Check between knots too, so feasible endpoints cannot hide a jump through
        # the disk. Allow a small chord error from the discrete Euler trajectory.
        clearance = minimum(segment_distance(relative[:, k], relative[:, k + 1])
                            for k in 1:problem.horizon)
        @test clearance >= 0.99 * RADIUS

        # After going around, the pusher makes contact on the far side and pushes
        # toward the goal, finishing at the reference contact pose behind the disk.
        @test any(k -> along[k] < -RADIUS / 2 && forces[k] > 0.1 &&
                       abs(gaps[k]) < tolerance &&
                       dot(direction, X[1:2, k + 1] - X[1:2, k]) > tolerance,
                  1:problem.horizon)
        @test norm(relative[:, end] + RADIUS * direction, Inf) < goal_tolerance
    end
end
