using Test
include("../../src/impact/aula_config.jl")

@testset "AulaConfig validation" begin
    config = AulaConfig()
    @test validate(config) === config
    for name in fieldnames(AulaConfig)
        value = getfield(config, name)
        if value isa AbstractFloat
            for invalid in (NaN, Inf, -Inf, -1.0)
                @test_throws ArgumentError validate(AulaConfig(; name => invalid))
            end
            if name != :penalty_decrease_ratio
                @test_throws ArgumentError validate(AulaConfig(; name => 0.0))
            end
        elseif value isa Int
            @test_throws ArgumentError validate(AulaConfig(; name => -1))
        elseif value isa Bool
            @test validate(AulaConfig(; name => !value)) isa AulaConfig
        end
    end
    for name in (:max_outer_iters, :max_inner_iters, :max_total_inner_iters, :newton_max_iter,
                 :newton_max_damping_tries)
        @test_throws ArgumentError validate(AulaConfig(; name => 0))
    end
    for kwargs in ((rho_scale=1.0,), (penalty_decrease_ratio=1.1,),
                   (print_level=4,), (auto_rho_clip_min=101.0,),
                   (inner_tol_ramp_start=9,), (newton_lambda_min=1e13,))
        @test_throws ArgumentError validate(AulaConfig(; kwargs...))
    end
    @test validate(AulaConfig(penalty_decrease_ratio=0.0)) isa AulaConfig
    @test validate(AulaConfig(penalty_decrease_ratio=1.0)) isa AulaConfig
    @test validate(AulaConfig(inner_tol_ramp_start=0, inner_tol_ramp_end=0,
                             max_stagnation_restarts=0, saddle_refinement_steps=0,
                             print_level=0)) isa AulaConfig
    @test validate(AulaConfig(auto_rho_clip_min=1.0, auto_rho_clip_max=1.0,
                             newton_lambda_min=1.0, newton_lambda_max=1.0,
                             print_level=3)) isa AulaConfig
end
