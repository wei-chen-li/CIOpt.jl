using Random
using Test
include("../../src/impact/complementarity_projection.jl")

@testset "Complementarity projection" begin
    # Complementarity projection vs brute force.
    rng = MersenneTwister(7)
    n = 200
    G, H, kG, kH = (Vector{Float64}(undef, n) for _ in 1:4)
    for i in 1:n
        G[i] = 4 * rand(rng) - 2
        H[i] = 4 * rand(rng) - 2
        kG[i] = 4 * rand(rng) - 2
        kH[i] = 4 * rand(rng) - 2
    end
    rho = 3.7
    sG, sH = similar(G), similar(H)
    project_complementarity!(G, H, kG, kH, rho, sG, sH)

    # Per-pair augmented penalty cost the projection minimises:
    #   ((G-sG) + kG/rho)^2 + ((H-sH) + kH/rho)^2
    function cost(i, y, z)
        a = G[i] - y + kG[i] / rho
        b = H[i] - z + kH[i] / rho
        return a * a + b * b
    end

    max_excess = 0.0
    max_comp = 0.0
    for i in 1:n
        c_proj = cost(i, sG[i], sH[i])
        # Brute-force over the two legs (z=0 sweep y>=0, y=0 sweep z>=0).
        best = Inf
        for k in 0:4000
            t = 4.0 * k / 4000.0  # candidate magnitude in [0,4]
            best = min(best, cost(i, t, 0.0), cost(i, 0.0, t))
        end
        max_excess = max(max_excess, c_proj - best)  # proj must be <= grid best (+eps)
        max_comp = max(max_comp, abs(sG[i] * sH[i]))
        if sG[i] < -1e-12 || sH[i] < -1e-12  # must stay nonnegative
            max_excess = 1e9
        end
    end
    @test max_excess < 1e-6
    @test max_comp < 1e-12
end
