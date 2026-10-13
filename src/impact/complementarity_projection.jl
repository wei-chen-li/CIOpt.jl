"""Exact pairwise slack minimizer for the disjunctive complementarity set.

For each pair, compare the minimizers on the two branches `(sG ≥ 0, sH = 0)`
and `(sG = 0, sH ≥ 0)`. This is the same closed-form BCD update used in
`impact_solver`; no product relaxation or smoothing is introduced.
"""
function project_complementarity!(comp::CompBlock, G::AbstractVector, H::AbstractVector)
    length(G) == comp.dim == length(H) || throw(DimensionMismatch("complementarity block size"))
    ρ, s = comp.rho, comp.scale
    rs = ρ * s
    aG = s .* G .+ comp.kappaG ./ ρ
    aH = s .* H .+ comp.kappaH ./ ρ
    sG_one = max.(G .+ comp.kappaG ./ rs, zero(eltype(G)))
    sH_two = max.(H .+ comp.kappaH ./ rs, zero(eltype(H)))
    costG = min.(aG, zero(eltype(G))).^2 .+ aH.^2
    costH = aG.^2 .+ min.(aH, zero(eltype(H))).^2
    for i in eachindex(G)
        if costG[i] <= costH[i]
            comp.sG[i] = sG_one[i]
            comp.sH[i] = 0
        else
            comp.sG[i] = 0
            comp.sH[i] = sH_two[i]
        end
    end
    return comp.sG, comp.sH
end
