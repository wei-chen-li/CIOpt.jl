"""Closed-form (Y, Z)-update used inside the BCD loop.

With the trajectory variables X fixed, the augmented-Lagrangian objective
decouples across complementarity pairs. For each scalar pair (y_i, z_i) under
the hard constraint 0 <= y_i ⊥ z_i >= 0, the exact minimiser is one of two
candidates (one branch per leg of the complementarity set):

  Case 1 (z_i = 0):  y_i* = max(0, G_i + kappaG_i / rho)
  Case 2 (y_i = 0):  z_i* = max(0, H_i + kappaH_i / rho)

We evaluate the penalty contribution of each candidate and keep the smaller one.
Writing the shifted residuals  aG = G + kappaG/rho,  aH = H + kappaH/rho,
the two costs reduce to  cost1 = min(aG,0)^2 + aH^2  and  cost2 = aG^2 + min(aH,0)^2.

This is an exact block minimisation, so it never increases the objective. The
implementation is vectorised over all complementarity pairs with Eigen arrays.
"""
function project_complementarity!(G::AbstractVector{T}, H::AbstractVector{T},
                                 kappaG::AbstractVector{T}, kappaH::AbstractVector{T}, rho_comp::T,
                                 sG::AbstractVector{T}, sH::AbstractVector{T}) where {T<:AbstractFloat}
    for i in eachindex(G, H, kappaG, kappaH, sG, sH)
        aG = G[i] + kappaG[i] / rho_comp
        aH = H[i] + kappaH[i] / rho_comp
        cost1 = abs2(min(aG, zero(T))) + abs2(aH)
        cost2 = abs2(aG) + abs2(min(aH, zero(T)))
        if cost1 <= cost2
            sG[i] = max(aG, zero(T))
            sH[i] = zero(T)
        else
            sG[i] = zero(T)
            sH[i] = max(aH, zero(T))
        end
    end
    return nothing
end
