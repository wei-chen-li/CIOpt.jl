"""
    SaddleBlock

Describes a contiguous group of rows in the penalty Jacobian assembled for one
augmented evaluation. `row_start` is one-based; `count` is the number of rows.
The remaining fields retain the block's penalty and conditioning metadata.
"""
struct SaddleBlock{T<:AbstractFloat}
    row_start::Int
    count::Int
    rho::T
    scale::T
    name::String
end

"""
    SaddleLayout

Records the primal dimension, cost-row count, penalty-row count, and penalty
block spans for an augmented evaluation. `evaluate_augmented` returns this
metadata alongside the assembled penalty Jacobian. The current Julia X update
forms its saddle system directly from that full Jacobian, so it does not yet use
these block spans for the cached, layout-specialized factorization implemented
in `impact_solver`.
"""
struct SaddleLayout{T<:AbstractFloat}
    n_z::Int
    n_cost::Int
    n_dual::Int
    blocks::Vector{SaddleBlock{T}}
end

SaddleLayout{T}(n::Int) where {T<:AbstractFloat} = SaddleLayout{T}(n, 0, 0, SaddleBlock{T}[])
