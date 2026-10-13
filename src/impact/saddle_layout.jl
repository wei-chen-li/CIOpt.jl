"""A contiguous block of rows in the current AuLa penalty Jacobian."""
struct SaddleBlock{T<:AbstractFloat}
    row_start::Int
    count::Int
    rho::T
    scale::T
    name::String
end

"""Primal/dual row layout used by the sparse saddle-system X update."""
struct SaddleLayout{T<:AbstractFloat}
    n_z::Int
    n_cost::Int
    n_dual::Int
    blocks::Vector{SaddleBlock{T}}
end

SaddleLayout{T}(n::Int) where {T<:AbstractFloat} = SaddleLayout{T}(n, 0, 0, SaddleBlock{T}[])
