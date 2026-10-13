"""A grouped equality or one-sided inequality channel for the AuLa iteration."""
mutable struct DualBlock{T<:AbstractFloat}
    name::String
    kind::Symbol                 # :equality or :inequality
    source::Symbol               # :constraint or :variable
    indices::Vector{Int}
    signs::Vector{T}             # derivative sign for lower/upper bound rows
    offsets::Vector{T}           # h(x) = sign * source[index] + offset
    scale::T
    tol::T
    rho::T
    rho_init::T
    kappa::Vector{T}
end

"""A split-slack complementarity channel `0 <= G ⟂ H >= 0`."""
mutable struct CompBlock{T<:AbstractFloat}
    name::String
    dim::Int
    scale::T
    tol::T
    rho::T
    rho_init::T
    kappaG::Vector{T}
    kappaH::Vector{T}
    sG::Vector{T}
    sH::Vector{T}
end
