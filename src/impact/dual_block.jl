"""
    DualBlock

One grouped equality or one-sided inequality channel. Its residual is
`h(x) = signs .* source[indices] + offsets`; lower and upper bounds are converted
to the convention `h(x) = 0` for equalities and `h(x) <= 0` for inequalities.

Equalities update `kappa += rho * scale * h`. Inequalities project that update
onto the nonnegative multipliers: `kappa = max(0, kappa + rho * scale * h)`.
The solver grows `rho` only when the unscaled violation remains above `tol` and
has not fallen enough since the previous outer iteration.
"""
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

"""
    CompBlock

State for one split-slack complementarity channel `0 <= G ⟂ H >= 0`. The
projection updates `sG,sH` on the union of the two nonnegative axes; then the
outer loop ascends `kappaG,kappaH` using residuals `scale*(G-sG)` and
`scale*(H-sH)`. A complementarity block has its own penalty, scale, and
termination tolerance, just like a `DualBlock`.
"""
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
