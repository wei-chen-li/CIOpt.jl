"""State and model data consumed by the AuLa and damped Gauss-Newton loops."""
mutable struct AulaSubproblem{M,T<:AbstractFloat}
    model::M
    nvar::Int
    ncon::Int
    ncomp::Int
    dual_blocks::Vector{DualBlock{T}}
    comp_blocks::Vector{CompBlock{T}}
    parameter_buffer::Vector{T}
    objective_value::T
    stationarity_enabled::Bool
    conditioned_complementarity::Bool
    stationarity_tol::T
    max_stagnation_restarts::Int
end

"""Current augmented least-squares/Newton data at one primal iterate."""
struct AugmentedEvaluation{T<:AbstractFloat}
    value::T
    gradient::Vector{T}
    lagrangian_hessian::SparseMatrixCSC{T,Int}
    penalty_jacobian::SparseMatrixCSC{T,Int}
    penalty_rhs::Vector{T}
    violations::Dict{String,T}
    comp_product::T
    comp_split_residual::T
end

function sync_params!(sub::AulaSubproblem{M,T}) where {M,T}
    empty!(sub.parameter_buffer)
    for b in sub.dual_blocks
        append!(sub.parameter_buffer, b.kappa)
        push!(sub.parameter_buffer, b.rho)
    end
    for c in sub.comp_blocks
        append!(sub.parameter_buffer, c.kappaG)
        append!(sub.parameter_buffer, c.kappaH)
        append!(sub.parameter_buffer, c.sG)
        append!(sub.parameter_buffer, c.sH)
        push!(sub.parameter_buffer, c.rho)
    end
    return sub.parameter_buffer
end

function eval_block_source(sub::AulaSubproblem, x, b::DualBlock)
    source = b.source === :constraint ? NLPModels.cons(sub.model, x) : x
    return b.signs .* source[b.indices] .+ b.offsets
end

function block_jacobian(sub::AulaSubproblem, x, b::DualBlock)
    Jsrc = b.source === :constraint ? NLPModels.jac(sub.model, x) : spzeros(eltype(x), 0, length(x))
    if b.source === :variable
        return sparse(b.signs .* sparse(1:length(b.indices), b.indices, ones(eltype(x), length(b.indices)),
                                        length(b.indices), sub.nvar))
    end
    return sparse(Diagonal(b.signs) * Jsrc[b.indices, :])
end

function _bound_data(lower, upper, atol; complementarity_variables=Set{Int}())
    T = promote_type(eltype(lower), eltype(upper))
    eq_idx, eq_val = Int[], T[]
    ineq_idx, ineq_sign, ineq_off = Int[], T[], T[]
    for i in eachindex(lower, upper)
        l, u = lower[i], upper[i]
        if isfinite(l) && isfinite(u) && abs(u - l) <= atol
            push!(eq_idx, i); push!(eq_val, (l + u) / 2)
        else
            if isfinite(l) && !(i in complementarity_variables)
                push!(ineq_idx, i); push!(ineq_sign, -one(T)); push!(ineq_off, l)
            end
            if isfinite(u)
                push!(ineq_idx, i); push!(ineq_sign, one(T)); push!(ineq_off, -u)
            end
        end
    end
    return (eq_idx, eq_val), (ineq_idx, ineq_sign, ineq_off)
end

"""Build the model adapter and its multiplier blocks from a vertical MPCC."""
function AulaSubproblem(mpcc::MPCCModels.AbstractMPCCModel, config::AulaConfig)
    if !MPCCModels.is_vertical(mpcc)
        error("Impact solver currently expects a vertical form MPCC.")
    end
    x0 = NLPModels.get_x0(mpcc)
    T = eltype(x0)
    nc, nv = NLPModels.get_ncon(mpcc), NLPModels.get_nvar(mpcc)
    lc, uc = NLPModels.get_lcon(mpcc), NLPModels.get_ucon(mpcc)
    lx, ux = NLPModels.get_lvar(mpcc), NLPModels.get_uvar(mpcc)
    atol = 10eps(T)
    ceq, cineq = _bound_data(lc, uc, atol)
    compvars = Set(vcat(MPCCModels.get_ind_cc1(mpcc), MPCCModels.get_ind_cc2(mpcc)))
    veq, vineq = _bound_data(lx, ux, atol; complementarity_variables=compvars)
    duals = DualBlock{T}[]

    if !isempty(ceq[1])
        push!(duals, DualBlock("equality", :equality, :constraint, ceq[1], ones(T, length(ceq[1])),
              -T.(ceq[2]), T(config.eq_scale), T(config.outer_tol_h), T(config.rho_eq_init),
              T(config.rho_eq_init), zeros(T, length(ceq[1]))))
    end
    if !isempty(cineq[1])
        push!(duals, DualBlock("inequality", :inequality, :constraint, cineq[1], T.(cineq[2]),
              T.(cineq[3]), T(config.ineq_scale), T(config.outer_tol_g), T(config.rho_ineq_init),
              T(config.rho_ineq_init), zeros(T, length(cineq[1]))))
    end
    if !isempty(veq[1])
        push!(duals, DualBlock("fix_point", :equality, :variable, veq[1], ones(T, length(veq[1])),
              -T.(veq[2]), T(config.fix_point_scale), T(config.outer_tol_h),
              T(config.rho_fix_point_init), T(config.rho_fix_point_init), zeros(T, length(veq[1]))))
    end
    if !isempty(vineq[1])
        push!(duals, DualBlock("variable_bounds", :inequality, :variable, vineq[1], T.(vineq[2]),
              T.(vineq[3]), T(config.fix_point_scale), T(config.outer_tol_g),
              T(config.rho_fix_point_init), T(config.rho_fix_point_init),
              zeros(T, length(vineq[1]))))
    end

    ncomp = MPCCModels.get_ncc(mpcc)
    comps = CompBlock{T}[]
    if ncomp > 0
        push!(comps, CompBlock("complementarity", ncomp, T(config.comp_scale),
              T(config.outer_tol_comp), T(config.rho_comp_init), T(config.rho_comp_init),
              zeros(T, ncomp), zeros(T, ncomp), zeros(T, ncomp), zeros(T, ncomp)))
    end
    sub = AulaSubproblem(mpcc, nv, nc, ncomp, duals, comps, T[], zero(T),
                         config.check_stationarity, config.conditioned_complementarity,
                         config.stationarity_tol, config.max_stagnation_restarts)
    sync_params!(sub)
    return sub
end

function _append_sparse_row!(I, J, V, rowidx, row, scale)
    inds, vals = findnz(sparse(vec(row)))
    for k in eachindex(inds)
        push!(I, rowidx); push!(J, inds[k]); push!(V, scale * vals[k])
    end
end

"""Evaluate objective, exact AL gradient/curvature, and penalty saddle rows."""
function evaluate_augmented(sub::AulaSubproblem{M,T}, x::AbstractVector;
                            update_projection::Bool=false) where {M,T}
    model = sub.model
    xv = T.(x)
    f = T(NLPModels.obj(model, xv))
    grad = T.(NLPModels.grad(model, xv))
    ylag = zeros(T, sub.ncon)
    cvals = T.(NLPModels.cons(model, xv))
    Jc = sparse(T.(NLPModels.jac(model, xv)))
    blocks = SaddleBlock{T}[]
    rowI, rowJ, rowV = Int[], Int[], T[]
    rhs = T[]
    violations = Dict{String,T}()
    value = f
    nrows = 0

    for b in sub.dual_blocks
        vals = b.signs .* (b.source === :constraint ? cvals[b.indices] : xv[b.indices]) .+ b.offsets
        Jb = b.source === :constraint ? sparse(Diagonal(b.signs) * Jc[b.indices, :]) :
             sparse(Diagonal(b.signs) * sparse(1:length(b.indices), b.indices,
                                               ones(T, length(b.indices)), length(b.indices), sub.nvar))
        unscaled_v = b.kind === :equality ? maximum(abs, vals; init=zero(T)) :
                     maximum(max.(vals, zero(T)); init=zero(T))
        violations[b.name] = unscaled_v
        start = nrows + 1
        for k in eachindex(vals)
            scaled = b.scale * vals[k]
            shifted = scaled + b.kappa[k] / b.rho
            active = b.kind === :equality || shifted > 0
            value += b.rho * (b.kind === :equality ? shifted^2 : max(shifted, zero(T))^2)
            if !active
                continue
            end
            q = b.kind === :equality ? b.kappa[k] + b.rho * scaled : b.rho * shifted
            row = Jb[k, :]
            row_scale = active ? sqrt(2 * b.rho) * b.scale : zero(T)
            _append_sparse_row!(rowI, rowJ, rowV, nrows + 1, row, row_scale)
            push!(rhs, active ? sqrt(2 * b.rho) * shifted : zero(T))
            nrows += 1
            if active && b.source === :constraint
                ylag[b.indices[k]] += 2 * b.scale * q * b.signs[k]
            end
        end
        push!(blocks, SaddleBlock(start, nrows - start + 1, b.rho, b.scale, b.name))
    end

    comp_product = zero(T)
    comp_split = zero(T)
    if sub.ncomp > 0
        G = T.(MPCCModels.comp_res_left(model, xv))
        H = T.(MPCCModels.comp_res_right(model, xv))
        JG = sparse(T.(MPCCModels.jac_comp_left(model, xv)))
        JH = sparse(T.(MPCCModels.jac_comp_right(model, xv)))
        for comp in sub.comp_blocks
            if update_projection
                project_complementarity!(comp, G, H)
            end
            rG, rH = G - comp.sG, H - comp.sH
            qG = comp.kappaG + comp.rho * comp.scale .* rG
            qH = comp.kappaH + comp.rho * comp.scale .* rH
            value += sum(comp.rho .* (comp.scale .* rG .+ comp.kappaG ./ comp.rho).^2)
            value += sum(comp.rho .* (comp.scale .* rH .+ comp.kappaH ./ comp.rho).^2)
            start = nrows + 1
            for k in 1:comp.dim
                _append_sparse_row!(rowI, rowJ, rowV, nrows + 1, JG[k, :], sqrt(2 * comp.rho) * comp.scale)
                push!(rhs, sqrt(2 * comp.rho) * (comp.scale * rG[k] + comp.kappaG[k] / comp.rho))
                nrows += 1
            end
            for k in 1:comp.dim
                _append_sparse_row!(rowI, rowJ, rowV, nrows + 1, JH[k, :], sqrt(2 * comp.rho) * comp.scale)
                push!(rhs, sqrt(2 * comp.rho) * (comp.scale * rH[k] + comp.kappaH[k] / comp.rho))
                nrows += 1
            end
            push!(blocks, SaddleBlock(start, nrows - start + 1, comp.rho, comp.scale, comp.name))
            comp_product = max(comp_product, maximum(abs, G .* H; init=zero(T)))
            comp_split = max(comp_split, maximum(abs, comp.scale .* rG; init=zero(T)),
                             maximum(abs, comp.scale .* rH; init=zero(T)))
            violations[comp.name] = max(maximum(abs, min.(G, 0); init=zero(T)),
                                        maximum(abs, min.(H, 0); init=zero(T)), comp_product)
        end
    end

    C = sparse(rowI, rowJ, rowV, nrows, sub.nvar)
    grad .+= C' * rhs
    Hlag = sparse(T.(NLPModels.hess(model, xv, ylag)))
    Hlag = sparse((Hlag + Hlag') / 2)
    layout = SaddleLayout(sub.nvar, 0, nrows, blocks)
    return value, grad, Hlag, C, rhs, layout, violations, comp_product, comp_split
end

function reset_duals!(sub::AulaSubproblem)
    for b in sub.dual_blocks
        fill!(b.kappa, 0)
        b.rho = b.rho_init
    end
    for c in sub.comp_blocks
        fill!(c.kappaG, 0); fill!(c.kappaH, 0)
        fill!(c.sG, 0); fill!(c.sH, 0)
        c.rho = c.rho_init
    end
    sync_params!(sub)
    return sub
end
