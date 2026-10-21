"""
    AulaSubproblem(mpcc::MPCCModels.AbstractMPCCModel, config=AulaConfig())

Objective and constraint callbacks for `h(x,p) = 0`, `g(x,p) ≤ 0`, and
`0 ≤ G(x,p) ⊥ H(x,p) ≥ 0`. Each callback accepts `(x, params)`; constraint
Jacobians have one row per constraint. Missing constraint families are empty.
Callbacks and dimensions are derived from `mpcc`.

`params_info` holds constraint counts and initial penalty values. `make_params`
uses it to create a named tuple of mutable penalty, multiplier and slack arrays.
The active solve state is passed explicitly to evaluators and Gauss-Newton;
it is not stored in the subproblem.

Constraint callbacks and feasibility diagnostics use unscaled values.
Termination options and tolerances are read from `AulaConfig` at solve time.
"""
struct AulaSubproblem{M,T<:AbstractFloat,F,I<:NamedTuple}
    model::M
    nvar::Int
    functions::F
    params_info::I
end

function AulaSubproblem(mpcc::MPCCModels.AbstractMPCCModel{T},
                        config::AulaConfig=AulaConfig()) where {T}
    validate(config)
    x0 = NLPModels.get_x0(mpcc)
    isempty(x0) && throw(ArgumentError("the decision vector must be nonempty"))
    n = NLPModels.get_nvar(mpcc)
    nc = MPCCModels.get_ncc(mpcc)
    lvar, uvar = NLPModels.get_lvar(mpcc), NLPModels.get_uvar(mpcc)
    lcon, ucon = NLPModels.get_lcon(mpcc), NLPModels.get_ucon(mpcc)
    comp_vars = Set(MPCCModels.get_cc_var_indices(mpcc))
    if nc > 0
        all(isfinite,MPCCModels.lcomp_left(mpcc)) &&
        all(isfinite,MPCCModels.lcomp_right(mpcc)) ||
            throw(ArgumentError("complementarity legs require finite lower bounds"))
    end

    # Only complementarity-variable lower bounds are redundant. Keep upper and
    # fixed bounds. Constraint complementarity rows are absent from the public
    # ordinary constraint API, so no source-model row bookkeeping is needed.
    equalities, inequalities = Tuple{Symbol,Int,T,T}[], Tuple{Symbol,Int,T,T}[]
    for (source,lo,hi) in ((:variable,lvar,uvar),(:constraint,lcon,ucon))
        for i in eachindex(lo,hi)
            if isfinite(lo[i]) && lo[i] == hi[i]
                push!(equalities,(source,i,one(T),-lo[i]))
            else
                if isfinite(lo[i]) && !(source === :variable && i in comp_vars)
                    push!(inequalities,(source,i,-one(T),lo[i]))
                end
                isfinite(hi[i]) && push!(inequalities,(source,i,one(T),-hi[i]))
            end
        end
    end
    function bound_values(bounds,x)
        c = any(b[1] === :constraint for b in bounds) ? NLPModels.cons(mpcc,x) : T[]
        return T[sign*(source === :variable ? x[i] : c[i])+offset
                 for (source,i,sign,offset) in bounds]
    end
    function bound_jacobian(bounds,x)
        J = any(b[1] === :constraint for b in bounds) ? NLPModels.jac(mpcc,x) : spzeros(T,0,n)
        rows = [source === :variable ? sparse([1],[i],[sign],1,n) : sign*J[i:i,:]
                for (source,i,sign,_) in bounds]
        return isempty(rows) ? spzeros(T,0,n) : reduce(vcat,rows)
    end
    h(x,p) = bound_values(equalities,x)
    g(x,p) = bound_values(inequalities,x)
    jac_h(x,p) = bound_jacobian(equalities,x)
    jac_g(x,p) = bound_jacobian(inequalities,x)
    # Allocate outputs ourselves so even a custom AbstractMPCCModel needs no
    # private work arrays for the public in-place residual API.
    G(x,p) = nc == 0 ? T[] : MPCCModels.comp_res_left!(mpcc,x,Vector{T}(undef,nc))
    H(x,p) = nc == 0 ? T[] : MPCCModels.comp_res_right!(mpcc,x,Vector{T}(undef,nc))
    jac_G(x,p) = nc == 0 ? spzeros(T,0,n) : MPCCModels.jac_comp_left(mpcc,x)
    jac_H(x,p) = nc == 0 ? spzeros(T,0,n) : MPCCModels.jac_comp_right(mpcc,x)
    sense = NLPModels.get_minimize(mpcc) ? one(T) : -one(T)
    ℓ(x,p) = sense*NLPModels.obj(mpcc,x)
    ∇ℓ(x,p) = sense*NLPModels.grad(mpcc,x)
    ∇²ℓ(x,p) = sparse(NLPModels.hess(mpcc,x;obj_weight=sense))
    objective(x,p) = NLPModels.obj(mpcc,x)

    # Store only the information needed to allocate fresh solve state.
    nh, ng = length(equalities), length(inequalities)
    params_info = (neq=nh, nineq=ng, ncomp=nc,
                   rho_eq_init=T(config.rho_eq_init),
                   rho_ineq_init=T(config.rho_ineq_init),
                   rho_comp_init=T(config.rho_comp_init))
    for rho in (params_info.rho_eq_init, params_info.rho_ineq_init, params_info.rho_comp_init)
        isfinite(rho) && rho > 0 ||
            throw(ArgumentError("initial penalties must remain finite and positive in $T"))
    end
    # Model callbacks depend on x, not on the AuLa state; validate their dimensions
    # without allocating parameter arrays that would immediately be discarded.
    probe = (;)
    for (callback,dim) in ((h,nh),(g,ng),(G,nc),(H,nc))
        length(callback(x0,probe)) == dim || throw(DimensionMismatch("constraint dimensions"))
    end
    for (callback,dim) in ((jac_h,nh),(jac_g,ng),(jac_G,nc),(jac_H,nc))
        size(callback(x0,probe)) == (dim,n) || throw(DimensionMismatch("constraint Jacobian dimensions"))
    end
    f = (; ℓ, ∇ℓ, ∇²ℓ, h, jac_h, g, jac_g, G, jac_G, H, jac_H, objective)
    return AulaSubproblem{typeof(mpcc),T,typeof(f),typeof(params_info)}(mpcc,n,f,params_info)
end

"""
    make_params(sub::AulaSubproblem)

Allocate independent solve state using `sub.params_info`: one penalty per
constraint/pair and zero multipliers and complementarity slacks. Every call
returns fresh arrays, suitable for starting a solve or a stagnation restart.
"""
function make_params(sub::AulaSubproblem{M,T}) where {M,T}
    info = sub.params_info
    return (rho_eq=fill(info.rho_eq_init,info.neq), kappa_eq=zeros(T,info.neq),
            rho_ineq=fill(info.rho_ineq_init,info.nineq), kappa_ineq=zeros(T,info.nineq),
            rho_comp=fill(info.rho_comp_init,info.ncomp),
            kappaG=zeros(T,info.ncomp), kappaH=zeros(T,info.ncomp),
            sG=zeros(T,info.ncomp), sH=zeros(T,info.ncomp))
end

# Assemble the augmented least-squares rows once here, rather than requiring
# callers to duplicate multiplier shifts, inequality masks, and slack indexing.
function eval_residual(sub::AulaSubproblem, x, p)
    f = sub.functions
    eq = sqrt.(p.rho_eq) .* (f.h(x,p) + p.kappa_eq ./ p.rho_eq)
    ineq = sqrt.(p.rho_ineq) .* max.(f.g(x,p) + p.kappa_ineq ./ p.rho_ineq, 0)
    G = sqrt.(p.rho_comp) .* ((f.G(x,p) - p.sG) + p.kappaG ./ p.rho_comp)
    H = sqrt.(p.rho_comp) .* ((f.H(x,p) - p.sH) + p.kappaH ./ p.rho_comp)
    return vcat(eq,ineq,G,H)
end

function eval_jacobian(sub::AulaSubproblem, x, p)
    f = sub.functions
    active = f.g(x,p) + p.kappa_ineq ./ p.rho_ineq .> 0
    return sparse(vcat(spdiagm(0 => sqrt.(p.rho_eq)) * f.jac_h(x,p),
                       spdiagm(0 => sqrt.(p.rho_ineq) .* active) * f.jac_g(x,p),
                       spdiagm(0 => sqrt.(p.rho_comp)) * f.jac_G(x,p),
                       spdiagm(0 => sqrt.(p.rho_comp)) * f.jac_H(x,p)))
end

function eval_gh(sub::AulaSubproblem, x, p)
    return copy(sub.functions.G(x,p)), copy(sub.functions.H(x,p))
end

function eval_augmented_objective(sub::AulaSubproblem, x, p)
    r = eval_residual(sub,x,p)
    return sub.functions.ℓ(x,p) + dot(r,r)/2
end

eval_task_objective(sub::AulaSubproblem, x, p) = sub.functions.objective(x,p)
eval_task_gradient(sub::AulaSubproblem, x, p) = sub.functions.∇ℓ(x,p)
function eval_augmented_gradient_inf(sub::AulaSubproblem, x, p)
    grad = sub.functions.∇ℓ(x,p) + eval_jacobian(sub,x,p)' * eval_residual(sub,x,p)
    return isempty(grad) ? zero(eltype(x)) : norm(grad,Inf)
end

