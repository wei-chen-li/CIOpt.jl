"""
    AulaSubproblem(mpcc::MPCCModels.AbstractMPCCModel, config=AulaConfig())

Objective and constraint callbacks for `eq(x) = 0`, `ineq(x) ≤ 0`, and
`0 ≤ G(x) ⊥ H(x) ≥ 0`. Constraint Jacobians have one row per constraint.
Missing constraint families are empty. Model-only evaluators take just `x`;
augmented evaluators additionally take the active parameter state.
Callbacks and dimensions are derived from `mpcc`.
The model is stored as `_mpcc`; its variable count is queried directly.

`_params_info` holds constraint counts and initial penalty values. `make_params`
uses it to create a named tuple of mutable penalty, multiplier and slack arrays.
The active solve state is passed explicitly to evaluators and Gauss-Newton;
it is not stored in the subproblem.

Constraint callbacks and feasibility diagnostics use unscaled values.
Termination options and tolerances are read from `AulaConfig` at solve time.
"""
struct AulaSubproblem{M,F,I<:NamedTuple}
    _mpcc::M
    _functions::F
    _params_info::I
end

function AulaSubproblem(mpcc::MPCCModels.AbstractMPCCModel{T},
                        config::AulaConfig=AulaConfig()) where {T}
    nvar = NLPModels.get_nvar(mpcc)
    ncomp = MPCCModels.get_ncc(mpcc)
    lvar, uvar = NLPModels.get_lvar(mpcc), NLPModels.get_uvar(mpcc)
    lcon, ucon = NLPModels.get_lcon(mpcc), NLPModels.get_ucon(mpcc)
    comp_vars = Set(MPCCModels.get_cc_var_indices(mpcc))
    if ncomp > 0
        all(isfinite, MPCCModels.lcomp_left(mpcc)) &&
        all(isfinite, MPCCModels.lcomp_right(mpcc)) ||
            throw(ArgumentError("complementarity legs require finite lower bounds"))
    end

    # Only complementarity-variable lower bounds are redundant. Keep upper and
    # fixed bounds. Constraint complementarity rows are absent from the public
    # ordinary constraint API, so no source-model row bookkeeping is needed.
    equalities, inequalities = Tuple{Symbol, Int, T, T}[], Tuple{Symbol, Int, T, T}[]
    for (source, lo, hi) in ((:variable, lvar, uvar), (:constraint, lcon, ucon))
        for i in eachindex(lo, hi)
            if isfinite(lo[i]) && lo[i] == hi[i]
                push!(equalities, (source, i, one(T), -lo[i]))
            else
                if isfinite(lo[i]) && !(source === :variable && i in comp_vars)
                    push!(inequalities, (source, i, -one(T), lo[i]))
                end
                isfinite(hi[i]) && push!(inequalities, (source, i, one(T), -hi[i]))
            end
        end
    end

    function constraint_value(type, x)
        c = any(t[1] === :constraint for t in type) ? NLPModels.cons(mpcc, x) : T[]
        return T[sign * (source === :variable ? x[i] : c[i]) + offset
                 for (source, i, sign, offset) in type]
    end

    function constraint_jacobian(type, x)
        J = any(t[1] === :constraint for t in type) ? NLPModels.jac(mpcc, x) : spzeros(T, 0, nvar)
        rows = [source === :variable ? sparse([1], [i], [sign], 1, nvar) : sign * J[i:i, :]
                for (source, i, sign, _) in type]
        return isempty(rows) ? spzeros(T, 0, nvar) : reduce(vcat, rows)
    end

    sign = NLPModels.get_minimize(mpcc) ? one(T) : -one(T)
    ℓ(x; original=false) = (original ? one(T) : sign) * NLPModels.obj(mpcc, x)
    ∇ℓ(x) = sign * NLPModels.grad(mpcc, x)
    ∇²ℓ(x) = sign * sparse(NLPModels.hess(mpcc, x))

    eq(x) = constraint_value(equalities, x)
    ineq(x) = constraint_value(inequalities, x)
    jac_eq(x) = constraint_jacobian(equalities, x)
    jac_ineq(x) = constraint_jacobian(inequalities, x)

    G(x) = ncomp == 0 ? T[] : MPCCModels.comp_res_left(mpcc, x)
    H(x) = ncomp == 0 ? T[] : MPCCModels.comp_res_right(mpcc, x)
    jac_G(x) = ncomp == 0 ? spzeros(T, 0, nvar) : MPCCModels.jac_comp_left(mpcc, x)
    jac_H(x) = ncomp == 0 ? spzeros(T, 0, nvar) : MPCCModels.jac_comp_right(mpcc, x)

    functions = (; ℓ, ∇ℓ, ∇²ℓ, eq, jac_eq, ineq, jac_ineq, G, jac_G, H, jac_H)

    neq, nineq = length(equalities), length(inequalities)
    params_info = (neq=neq, nineq=nineq, ncomp=ncomp,
                   rho_eq_init=T(config.rho_eq_init),
                   rho_ineq_init=T(config.rho_ineq_init),
                   rho_comp_init=T(config.rho_comp_init))

    return AulaSubproblem(mpcc, functions, params_info)
end

"""
    make_params(sub::AulaSubproblem)

Allocate independent solve state using the stored dimensions and initial penalties:
one penalty per constraint/pair and zero multipliers and complementarity slacks.
"""
function make_params(sub::AulaSubproblem)
    info = sub._params_info
    T = typeof(info.rho_eq_init)
    for rho in (info.rho_eq_init, info.rho_ineq_init, info.rho_comp_init)
        isfinite(rho) && rho > 0 ||
            throw(ArgumentError("initial penalties must remain finite and positive in $T"))
    end
    return (rho_eq=fill(info.rho_eq_init,info.neq), kappa_eq=zeros(T,info.neq),
            rho_ineq=fill(info.rho_ineq_init,info.nineq), kappa_ineq=zeros(T,info.nineq),
            rho_comp=fill(info.rho_comp_init,info.ncomp),
            kappaG=zeros(T,info.ncomp), kappaH=zeros(T,info.ncomp),
            sG=zeros(T,info.ncomp), sH=zeros(T,info.ncomp))
end

get_nvar(sub::AulaSubproblem) = NLPModels.get_nvar(sub._mpcc)

eval_obj(sub::AulaSubproblem, x; original=false) = sub._functions.ℓ(x; original)
eval_obj_grad(sub::AulaSubproblem, x) = sub._functions.∇ℓ(x)
eval_obj_hess(sub::AulaSubproblem, x) = sub._functions.∇²ℓ(x)

eval_eq(sub::AulaSubproblem, x) = sub._functions.eq(x)
eval_ineq(sub::AulaSubproblem, x) = sub._functions.ineq(x)

eval_GH(sub::AulaSubproblem, x) = sub._functions.G(x), sub._functions.H(x)

function eval_al_residual(sub::AulaSubproblem, x, p)
    eq = sqrt.(p.rho_eq) .* (eval_eq(sub, x) + p.kappa_eq ./ p.rho_eq)
    ineq = sqrt.(p.rho_ineq) .* max.(eval_ineq(sub, x) + p.kappa_ineq ./ p.rho_ineq, 0)
    G, H = eval_GH(sub, x)
    comp_G = sqrt.(p.rho_comp) .* (G - p.sG + p.kappaG ./ p.rho_comp)
    comp_H = sqrt.(p.rho_comp) .* (H - p.sH + p.kappaH ./ p.rho_comp)
    return vcat(eq, ineq, comp_G, comp_H)
end

function eval_al_residual_jacobian(sub::AulaSubproblem, x, p)
    active = eval_ineq(sub, x) + p.kappa_ineq ./ p.rho_ineq .> 0
    f = sub._functions
    return sparse(vcat(spdiagm(0 => sqrt.(p.rho_eq)) * f.jac_eq(x),
                       spdiagm(0 => sqrt.(p.rho_ineq) .* active) * f.jac_ineq(x),
                       spdiagm(0 => sqrt.(p.rho_comp)) * f.jac_G(x),
                       spdiagm(0 => sqrt.(p.rho_comp)) * f.jac_H(x)))
end

function eval_al_objective(sub::AulaSubproblem, x, p)
    r = eval_al_residual(sub, x, p)
    return eval_obj(sub, x) + dot(r, r)/2
end

function eval_al_objective_grad_inf(sub::AulaSubproblem, x, p)
    grad = eval_obj_grad(sub, x) + eval_al_residual_jacobian(sub, x, p)' * eval_al_residual(sub, x, p)
    return isempty(grad) ? zero(eltype(x)) : norm(grad, Inf)
end
