# CIOpt/impact

CIOpt/impact ports the solver core of [IMPACT](https://github.com/JonasPflaume/IMPACT) to Julia and accepts MPCCs
represented by [MPCCModels.jl](https://github.com/MadNLP/MPCCModels.jl).

The main solver components have direct Julia counterparts:

| [IMPACT](https://github.com/JonasPflaume/IMPACT/tree/main/impact_solver) component | Julia counterpart |
|---|---|
| `aula_config.h` | [`aula_config.jl`](aula_config.jl) |
| `aula_solver.{h,cpp}` | [`aula_solver.jl`](aula_solver.jl) |
| `aula_subproblem.{h,cpp}` | [`aula_subproblem.jl`](aula_subproblem.jl) |
| `complementarity_projection.h` | [`complementarity_projection.jl`](complementarity_projection.jl) |
| `gauss_newton_solver.{h,cpp}` | [`gauss_newton_solver.jl`](gauss_newton_solver.jl) |

The Julia solver retains impact's safeguarded multiplier and penalty updates,
balanced penalty initialization, forcing sequence, stagnation restarts, split
complementarity slacks, closed-form branch projection, and sparse normal or
primal-dual saddle X updates. MPCCModels supplies a general NLP objective and
its Hessian, while impact's original subproblem uses least-squares residual rows.
CIOpt therefore forms the same augmented penalty channels and solves the X update
from the objective Hessian plus Gauss-Newton penalty curvature.

```julia
solver = CIOpt.ImpactSolver(mpcc)
result = CIOpt.solve!(solver)
```
