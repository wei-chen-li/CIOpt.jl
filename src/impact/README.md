# CIOpt/impact

CIOpt/impact ports the solver core of [IMPACT](https://github.com/JonasPflaume/IMPACT) to Julia and accepts MPCCs
represented by [MPCCModels.jl](https://github.com/MadNLP/MPCCModels.jl).

The main solver components have direct Julia counterparts:

| [IMPACT](https://github.com/JonasPflaume/IMPACT/tree/main/impact_solver) component | Julia counterpart |
|---|---|
| `aula_config.h` | [`aula_config.jl`](aula_config.jl) |
| `dual_block.h` | [`dual_block.jl`](dual_block.jl) |
| `saddle_layout.h` | [`saddle_layout.jl`](saddle_layout.jl) |
| `complementarity_projection.h` | [`complementarity_projection.jl`](complementarity_projection.jl) |
| `aula_subproblem.{h,cpp}` | [`aula_subproblem.jl`](aula_subproblem.jl) |
| `gauss_newton_solver.{h,cpp}` | [`gauss_newton_solver.jl`](gauss_newton_solver.jl) |
| `aula_solver.{h,cpp}` | [`aula_solver.jl`](aula_solver.jl) |

The Julia solver retains impact's safeguarded multiplier and penalty updates,
balanced penalty initialization, forcing sequence, stagnation restarts, split
complementarity slacks, closed-form branch projection, and sparse normal or
primal-dual saddle X updates. The public entry point is `solve(mpcc)`.

MPCCModels supplies a general NLP objective and its Hessian, while impact's
original subproblem uses least-squares residual rows. CIOpt therefore forms the
same augmented penalty channels and solves the X update from the NLP Lagrangian
Hessian plus Gauss-Newton penalty curvature. Complementarity remains the exact
set `0 ≤ G ⟂ H ≥ 0`; there is no product relaxation.
