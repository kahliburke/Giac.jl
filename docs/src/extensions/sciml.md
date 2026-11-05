# SciML Integration

With [SciMLBase](https://github.com/SciML/SciMLBase.jl) loaded (it comes with any SciML solver
package), a model written as Giac expressions becomes a SciML problem. The extension
(`GiacSciMLBaseExt`) compiles the right-hand side and its exact derivatives to native Julia code,
so a solver calls them at the speed of hand-written functions, without allocating.

```julia
using Giac, OrdinaryDiffEqRosenbrock
@giac_var x[1:2] g ℓ c
F = [x2, -g / ℓ * sin(x1) - c * x2]
prob = ODEProblem(F, [x1 => π / 2, x2 => 0], (0, 10), [g => 9.81, ℓ => 1, c => 0.2])
sol = solve(prob, Rodas5P())
```

The states and parameters are the variables of the assignments, in their order. Every other
name in `F` is an error, except the time variable, given as `t = …`.

| Function | Gives |
|----------|-------|
| `ODEProblem(F, u0, tspan, p; t)` | the initial-value problem `ẋ = F(x, p, t)` |
| `ODEFunction(F, x; p, t)` | its function, with `jac`, `tgrad` and `paramjac` |
| `NonlinearProblem(F, u0, p)` | the system `F(x, p) = 0`, from an initial guess |
| `NonlinearFunction(F, x; p)` | its function, with `jac` |

Assignments are written `x => values` for a vector of variables, or `[x1 => v1, x2 => v2]`.
Values may be exact (`π`, `1//3`, a `GiacExpr`); they become floating-point numbers.

## Exact derivatives

By default the function carries the exact Jacobian `∂F/∂x`, the time gradient `∂F/∂t` (with a
time variable) and the parameter Jacobian `∂F/∂p` (with parameters), each differentiated by Giac
and compiled. Implicit solvers such as `Rodas5P` and `FBDF` use the Jacobian in place of automatic
differentiation or finite differences. Keywords:

| Keyword | Default | |
|---------|---------|---|
| `jac`, `tgrad`, `paramjac` | `true` | compile that derivative |
| `sparse` | `false` | the Jacobian as a sparse matrix with the symbolic one's structure (needs SparseArrays) |
| `inplace` | `true` | in-place functions `f!(du, u, p, t)`; `false` for `f(u, p, t)` |
| `mass_matrix` | | passed to `ODEFunction`, for a semi-explicit DAE |

The other keywords of `ODEProblem` (`callback`, …) go to the problem.

## Symbolic indexing

The problem carries its variables, so the problem and its solution are indexed by them, and an
expression in them is computed along the solution:

```julia
sol[x1]                         # x₁ at every saved time
sol(2.5; idxs = [x1, x2])
sol[x1^2 + x2^2]                # an expression of the states, compiled on first use
prob.ps[g]                      # 9.81
remake(prob; p = [g => 3.7], u0 = [x1 => 0.3])
```

## Functions from `@giac_function`

A function from [`@giac_function`](@ref) takes SciML's arguments `(u, p, t)`: the states, then
optionally the parameters and the time variable. Its argument names are local, so values are
given in argument order:

```julia
@giac_function P(q[1:2], k[1:2]) = [q2, -k1 * sin(q1) - k2 * q2]
prob = ODEProblem(P, [π / 2, 0], (0, 10), [9.81, 0.2])
```

## Equilibria

`NonlinearProblem` takes the same model, and an equation `a ~ b` stands for `a - b = 0`:

```julia
using NonlinearSolve
solve(NonlinearProblem(F, [x1 => 3.0, x2 => 0.1], [g => 9.81, ℓ => 1, c => 0.2])).u    # [π, 0]
```
