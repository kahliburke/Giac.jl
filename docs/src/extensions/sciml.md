# SciML Integration

With [SciMLBase](https://github.com/SciML/SciMLBase.jl) loaded (it comes with any SciML solver
package), a model written as Giac expressions becomes a SciML problem. The extension
(`GiacSciMLBaseExt`) compiles the right-hand side and its exact derivatives to native Julia code,
so a solver calls them at the speed of hand-written functions, without allocating. A
subexpression the entries share is computed once.

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
| `SDEProblem(F, G, u0, tspan, p; t)` | the stochastic problem `dx = F dt + G dW` |
| `SDEFunction(F, G, x; p, t)` | its drift and noise functions |
| `NonlinearProblem(F, u0, p)` | the system `F(x, p) = 0`, from an initial guess |
| `NonlinearFunction(F, x; p)` | its function, with `jac` |

Assignments are written `x => values` for a vector of variables, or `[x1 => v1, x2 => v2]`.
Values may be exact (`π`, `1//3`, a `GiacExpr`); they become floating-point numbers.

## Exact derivatives

By default the function carries the exact Jacobian `∂F/∂x`, the time gradient `∂F/∂t` (with a
time variable) and the parameter Jacobian `∂F/∂p` (with parameters), each differentiated by Giac
and compiled. Implicit solvers such as `Rodas5P` and `FBDF` use the Jacobian in place of automatic
differentiation or finite differences. A right-hand side is differentiated only by the variables
it contains, so building a large sparse model costs time in proportion to the Jacobian's
nonzeros. Keywords:

| Keyword | Default | |
|---------|---------|---|
| `jac`, `tgrad`, `paramjac` | `true` | compile that derivative |
| `sparse` | `false` | the Jacobian as a sparse matrix with the symbolic one's structure (needs SparseArrays) |
| `inplace` | `true` | in-place functions `f!(du, u, p, t)`; `false` for `f(u, p, t)` |
| `static` | `false` | (for a problem) static arrays for the state and parameters, with out-of-place functions that return static arrays (needs StaticArrays) |
| `mass_matrix` | | passed to `ODEFunction`, for a semi-explicit DAE |

The other keywords of `ODEProblem` (`callback`, …) go to the problem.

## Equations of motion

A model may be written as equations in unknown functions of time and their derivatives, of any
order. [`first_order`](@ref) (part of Giac itself) brings them to a first-order system: each
function and its derivatives below the highest are states, written `q`, `q′`, and Giac solves
the equations for the highest derivatives, so they must be linear in them.

```julia
@giac_var t q(t) c k
D = Differential(t)
eqs = [D(D(q)) + c * D(q) + k * sin(q) ~ 0]
first_order(eqs)                    # d/dt [q, q′] = [q′, -c*q′-k*sin(q)]
prob = ODEProblem(eqs, [q => π / 2, D(q) => 0], (0, 10), [c => 0.2, k => 9.81])
sol[D(q)]                           # the state q′
sol[D(D(q))]                        # the solved acceleration
```

Initial values are given by the functions and derivatives themselves. An equation without
derivatives is an algebraic constraint on an unknown that appears without derivatives, and
makes the problem a DAE, with a mass matrix whose rows for those unknowns are zero.

## Inputs

`inputs` gives a model's input variables their values. An expression is substituted into the
model: a feedback law becomes part of the right-hand side and of its exact Jacobian. A Julia
function of time (a signal, an interpolated measurement) or of `(u, p, t)` (a controller) is
called by the compiled code:

```julia
@giac_var x[1:2] u w k1 k2
G = [x2, -x1 + u]
prob = ODEProblem(G, [x1 => 1, x2 => 0], (0, 10), [k1 => 2, k2 => 3];
                  inputs = [u => -k1 * x1 - k2 * x2 + w, w => t -> sin(t)])
sol[w]                              # an input, observed
```

A Julia function has no symbolic derivatives: one of the state leaves out the Jacobian and the
parameter Jacobian, and one of time the time gradient, so the solver differentiates the
compiled right-hand side itself.

## Events

`events` lists events `condition => affect`. The solver finds where each condition happens by
locating a zero crossing. `a ~ b` occurs when `a - b` crosses zero in either direction. An
inequality `a < b` occurs when it becomes true. Write a `~` condition in parentheses, because
`~` binds more loosely than `=>`. An affect is either assignments to states and parameters,
computed from the values at the event, or a Julia function of the integrator:

```julia
@giac_var x[1:2] g r k
ball = ODEProblem([x2, -g], [x1 => 1, x2 => 0], (0, 5), [g => 9.81, r => 0.8];
                  events = [x1 < 0 => [x2 => -r * x2]])
ODEProblem([-k * x1], [x1 => 1], (0, 2), [k => 1]; events = [(x1 ~ 1 // 2) => [k => 5]])
ODEProblem(eqs, u0, tspan, p; events = [(D(q) ~ 0) => terminate!])
```

Don't let an affect place the state exactly on the event's own surface, for example by setting
the ball's height to 0. The solver then misses the next crossing. A parameter that an event
changes is changed only within that solve. The problem's own
parameter values do not change. A `callback` given to the problem is kept alongside the events.

## Stochastic models

`SDEProblem(F, G, u0, tspan, p)` gives `dx = F dt + G dW`. The drift `F` works as for
`ODEProblem`, including equations, inputs and events. The noise `G` is compiled with it.
A vector `G` gives a separate noise on each state. A matrix `G` gives a column for each noise
process, and sets the problem's `noise_rate_prototype`:

```julia
using StochasticDiffEq
@giac_var x θ μ σ
ou = SDEProblem([θ * (μ - x)], [σ], [x => 1], (0, 10), [θ => 2, μ => 0, σ => 0.3])
solve(ou, SOSRI())
```

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
