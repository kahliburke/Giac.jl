# Polynomial systems

With [AlgebraicSolving.jl](https://algebraic-solving.github.io/) loaded, systems of polynomial
equations are solved exactly by [msolve](https://msolve.lip6.fr): real solutions isolated in
intervals guaranteed to contain them, exact parametrizations of all complex solutions, Gröbner
bases, elimination and normal forms. Its F4 implementation is usually faster than GIAC's
`gbasis` on systems of several variables.

```julia
using Giac, AlgebraicSolving
```

A system is a vector of polynomials, each meaning `= 0`, or of equations `a ~ b`, with integer
or rational coefficients. Its variables are those given, or else all of them in the order they
were declared. AlgebraicSolving's own functions take Giac expressions:

| Function | Result |
|:---|:---|
| `real_solutions(eqs, vars)` | the real solutions, each a [`Giac.RealSolution`](@ref) |
| `rational_solutions(eqs, vars)` | the rational solutions, as `x => value` pairs |
| `rational_parametrization(eqs, vars; t)` | all complex solutions, as a [`Giac.Parametrization`](@ref) |
| `groebner_basis(eqs, vars)` | the reduced Gröbner basis, degree reverse lexicographic in `vars` |
| `eliminate(eqs, xs; vars)` | the relations among the other variables, `xs` eliminated |
| `normal_form(e, eqs, vars)` | `e` reduced modulo `eqs` |
| `dimension(eqs, vars)` | the dimension of the complex solutions: `0` finitely many, `-1` none |

The first three need finitely many complex solutions.

## Real solutions

```julia
@giac_var x y
sols = real_solutions([x^2 + y^2 - 1, y - x^2])
# (x ≈ 0.7861513777574233, y ≈ 0.6180339887498949)
# (x ≈ -0.7861513777574233, y ≈ 0.6180339887498949)

s = sols[1]
s[x]                     # the midpoint of x's interval, a Rational{BigInt}
s.lower, s.upper         # the intervals' ends, one per variable
substitute(x*y, s...)    # evaluated at the midpoints
```

`precision` sets the intervals' width to about `2^-precision` (64 bits by default).

## Rational parametrization

All complex solutions, exactly: each variable is a rational function of a new variable `t`,
evaluated at the roots of one polynomial in `t`. `t` is a variable of the system when that one
separates the solutions, and otherwise a linear combination of them.

```julia
p = rational_parametrization([x^2 + y^2 - 1, y - x^2])
# Rational parametrization by T = -x-y, 4 solutions
#   -1+6*T-2*T^3+T^4 = 0
#   x = (-6+8*T+2*T^2)/(6-6*T^2+4*T^3)
#   y = (2+10*T-2*T^2-2*T^3)/(6-6*T^2+4*T^3)
```

## Elimination and normal forms

`eliminate` gives the relations that hold among the remaining variables on every solution, as
when finding the implicit equation of a parametrized curve, or eliminating a system's states to
relate its inputs and outputs:

```julia
@giac_var t
eliminate([x - (1 - t^2), y - 2t], t)          # [4*x+y^2-4]
```

`normal_form` reduces an expression modulo the system. Two polynomials have the same normal
form when their difference lies in the ideal of the system:

```julia
normal_form(x^4 + 2x^2*y^2 + y^4, [x^2 + y^2 - 1])      # 1
```

## API

```@docs
Giac.RealSolution
Giac.Parametrization
```
