# Oscar.jl Integration

With [Oscar](https://www.oscar-system.org/) loaded, the ideal of a system of polynomial
expressions can be taken apart in Singular. The extension (`GiacOscarExt`) adds methods to
Oscar's own functions for a vector of expressions or equations `a ~ b`; the system crosses by the
[Nemo bridge](nemo.md), variables by name, and each ideal comes back as its reduced Gröbner basis,
of `GiacExpr`s, so a component reads as a system solved as far as it goes.

```julia
using Giac, Oscar
```

| Function | Gives |
|----------|-------|
| `minimal_primes(eqs)` | the irreducible components of the set where `eqs` vanish |
| `primary_decomposition(eqs)` | the primary components, `(primary, prime)`, with multiplicity |
| `radical(eqs)` | the polynomials vanishing on the solution set |
| `saturation(eqs, g)` | the ideal without the components on which `g` vanishes |

Every name in `eqs` is a variable, parameters included, so a component can be a relation between
a parameter and the state. The equilibria of `ẋ = μx − x³` form two branches, which cross at
the pitchfork:

```julia
@giac_var x μ
minimal_primes([μ*x - x^3])                 # [[x^2-μ], [x]]
```

A component of a primary decomposition larger than its prime is a solution of multiplicity
above one:

```julia
@giac_var x y
primary_decomposition([x^2, x*y])
# [(primary = [x], prime = [x]), (primary = [y, x^2], prime = [y, x])]
```

`saturation` removes a degenerate case: the equilibria of a system away from `g = 0`,

```julia
@giac_var x y a
saturation([a*x, a*y - x], a)               # [y, x]: for a ≠ 0, only the origin
```

Singular is not safe to call from two threads at once, so these calls take turns behind a lock;
parallel notebook cells that decompose ideals run one after another. Oscar's functions called
on its own ideals are not covered by the lock.

## Versions

The released Oscar 1.8 requires AlgebraicSolving 0.10 and Nemo 0.56; with AlgebraicSolving 0.10
the [msolve extension](../polynomial_systems.md) provides every function but
`real_root_classification`. Oscar's development version (1.9) uses AlgebraicSolving 0.13 and Nemo
0.57, so all of them load together:

```julia
import Pkg
Pkg.add(url = "https://github.com/oscar-system/Oscar.jl", rev = "master")
```
