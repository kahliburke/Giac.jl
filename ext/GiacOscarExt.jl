# ---------------------------------------------------------------------------
# GiacOscarExt — polynomial ideals of Giac expressions, decomposed in Oscar.
#
# Loaded with Oscar. Oscar's polynomial rings are Nemo's, so a system crosses by the Nemo bridge
# (variables by name, in declared order) and its ideal goes to Singular. The methods extend
# Oscar's own functions, for systems given as vectors of expressions or equations:
#
#   minimal_primes(eqs)          the irreducible components of the solution set
#   primary_decomposition(eqs)   primary components with their primes, multiplicity included
#   radical(eqs)                 the polynomials vanishing on the solution set
#   saturation(eqs, g)           the components on which g is not identically zero
#
# Each result is a reduced Gröbner basis, as `GiacExpr`s. All the names in `eqs` are variables,
# parameters included, so a component may be a relation among parameters and states (a branch of
# equilibria, say).
# ---------------------------------------------------------------------------

module GiacOscarExt

using Giac
using Oscar
import Oscar: minimal_primes, primary_decomposition, radical, saturation

const _System = Union{AbstractVector{GiacExpr}, Tuple{Vararg{GiacExpr}}}

# Singular is not safe to call from two threads at once, and notebook cells run in parallel
const _SINGULAR = ReentrantLock()

_expression(e) = Giac.iscall(e) && Giac.symb_funcname(e) == "=" ? Giac.lhs(e) - Giac.rhs(e) : e

# the ideal of the polynomials `eqs` (numerators, for rational functions), and its ring
function _ideal(eqs)
    vals = Giac.to_nemo(GiacExpr[_expression(convert(GiacExpr, e)) for e in eqs])
    polys = [v isa Oscar.FracElem ? numerator(v) : v for v in vals]
    return ideal(parent(first(polys)), polys)
end

# an ideal's generators: its reduced Gröbner basis, in which each component reads as a system
# solved as far as it goes (`z - ρ + 1, x - y, y^2 - ρβ + β`)
_generators(I) = GiacExpr[Giac.to_giac(g) for g in groebner_basis(I; complete_reduction = true)]

"""
    minimal_primes(eqs::Vector{GiacExpr})

The irreducible components, over the rationals, of the set where the polynomials `eqs` vanish,
each as the generators of its prime ideal.

```julia
using Giac, Oscar
@giac_var x μ
minimal_primes([x^3 - μ*x])          # [[x^2-μ], [x]]: two branches of equilibria
```
"""
minimal_primes(eqs::_System) = lock(() -> [_generators(P) for P in minimal_primes(_ideal(eqs))], _SINGULAR)

"""
    primary_decomposition(eqs::Vector{GiacExpr})

The primary components of the ideal of `eqs`, each with its prime, as named tuples
`(primary, prime)` of generators. A primary component larger than its prime carries a
multiplicity: `x^2` over `x`.
"""
primary_decomposition(eqs::_System) = lock(_SINGULAR) do
    [(primary = _generators(Q), prime = _generators(P)) for (Q, P) in primary_decomposition(_ideal(eqs))]
end

"""
    radical(eqs::Vector{GiacExpr})

Generators of the radical of the ideal of `eqs`: every polynomial that vanishes wherever `eqs`
all do, without multiplicity.
"""
radical(eqs::_System) = lock(() -> _generators(radical(_ideal(eqs))), _SINGULAR)

"""
    saturation(eqs::Vector{GiacExpr}, g::GiacExpr)

The ideal of `eqs` with the components on which `g` vanishes removed: the solutions where `g` is
not identically zero, closed up.
"""
saturation(eqs::_System, g::GiacExpr) = lock(_SINGULAR) do
    I = _ideal(vcat(collect(eqs), g))
    R, G = base_ring(I), gens(I)
    _generators(saturation(ideal(R, G[1:(end - 1)]), ideal(R, [G[end]])))
end

end # module
