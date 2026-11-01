# Writing an expression as a combination of given blocks with unknown polynomial multipliers,
# solved exactly by matching coefficients (GiacNemoExt). This file has the result type and its
# display.

"""
    Decomposition

The result of [`decompose`](@ref): `target = Σ multipliers[k] * blocks[k]`, with `directions`
the free directions of the solution family: adding `λ * directions[j][k]` to every
`multipliers[k]` keeps the identity, for any `λ`.
"""
struct Decomposition
    target::GiacExpr
    blocks::Vector{GiacExpr}
    names::Vector{Union{String, Nothing}}     # a block's name, if it was given one
    multipliers::Vector{GiacExpr}
    directions::Vector{Vector{GiacExpr}}
end

"""
    combination(d::Decomposition, λ = zeros)

The multipliers of the member of the solution family at `λ`, one weight per free direction.
"""
function combination(d::Decomposition, λ = zeros(Int, length(d.directions)))
    length(λ) == length(d.directions) || throw(ArgumentError("one weight per free direction ($(length(d.directions)))"))
    return [m + sum((λ[j] * d.directions[j][k] for j in eachindex(λ)); init = zero(GiacExpr)) for (k, m) in enumerate(d.multipliers)]
end

_block_text(n, b) = n !== nothing ? n : _node(b) isa AddNode ? "(" * string(b) * ")" : string(b)
_block_tex(n, b) = n !== nothing ? _tex_name(n) : _paren_tex(b)

function Base.show(io::IO, ::MIME"text/plain", d::Decomposition)
    println(io, "Decomposition of ", d.target, ":")
    for (n, b, m) in zip(d.names, d.blocks, d.multipliers)
        iszero(m) || println(io, "  (", m, ") · ", _block_text(n, b))
    end
    k = length(d.directions)
    print(io, k == 0 ? "  unique" : "  $k free direction" * (k == 1 ? "" : "s"))
end
Base.show(io::IO, d::Decomposition) = print(io, "Decomposition(", length(d.blocks), " blocks, ", length(d.directions), " free)")

function Base.show(io::IO, ::MIME"text/latex", d::Decomposition)
    terms = [(_paren_tex(m), _block_tex(n, b)) for (n, b, m) in zip(d.names, d.blocks, d.multipliers) if !iszero(m)]
    rhs = isempty(terms) ? "0" : join((m * " \\, " * n for (m, n) in terms), " + ")
    print(io, "\$\$", giac_latex(d.target), " = ", rhs, "\$\$")
end
_paren_tex(m::GiacExpr) = (s = giac_latex(m); _node(m) isa AddNode ? "\\left(" * s * "\\right)" : s)

"""
    decompose(target, blocks; vars = all, degree = automatic, param = ())

Write `target` as `Σ Mₖ Bₖ` over the `blocks` `Bₖ`, each multiplier `Mₖ` a polynomial with
unknown rational coefficients, solved exactly by matching coefficients. Needs Nemo
(`using Nemo`).

- `blocks`: expressions, or `name => expression` pairs to name them in the result.
- `vars`: the variables a multiplier may contain (all variables, unless given).
- `degree`: the total degree of each multiplier in `vars`, one number or one per block; by
  default the target's degree less the block's.
- `param`: substitutions applied first, for relations among the variables (`l′ => t₂/g`).

Rational functions are cleared of denominators first. Returns a [`Decomposition`](@ref), whose
`directions` give the whole family of solutions, or `nothing` when there is none of these
degrees. Each multiplier's sign can then be checked with [`positivity`](@ref).

```julia
using Nemo
@giac_var p r L
target = (p*(L - r) + (L - p)*r)^2 - (p - r)^2 * L^2
decompose(target, [p*(L - p)*r*(L - r)])        # 4 · p(L-p)r(L-r)
```
"""
function decompose end
