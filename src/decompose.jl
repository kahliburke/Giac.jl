# Writing an expression as a combination of given blocks with unknown polynomial multipliers,
# solved exactly by matching coefficients (GiacNemoExt). This file has the result type and its
# display.

"""
    Decomposition

The result of [`decompose`](@ref): `target = Σ multipliers[k] * blocks[k]`, with `directions`
the free directions of the solution family: adding `λ * directions[j][k]` to every
`multipliers[k]` keeps the identity, for any `λ`. `factors[k]` lists the building blocks, by
position, that `blocks[k]` is the product of.
"""
struct Decomposition
    target::GiacExpr
    blocks::Vector{GiacExpr}
    names::Vector{Union{String, Nothing}}     # a block's name, if it was given one
    multipliers::Vector{GiacExpr}
    directions::Vector{Vector{GiacExpr}}
    factors::Vector{Vector{Int}}
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
    terms = [(_multiplier_tex(m), _block_tex(n, b)) for (n, b, m) in zip(d.names, d.blocks, d.multipliers) if !iszero(m)]
    rhs = isempty(terms) ? "0" : replace(join((m * n for (m, n) in terms), " + "), "+ -" => "- ")
    print(io, "\$\$", giac_latex(d.target), " = ", rhs, "\$\$")
end
_paren_tex(m::GiacExpr) = (s = giac_latex(m); _node(m) isa AddNode ? "\\left(" * s * "\\right)" : s)

# a multiplier written before its block: nothing for 1, a sign for -1
function _multiplier_tex(m::GiacExpr)
    isone(m) && return ""
    isone(-m) && return "-"
    return _paren_tex(m) * " \\, "
end

"""
    Certified

The result of [`certify`](@ref): a member of a [`Decomposition`](@ref)'s family whose
multipliers each have the sign wanted. `λ` are the weights of the free directions that give it,
`multipliers` its multipliers, block by block, and `checks` the [`positivity`](@ref) report of
each signed multiplier (`nothing` for a zero one).
"""
struct Certified
    decomposition::Decomposition
    λ::Vector{Rational{BigInt}}
    multipliers::Vector{GiacExpr}
    checks::Vector{Union{Nothing, Positivity}}
end

_member(c::Certified) = (d = c.decomposition;
    Decomposition(d.target, d.blocks, d.names, c.multipliers, Vector{GiacExpr}[], d.factors))

function Base.show(io::IO, ::MIME"text/plain", c::Certified)
    d = c.decomposition
    println(io, "Certified: ", d.target, " =")
    for (n, b, m, r) in zip(d.names, d.blocks, c.multipliers, c.checks)
        iszero(m) && continue
        println(io, "  (", m, ") · ", _block_text(n, b), r === nothing ? "" : string("   ", _sign_word(r.sign, r.strict)))
    end
    isempty(c.λ) || print(io, "  weights of the free directions: ", join(_rational_text.(c.λ), ", "))
end
Base.show(io::IO, c::Certified) = print(io, "Certified(", count(!iszero, c.multipliers), " blocks)")
Base.show(io::IO, m::MIME"text/latex", c::Certified) = show(io, m, _member(c))

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
- `products = n`: use as blocks every product of up to `n` of the given ones (repeats allowed),
  each named by its factors, so that the search finds which products the target is made of.

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

"""
    certify(d::Decomposition; order = (), positive = (), param = (), signs = 1)

A member of the solution family of `d` whose multipliers each have one sign, certified by
coefficient signs under the ordering, as in [`positivity`](@ref). `signs` gives the sign wanted
for each multiplier (one number for all). Alternatively `block_signs` gives the known sign of each
building block and `target_sign` the sign to prove: each multiplier then takes the sign that
makes its term agree with the target, through the product of its block's factors' signs. The weights of the free directions are found by a
linear program (JuMP with HiGHS), made rational, and checked exactly in Nemo. Needs
`using JuMP, HiGHS, Nemo`.

Returns a [`Certified`](@ref Giac.Certified), which shows the member with each multiplier's
certified sign, or `nothing` when no member is certified this way.

```julia
using Nemo, JuMP, HiGHS
@giac_var x y
d = decompose(x^3 + y^3, [x + y, x*y]; degree = [2, 1])
certify(d; positive = (x, y), signs = (1, -1))   # x³+y³ = (x²+xy+y²)(x+y) - 2(x+y)·xy
```
"""
function certify end
