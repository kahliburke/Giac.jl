# The sign of an expression under an ordering of its variables, read off coefficient signs after
# writing each ordered variable as the one before it plus a positive gap. The exact check is done
# in Nemo (GiacNemoExt); this file has the result types and their display.

"""
    SignReport

The sign of one polynomial (a numerator or a denominator) after the gap substitution:
- `sign`: `1` or `-1` when every coefficient has that sign, `0` when not decided;
- `strict`: whether the sign is strict (all variables strictly positive, polynomial not zero);
- `polynomial`: the polynomial in the gap and positive variables;
- `against`: its terms of the minority sign, the ones that stop a certificate, largest
  coefficient first;
- `free`: variables whose sign is not known, which stop a certificate when they appear;
- `terms`: the numbers of positive and of negative terms;
- `degrees`: the range of the terms' total degrees;
- `opposing`: each variable occurring in the terms against, with how many of them it is in.

[`group_terms`](@ref Giac.group_terms) sorts the terms against by the variables chosen.
"""
struct SignReport
    sign::Int
    strict::Bool
    polynomial::GiacExpr
    against::Vector{GiacExpr}
    free::Vector{GiacExpr}
    terms::Tuple{Int, Int}
    degrees::UnitRange{Int}
    opposing::Vector{Pair{String, Int}}
end

"""
    Positivity

The result of [`positivity`](@ref): `sign` is `1`, `-1` or `0` (not decided), for the whole
expression; `numerator` and `denominator` are [`SignReport`](@ref)s; `gaps` the substitution
used, each ordered variable written as the one before it plus a gap `Δ…`.
"""
struct Positivity
    sign::Int
    strict::Bool
    numerator::SignReport
    denominator::SignReport
    gaps::Vector{Pair{GiacExpr, GiacExpr}}
end

Base.Bool(p::Positivity) = p.sign == 1 && p.strict

function _sign_word(sign, strict)
    sign == 0 && return "not decided by coefficient signs"
    return (sign > 0 ? (strict ? "positive" : "nonnegative") : (strict ? "negative" : "nonpositive"))
end

# GIAC writes the identifier Δ as Delta
_gap_text(x) = replace(string(x), "Delta" => "Δ")

function Base.show(io::IO, ::MIME"text/plain", p::Positivity)
    println(io, "Positivity: ", _sign_word(p.sign, p.strict))
    for (label, r) in (("numerator", p.numerator), ("denominator", p.denominator))
        is_constant(r.polynomial) && r.sign != 0 && label == "denominator" && continue
        print(io, "  ", label, ": ")
        _summary(io, r; indent = "    ")
    end
    isempty(p.gaps) || print(io, "  gaps: ", join((_gap_text(first(g)) * " = " * _gap_text(last(g)) for g in p.gaps), ", "))
end

# a report in a few lines: the sign, the term counts and degrees, and for a mixed sign what is
# against it: which variables, how often, and the terms with the largest coefficients
function _summary(io::IO, r::SignReport; indent = "  ")
    pos, neg = r.terms
    println(io, _sign_word(r.sign, r.strict), "; ", pos + neg, pos + neg == 1 ? " term" : " terms",
            " (", pos, " positive, ", neg, " negative), degree ", first(r.degrees) == last(r.degrees) ? first(r.degrees) : r.degrees)
    isempty(r.free) || println(io, indent, "unknown sign: ", join(_gap_text.(r.free), ", "))
    isempty(r.against) && return
    n = length(r.against)
    println(io, indent, n, n == 1 ? " term" : " terms", " against, in: ",
            join((_gap_text(v) * " ×" * string(k) for (v, k) in first(r.opposing, 8)), ", "), length(r.opposing) > 8 ? ", …" : "")
    println(io, indent, "largest: ", join(_gap_text.(first(r.against, 3)), ", "), n > 3 ? ", …" : "")
    r.sign == 0 && isempty(r.free) && println(io, indent, "(a sum-of-squares certificate may still exist)")
end

Base.show(io::IO, ::MIME"text/plain", r::SignReport) = (print(io, "SignReport: "); _summary(io, r))
Base.show(io::IO, r::SignReport) = print(io, "SignReport(", _sign_word(r.sign, r.strict), ", ", sum(r.terms), " terms)")
Base.show(io::IO, p::Positivity) = print(io, "Positivity(", _sign_word(p.sign, p.strict), ")")

"""
    positivity(e; order = (), positive = (), param = (), strict = true)

Whether `e` is positive given an ordering of its variables, certified exactly by coefficient
signs. Needs Nemo to be loaded (`using Nemo`).

- `order`: a chain `(0, u₀, u, t₁)`, meaning `0 < u₀ < u < t₁`, or several chains in a vector.
  A chain may start at a number (`(1, g)` is `g > 1`) or at a variable whose sign is not given.
  Each variable after the first is written as the one before it plus a gap `Δt₁ > 0`.
- `positive`: variables that are positive and in no chain, such as weights.
- `param`: substitutions applied first, for relations the chains cannot state: with
  `t₂² = l′u′`, `param = (l′ => t₂/g, u′ => t₂*g)` and `g > 1` in `order`.
- `strict = false` reads each chain as `≤`.

A rational function is put over a common denominator, and the numerator and denominator are
checked separately. When every coefficient of a polynomial in the gaps and positive variables
is positive, so is the polynomial; when the signs are mixed, the result lists the terms against
the majority sign. Returns a [`Positivity`](@ref); `Bool(result)` is `true` for a certified
strict positive.

```julia
using Nemo
@giac_var u t₁ t₂ A B
positivity((t₂ - u) * (t₁ - u) * A; order = (u, t₁, t₂), positive = (A,))   # positive
positivity(A^2 - 2A*B + B^2; positive = (A, B))                          # not decided: -2AB
```
"""
function positivity end

_chains(order) = isempty(order) ? Any[] :
    (order isa Tuple && !(first(order) isa Union{Tuple, AbstractVector})) ? Any[order] : collect(Any, order)

# Each ordered variable as the one before it plus a gap `Δv`: the substitutions, and the names
# known positive (the gaps, the `positive` variables, a chain's first variable after 0)
function _gap_substitution(order, positive)
    gaps = Pair{GiacExpr, GiacExpr}[]
    known = Set{String}(string.(positive))
    for chain in _chains(order)
        items = collect(Any, chain)
        start = popfirst!(items)
        acc = convert(GiacExpr, start)
        if start isa Number && iszero(start) && !isempty(items)
            v = popfirst!(items)                   # 0 < v: v itself is the first gap
            push!(known, string(v))
            acc = convert(GiacExpr, v)
        end
        for v in items
            Δ = giac_identifier("Δ" * string(v))
            push!(known, string(Δ))
            acc = acc + Δ
            push!(gaps, convert(GiacExpr, v) => acc)
        end
    end
    return gaps, known
end

"""
    Giac.group_terms(terms, vars)

The `terms` (the terms against in a [`SignReport`](@ref), or any list of monomials) grouped by
their part in the variables `vars`: pairs `monomial in vars => sum of the cofactors`, the group
with most terms first. Which gaps, and which weights, the terms against a certificate share.

```julia
r = positivity(e; order, positive)
Giac.group_terms(r.numerator.against, (A, A′, A₃))
```
"""
function group_terms(terms, vars)
    names = Set(string.(vars))
    groups = Dict{Node, Vector{Node}}()
    for t in terms
        n = _node(t)
        coef, factors = n isa MulNode ? (n.coef, n.factors) : n isa NumNode ? (n.v, Pair{Node, Node}[]) : (1, Pair{Node, Node}[n => _ONE])
        inside = Pair{Node, Node}[f for f in factors if first(f) isa SymNode && first(f).name in names]
        outside = Pair{Node, Node}[f for f in factors if !(first(f) isa SymNode && first(f).name in names)]
        key = isempty(inside) ? _ONE : _mul((_pow(b, e) for (b, e) in inside)...)
        push!(get!(groups, key, Node[]), _mul(NumNode(coef), (_pow(b, e) for (b, e) in outside)...))
    end
    ordered = sort!(collect(groups); by = g -> -length(last(g)))
    return [GiacExpr(k) => GiacExpr(_add(v...)) for (k, v) in ordered]
end
