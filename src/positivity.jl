# The sign of an expression under an ordering of its variables, read off coefficient signs after
# writing each ordered variable as the one before it plus a positive gap. The exact check is done
# in Nemo (GiacNemoExt); this file has the result types and their display.

"""
    SignReport

The sign of one polynomial (a numerator or a denominator) after the gap substitution:
- `sign`: `1` or `-1` when every coefficient has that sign, `0` when not decided;
- `strict`: whether the sign is strict (all variables strictly positive, polynomial not zero);
- `polynomial`: the polynomial in the gap and positive variables;
- `against`: its terms of the minority sign, the ones that stop a certificate;
- `free`: variables whose sign is not known, which stop a certificate when they appear.
"""
struct SignReport
    sign::Int
    strict::Bool
    polynomial::GiacExpr
    against::Vector{GiacExpr}
    free::Vector{GiacExpr}
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
        print(io, "  ", label, ": ", _sign_word(r.sign, r.strict))
        isempty(r.free) || print(io, "; unknown sign: ", join(_gap_text.(r.free), ", "))
        if !isempty(r.against)
            n = length(r.against)
            print(io, "; ", n, n == 1 ? " term" : " terms", " against: ", join(_gap_text.(first(r.against, 6)), ", "), n > 6 ? ", …" : "")
            r.sign == 0 && isempty(r.free) && print(io, " (a sum-of-squares certificate may still exist)")
        end
        println(io)
    end
    isempty(p.gaps) || print(io, "  gaps: ", join((_gap_text(first(g)) * " = " * _gap_text(last(g)) for g in p.gaps), ", "))
end
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
