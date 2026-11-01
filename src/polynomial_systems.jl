# Results of solving polynomial systems exactly. The solving is done by msolve through
# AlgebraicSolving.jl (GiacAlgebraicSolvingExt), which adds methods for Giac expressions to its
# `real_solutions`, `rational_solutions`, `rational_parametrization`, `groebner_basis`,
# `eliminate`, `normal_form` and `dimension`.

"""
    RealSolution

A real solution of a polynomial system, from `real_solutions`. Each variable lies in a closed
interval with rational ends, `lower[i] ≤ vars[i] ≤ upper[i]`, guaranteed to contain it. `s[x]` is
the midpoint of `x`'s interval, a `Rational{BigInt}` (`Float64(s[x])` for a decimal). Iterating
gives `x => s[x]` pairs, so `substitute(e, s...)` evaluates an expression there.
"""
struct RealSolution
    vars::Vector{GiacExpr}
    lower::Vector{Rational{BigInt}}
    upper::Vector{Rational{BigInt}}
end

function _solution_index(s::RealSolution, x)
    name = string(x)
    i = findfirst(v -> string(v) == name, s.vars)
    i === nothing && throw(KeyError(x))
    return i
end

Base.getindex(s::RealSolution, x) = (i = _solution_index(s, x); (s.lower[i] + s.upper[i]) / 2)
Base.keys(s::RealSolution) = s.vars
Base.length(s::RealSolution) = length(s.vars)
Base.iterate(s::RealSolution, i = 1) = i > length(s.vars) ? nothing : (s.vars[i] => s[s.vars[i]], i + 1)

function Base.show(io::IO, s::RealSolution)
    print(io, "(", join((string(v, " ≈ ", Float64(s[v])) for v in s.vars), ", "), ")")
end

"""
    Parametrization

The solutions of a polynomial system with finitely many, from `rational_parametrization`: they
are the values at the roots of one polynomial in a new variable `t`.

- `t`: the new variable, equal to `form`, a linear combination of the system's variables;
- `eliminant`: the polynomial in `t` whose roots give the solutions, one each;
- `values`: each variable as a rational function of `t`, `x => p(t) / denominator`.
"""
struct Parametrization
    t::GiacExpr
    form::GiacExpr
    eliminant::GiacExpr
    denominator::GiacExpr
    values::Vector{Pair{GiacExpr, GiacExpr}}
    degree::Int
end

function Base.show(io::IO, ::MIME"text/plain", p::Parametrization)
    println(io, "Rational parametrization by ", p.t, " = ", p.form, ", ", p.degree,
            p.degree == 1 ? " solution" : " solutions")
    println(io, "  ", p.eliminant, " = 0")
    for (i, (x, v)) in enumerate(p.values)
        print(io, "  ", x, " = ", v)
        i < length(p.values) && println(io)
    end
end
Base.show(io::IO, p::Parametrization) = print(io, "Parametrization(", p.t, ", ", p.degree, " solutions)")

"""
    RootCase

One case of a [`RootClassification`](@ref): where the classification's polynomials have the
`signs` given (`-1` or `1` each), the system has `count` real solutions; `witness` is a point of
the parameters in that case, as `p => value` pairs.
"""
struct RootCase
    signs::Vector{Int}
    count::Int
    witness::Vector{Pair{GiacExpr, GiacExpr}}
end

"""
    RootClassification

The number of real solutions of a system with parameters, from `real_root_classification`: in
each of the `cases`, the signs of the polynomials `polys` in the parameters `params` fix the
number of real solutions. It holds where none of the polynomials `excluded` vanishes.
"""
struct RootClassification
    params::Vector{GiacExpr}
    polys::Vector{GiacExpr}
    cases::Vector{RootCase}
    excluded::Vector{GiacExpr}
end

_solutions(n) = string(n, n == 1 ? " real solution" : " real solutions")

# a witness value: exact when an integer, else a few digits
function _witness_text(w)
    v = to_julia(last(w))
    text = v isa Integer || (v isa Rational && isone(denominator(v))) ? string(v) : "≈ " * string(round(Float64(v); sigdigits = 4))
    return string(first(w), startswith(text, "≈") ? " " : " = ", text)
end

function Base.show(io::IO, ::MIME"text/plain", r::RootClassification)
    print(io, "Real root classification in ", join(string.(r.params), ", "))
    isempty(r.excluded) || print(io, ", where ", join((string(e, " ≠ 0") for e in r.excluded), ", "))
    if length(r.cases) <= 6
        for c in r.cases
            conditions = join((string(p, s < 0 ? " < 0" : " > 0") for (p, s) in zip(r.polys, c.signs)), ", ")
            print(io, "\n  ", isempty(conditions) ? "always" : conditions, ": ", _solutions(c.count),
                  "  (e.g. ", join(_witness_text.(c.witness), ", "), ")")
        end
    else
        # many sign cases: how many give each number of solutions
        print(io, "\n  ", length(r.cases), " sign cases of ", length(r.polys), " polynomials")
        for n in sort!(unique(c.count for c in r.cases))
            group = [c for c in r.cases if c.count == n]
            print(io, "\n  ", _solutions(n), ": ", length(group), length(group) == 1 ? " case" : " cases",
                  "  (e.g. ", join(_witness_text.(first(group).witness), ", "), ")")
        end
    end
end
Base.show(io::IO, r::RootClassification) =
    print(io, "RootClassification(", join(string.(r.params), ", "), "; ", length(r.cases), " cases)")
