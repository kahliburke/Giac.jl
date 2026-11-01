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
