# ---------------------------------------------------------------------------
# GiacAlgebraicSolvingExt: polynomial systems solved exactly by msolve.
#
# Loaded with AlgebraicSolving (and Nemo, which it builds on). Adds methods for Giac expressions
# to AlgebraicSolving's functions: a system is a vector of polynomials (each `= 0`) or equations
# `a ~ b`, with rational coefficients, in the variables given or else all of its variables in
# the order they were declared. The system crosses to a Nemo polynomial ring by name, and the
# results come back as Giac expressions.
# ---------------------------------------------------------------------------

module GiacAlgebraicSolvingExt

using Giac
using Giac: GiacExpr
using Nemo
import AlgebraicSolving
const AS = AlgebraicSolving

const _System = Union{AbstractVector{<:GiacExpr}, Tuple{GiacExpr, Vararg{GiacExpr}}}
const _Vars = Union{Nothing, GiacExpr, AbstractVector{<:GiacExpr}, Tuple{Vararg{GiacExpr}}}

_vector(vars::GiacExpr) = GiacExpr[vars]
_vector(vars) = GiacExpr[v for v in vars]

_expression(e::GiacExpr) = Giac._is_equation(e) ? Giac.lhs(e) - Giac.rhs(e) : e

_names(e::GiacExpr) = Giac._symbol_names(Giac._node(e))

# all the variables of the expressions, in the order they were declared
function _variables(es)
    names = sort!(collect(union(Set{String}(), (_names(e) for e in es)...)); by = s -> (Giac._declared_order(s), s))
    return GiacExpr[Giac.giac_identifier(n) for n in names]
end

# The system as an ideal of the polynomial ring over QQ in `vars`, in that order. `first`
# variables come before the rest (an elimination's block, or parameters).
function _system(eqs, vars; first = GiacExpr[], ordering = :lex)
    polys, vs, R = _polynomials(eqs, vars; first, ordering)
    return AS.Ideal(polys), vs, R
end

function _polynomials(eqs, vars; first = GiacExpr[], ordering = :lex)
    es = GiacExpr[_expression(e) for e in eqs]
    rest = vars === nothing ? GiacExpr[v for v in _variables(es) if !any(f -> string(f) == string(v), first)] : _vector(vars)
    vs = vcat(first, rest)
    names = string.(vs)
    allunique(names) || throw(ArgumentError("a variable is listed twice: $(join(names, ", "))"))
    isempty(vs) && throw(ArgumentError("the system has no variables"))
    for e in es
        extra = setdiff(_names(e), names)
        isempty(extra) || throw(ArgumentError("$(join(sort!(collect(extra)), ", ")) in $e is not among the variables $(join(names, ", "))"))
    end
    R, _ = polynomial_ring(QQ, Symbol.(names); internal_ordering = ordering)
    return [_polynomial(e, R) for e in es], vs, R
end

function _polynomial(e, R)
    try
        return Giac.to_nemo(e, R)
    catch err
        err isa ErrorException || rethrow()
        reason = replace(err.msg, r"^to_nemo: " => "", "; use its fraction field" => "")
        throw(ArgumentError("$e is not a polynomial with rational coefficients in $(join(symbols(R), ", ")): $reason"))
    end
end

# runs a solver that needs finitely many solutions, with a clearer error when there are not
function _finite(f)
    try
        return f()
    catch err
        (err isa ErrorException && occursin("Dimension of ideal is greater than zero", err.msg)) || rethrow()
        throw(ArgumentError("the system has infinitely many solutions; `dimension` gives the " *
                            "dimension of the solution set, and `eliminate` relations among its variables"))
    end
end

_rational(c::QQFieldElem) = Rational{BigInt}(c)

# a univariate polynomial over QQ as an expression in `t`
function _univariate(p, t::GiacExpr)
    s = Giac._node(t)
    terms = Giac.Node[]
    for k in 0:degree(p)
        c = coeff(p, k)
        iszero(c) && continue
        push!(terms, Giac._mul(Giac.NumNode(_rational(c)), Giac._pow(s, Giac.NumNode(k))))
    end
    return GiacExpr(isempty(terms) ? Giac._ZERO : Giac._add(terms...))
end

_whole(c) = (r = _rational(QQ(c)); isone(denominator(r)) ? numerator(r) : r)

"""
    real_solutions(eqs, vars = nothing; precision = 64)

The real solutions of a polynomial system with finitely many complex solutions, isolated
exactly by msolve: a vector of [`Giac.RealSolution`](@ref)s, each variable in an interval with
rational ends about `2^-precision` wide that is guaranteed to contain it. `eqs` holds
polynomials (each `= 0`) or equations `a ~ b`, with rational coefficients; `vars` the variables,
by default all of them in the order declared. Other keywords go to AlgebraicSolving
(`info_level`, ...).

```julia
using Giac, AlgebraicSolving
@giac_var x y
sols = real_solutions([x^2 + y^2 - 1, y - x^2])
sols[1][x]                          # a rational within 2^-64 of the solution
substitute(x + y, sols[1]...)
```
"""
function AS.real_solutions(eqs::_System, vars::_Vars = nothing; precision::Integer = 64, kw...)
    I, vs, _ = _system(eqs, vars)
    boxes = _finite(() -> AS.real_solutions(I; precision = Int(precision), interval = true, kw...))
    return [Giac.RealSolution(vs, [_rational(min(b...)) for b in box], [_rational(max(b...)) for b in box]) for box in boxes]
end

"""
    rational_solutions(eqs, vars = nothing)

The solutions of a polynomial system that are rational, each a vector of `x => value` pairs
(`substitute(e, sol...)` evaluates an expression there). The system must have finitely many
complex solutions; see [`real_solutions`](@ref) for the arguments.
"""
function AS.rational_solutions(eqs::_System, vars::_Vars = nothing; kw...)
    I, vs, _ = _system(eqs, vars)
    sols = _finite(() -> AS.rational_solutions(I; kw...))
    return [[v => Giac.to_giac(c) for (v, c) in zip(vs, s)] for s in sols]
end

"""
    rational_parametrization(eqs, vars = nothing; t = :T)

All the complex solutions of a polynomial system with finitely many, exactly: a
[`Giac.Parametrization`](@ref), each variable a rational function of a new variable `t` at the
roots of one polynomial. `t` is a variable of the system when that one separates the solutions,
and otherwise a linear combination of them.

```julia
@giac_var x y
p = rational_parametrization([x^2 + y^2 - 1, y - x^2])
p.form                               # -x-y: neither variable separates the 4 solutions
p.eliminant                          # -1+6*T-2*T^3+T^4
```
"""
function AS.rational_parametrization(eqs::_System, vars::_Vars = nothing; t = :T, kw...)
    I, vs, _ = _system(eqs, vars)
    rp = _finite(() -> AS.rational_parametrization(I; kw...))
    n = length(vs)
    T = t isa GiacExpr ? t : Giac.giac_identifier(string(t))
    any(v -> string(v) == string(T), vs) && throw(ArgumentError("the new variable $T is a variable of the system; pass another as `t`"))
    den = _univariate(rp.denom, T)
    value(p) = _univariate(p, T) / den
    if isempty(rp.cfs_lf)
        # the last variable separates the solutions and is the parameter itself
        form = vs[end]
        values = Pair{GiacExpr, GiacExpr}[vs[i] => value(rp.param[i]) for i in 1:(n - 1)]
        push!(values, vs[end] => T)
    else
        # a new variable with c₁x₁ + … + cₙxₙ + cₜt = 0
        c = rp.cfs_lf
        length(c) == n + 1 || error("rational_parametrization: unexpected linear form of length $(length(c)) for $n variables")
        form = sum(convert(GiacExpr, _whole(-c[i] // c[n + 1])) * vs[i] for i in 1:n)
        values = Pair{GiacExpr, GiacExpr}[vs[i] => value(rp.param[i]) for i in 1:n]
    end
    return Giac.Parametrization(T, form, _univariate(rp.elim, T), den, values, degree(rp.elim))
end

"""
    groebner_basis(eqs, vars = nothing)

The reduced Gröbner basis of the polynomials `eqs` for the degree reverse lexicographic order
of `vars` (by default all their variables in the order declared), computed by msolve's F4.
"""
function AS.groebner_basis(eqs::_System, vars::_Vars = nothing; kw...)
    I, _, _ = _system(eqs, vars)
    return GiacExpr[Giac.to_giac(g) for g in AS.groebner_basis(I; kw...)]
end

"""
    eliminate(eqs, xs; vars = nothing)

Generators of the polynomials in the remaining variables that `eqs` imply, with the variables
`xs` eliminated: the relations among `vars` (by default the other variables of `eqs`) that hold
on every solution. `xs` is a variable or a vector of them.

```julia
@giac_var t x y
eliminate([x - (1 - t^2), y - 2t], t)          # the curve, implicitly: [y² + 4x − 4]
```
"""
function AS.eliminate(eqs::_System, xs::Union{GiacExpr, AbstractVector{<:GiacExpr}, Tuple{Vararg{GiacExpr}}};
                      vars::_Vars = nothing, kw...)
    first = _vector(xs)
    I, _, _ = _system(eqs, vars; first)
    return GiacExpr[Giac.to_giac(g) for g in AS.eliminate(I, length(first); kw...)]
end

"""
    normal_form(e, eqs, vars = nothing)

`e` reduced modulo the polynomials `eqs`: the remainder on division by their Gröbner basis
(degree reverse lexicographic in `vars`). Two polynomials have the same normal form exactly when
their difference lies in the ideal of `eqs`, so it simplifies `e` under the constraints
`eqs = 0`.

```julia
@giac_var c s
normal_form(c^4 + 2c^2*s^2 + s^4, [c^2 + s^2 - 1])      # 1
```
"""
function AS.normal_form(e::GiacExpr, eqs::_System, vars::_Vars = nothing; kw...)
    vs = vars === nothing ? _variables(vcat(GiacExpr[_expression(q) for q in eqs], [e])) : vars
    # msolve reduces only over finite fields: the remainder on dividing by the Gröbner basis, in
    # a ring with the same order
    I, _, R = _system(eqs, vs; ordering = :degrevlex)
    G = AS.groebner_basis(I; kw...)
    return Giac.to_giac(last(divrem(_polynomial(e, R), G)))
end

# real_root_classification came in AlgebraicSolving 0.13; with an older version, which Oscar
# requires, the other methods are defined without it
@static if isdefined(AS, :real_root_classification)

    """
        real_root_classification(eqs, vars = nothing; params, positive = [])

    How the number of real solutions of a system depends on its parameters: a
    [`Giac.RootClassification`](@ref), whose cases each give the signs of some polynomials in the
    parameters and the number of real solutions where they hold, with a point of the parameters as
    an example. With `positive`, only the solutions where each of those expressions is positive are
    counted. `eqs` must have finitely many complex solutions for generic values of the parameters.

    ```julia
    @giac_var x a b
    real_root_classification([x^3 - a*x + b], [x]; params = [a, b])
    # Real root classification in a, b, where 4*a^3-27*b^2 ≠ 0
    #   6*a < 0, 4*a^3-27*b^2 < 0: 1 real solution  (e.g. a = -1, b = 0)
    #   6*a > 0, 4*a^3-27*b^2 < 0: 1 real solution  (e.g. a = 1, b = -2)
    #   6*a > 0, 4*a^3-27*b^2 > 0: 3 real solutions  (e.g. a = 1, b = 0)
    ```
    """
    function AS.real_root_classification(eqs::_System, vars::_Vars = nothing; params, positive = GiacExpr[], kw...)
        ps = _vector(params)
        gs = GiacExpr[_expression(convert(GiacExpr, g)) for g in positive]
        pnames = Set(string.(ps))
        vs = vars === nothing ? GiacExpr[v for v in _variables(vcat(GiacExpr[_expression(e) for e in eqs], gs)) if !(string(v) in pnames)] :
             _vector(vars)
        polys, _, _ = _polynomials(vcat(GiacExpr[e for e in eqs], gs), vs; first = ps)
        # polynomials in the variables over the rational functions of the parameters
        np = length(ps)
        K, _ = polynomial_ring(QQ, Symbol.(string.(ps)))
        F = fraction_field(K)
        S, _ = polynomial_ring(F, Symbol.(string.(vs)))
        function lifted(p)
            parts = Dict{Vector{Int}, typeof(zero(K))}()
            for (c, ev) in zip(coefficients(p), exponent_vectors(p))
                key = ev[(np + 1):end]
                parts[key] = get(parts, key, zero(K)) + c * _monomial(K, ev[1:np])
            end
            ctx = MPolyBuildCtx(S)
            foreach(((ev, c),) -> push_term!(ctx, F(c), ev), parts)
            return finish(ctx)
        end
        lifts = map(lifted, polys)
        neq = length(eqs)
        result = AS.real_root_classification(AS.ParametricIdeal(lifts[1:neq]), lifts[(neq + 1):end]; kw...)
        sets = result isa AS.Intersect ? result.S : [result]
        excluded = GiacExpr[Giac.to_giac(q) for s in sets if s isa AS.BasicSemialgebraicSet for q in s.ineqs]
        enum = only(s for s in sets if s isa AS.SignEnum)
        cases = [Giac.RootCase(collect(Int, sg), n, Pair{GiacExpr, GiacExpr}[p => Giac.to_giac(v) for (p, v) in zip(ps, w)])
                 for (sg, n, w) in zip(enum.signs, enum.counts, enum.witnesses)]
        return Giac.RootClassification(ps, GiacExpr[Giac.to_giac(q) for q in enum.polys], cases, excluded)
    end

    _monomial(R, ev) = (ctx = MPolyBuildCtx(R); push_term!(ctx, one(base_ring(R)), ev); finish(ctx))

end

# Equilibria with their linearization's stability: the Hurwitz determinants of the Jacobian's
# characteristic polynomial join the system as new variables dₖ = Δₖ(x), so msolve isolates
# their values too, and a value's interval that excludes zero gives its sign for certain.
function Giac.Control.equilibria(f::AbstractVector, x; precision::Integer = 64)
    F = GiacExpr[convert(GiacExpr, e) for e in f]
    X = _vector(x)
    n = length(X)
    length(F) == n || throw(ArgumentError("equilibria: one component of f per state variable"))
    s = Giac.giac_identifier("hurwitz_s")
    P = Giac.Commands.invoke_cmd(:charpoly, Giac.GiacMatrix(Giac.jacobian(F, X)), s)
    Δ = Giac.Control.hurwitz_conditions(P, s)
    D = [Giac.giac_identifier("hurwitz_d$k") for k in 1:n]
    for bits in (Int(precision), 4Int(precision))
        sols = AS.real_solutions(vcat(F, D .- Δ), vcat(X, D); precision = bits)
        signs = [[s.lower[n + k] > 0 ? 1 : s.upper[n + k] < 0 ? -1 : 0 for k in 1:n] for s in sols]
        # a determinant too close to zero to sign: more precision once, then undecided
        bits == Int(precision) && any(sg -> 0 in sg, signs) && continue
        return [Giac.Control.Equilibrium(Giac.RealSolution(X, sol.lower[1:n], sol.upper[1:n]), _routh_changes(sg))
                for (sol, sg) in zip(sols, signs)]
    end
end

# the eigenvalues in the right half-plane, from the signs of the Hurwitz determinants (monic
# characteristic polynomial): the sign changes along 1, Δ₁, Δ₂/Δ₁, …; nothing if one is zero
function _routh_changes(signs)
    0 in signs && return nothing
    column = [1; signs[1]; [signs[k] * signs[k - 1] for k in 2:length(signs)]]
    return count(k -> column[k] != column[k + 1], 1:(length(column) - 1))
end

"""
    dimension(eqs, vars = nothing)

The dimension of the set of complex solutions of the polynomials `eqs`: `0` when there are
finitely many, `-1` when there are none.
"""
function AS.dimension(eqs::_System, vars::_Vars = nothing)
    I, _, _ = _system(eqs, vars)
    return AS.dimension(I)
end

end # module
