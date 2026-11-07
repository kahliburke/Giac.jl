# Grading an integral's answer, and integrating one as a unit of a sweep. This file needs only
# Giac.jl: it is what a sweep's task processes load.

module IntegrationGrading

using Giac

const C = Giac.Commands

export leaf_count, grade, optimal_size, integrate_unit

"""
    leaf_count(e)

The size of `e` counted as Mathematica's `LeafCount` counts it, so that it compares with the
suite's optimal sizes: every head and atom is a leaf, a fraction is `Rational[p, q]`, a
difference is a sum with a factor −1, a quotient a power −1, a square root the power 1/2, and
`exp(u)` the power `E^u`.
"""
leaf_count(e::GiacExpr) = _leaves(Giac._node(e))

_leaves(v::Integer) = 1
_leaves(v::Rational) = isinteger(v) ? 1 : 3
_leaves(v::Complex) = 3
_leaves(v::Number) = 1
_leaves(n::Giac.NumNode) = _leaves(n.v)
_leaves(n::Giac.SymNode) = 1
_leaves(n::Giac.GenNode) = 1
function _leaves(n::Giac.CallNode)
    n.f == "exp" && return 2 + sum(_leaves, n.args)                    # Power[E, u]
    n.f == "sqrt" && return 4 + sum(_leaves, n.args)                   # Power[u, Rational[1, 2]]
    1 + sum(_leaves, n.args; init = 0)
end
# a term c·m, as Times[c, m] with Times flattened
function _term_leaves(m::Giac.Node, c::Number)
    isone(c) && return _leaves(m)
    m === Giac._UNIT && return _leaves(c)
    m isa Giac.MulNode ? _leaves(m) + _leaves(c) : 1 + _leaves(c) + _leaves(m)
end
_leaves(n::Giac.AddNode) = (length(n.terms) == 1 ? 0 : 1) + sum(_term_leaves(m, c) for (m, c) in n.terms)
_power_leaves(b, e) = isequal(e, Giac._ONE) ? _leaves(b) : 1 + _leaves(b) + _leaves(e)
function _leaves(n::Giac.MulNode)
    parts = sum(_power_leaves(b, e) for (b, e) in n.factors; init = 0)
    k = length(n.factors) + (isone(n.coef) ? 0 : 1)
    k == 1 && isone(n.coef) ? parts : 1 + parts + (isone(n.coef) ? 0 : _leaves(n.coef))
end

const ELEMENTARY = Set(["exp", "ln", "log", "sqrt", "sin", "cos", "tan", "sec", "csc", "cot",
                        "asin", "acos", "atan", "acot", "asec", "acsc", "sinh", "cosh", "tanh",
                        "sech", "csch", "coth", "asinh", "acosh", "atanh", "acoth", "abs", "sign"])
_functions(n::Giac.CallNode, acc) = (push!(acc, n.f); foreach(a -> _functions(a, acc), n.args); acc)
_functions(n::Giac.AddNode, acc) = (foreach(t -> _functions(t[1], acc), n.terms); acc)
_functions(n::Giac.MulNode, acc) = (foreach(f -> (_functions(f[1], acc); _functions(f[2], acc)), n.factors); acc)
_functions(n::Giac.Node, acc) = acc
special_functions(e::GiacExpr) = sort!(collect(setdiff(_functions(Giac._node(e), Set{String}()), ELEMENTARY)))

_has_i(n::Giac.NumNode) = n.v isa Complex
_has_i(n::Giac.SymNode) = false
_has_i(n::Giac.CallNode) = any(_has_i, n.args)
_has_i(n::Giac.AddNode) = any(t -> _has_i(t[1]) || t[2] isa Complex, n.terms)
_has_i(n::Giac.MulNode) = n.coef isa Complex || any(f -> _has_i(f[1]) || _has_i(f[2]), n.factors)
_has_i(n::Giac.Node) = occursin(r"\bi\b", string(n))
has_complex(e::GiacExpr) = _has_i(Giac._node(e))

# values for the parameters, distinct and away from special points, and the points: exact, so that
# what is evaluated is the exact expression, without the cancellation of floating point
_parameter_value(name) = 1 // 2 + Int(hash(name) % 997) // 664
const POINTS = [37 // 100, 83 // 100, 161 // 100, 229 // 100]

# Whether d/dx answer is the integrand: `true` when |d/dx answer − integrand|, relative to the
# integrand, is small at two of the points where the integrand is finite, to 30 digits, or when
# the difference simplifies to 0; `false` when it is not; `nothing` when there are too few points
# to tell. Two points are enough: an answer and the integrand can take different branches of a
# power or a logarithm on part of the line (sec(x)^(3/2) and 1/sqrt(cos(x))/cos(x) where
# cos x < 0), while a wrong answer differs almost everywhere.
function verify(answer::GiacExpr, integrand::GiacExpr, x::GiacExpr)
    d = C.diff(answer, x) - integrand
    names = setdiff(string.(Giac.to_julia(C.lname(integrand + answer))), [string(x)])
    params = Dict(Giac.giac_identifier(n) => _parameter_value(n) for n in names)
    matched, differed = 0, 0
    for p in POINTS
        values = merge(params, Dict(x => p))
        f = _number(Giac.substitute(integrand, values))
        (f === nothing || !isfinite(f) || abs(f) < 1e-12) && continue
        g = _number(Giac.substitute(d, values))
        g === nothing && continue
        abs(g) <= 1e-8 * max(abs(f), 1.0) ? (matched += 1) : (differed += 1)
    end
    matched >= 2 && return true
    _simplifies_to_zero(d) && return true
    differed > 0 ? false : nothing
end
_simplifies_to_zero(d) = try string(C.simplify(d)) == "0" catch; false end
function _number(e)
    try
        v = Giac.to_julia(C.evalf(e, 30))
        v isa Number ? ComplexF64(v) : nothing
    catch
        nothing
    end
end

"""
    grade(answer, integrand, x, optimal_size, optimal_special, optimal_complex) -> (grade, note)

The suite's grade of `answer`: A, correct and at most twice the optimal size; B, correct and
larger; C, correct but with complex numbers or special functions the optimal answer does not
have (`optimal_special`, `optimal_complex` say whether it has them); F, unevaluated or wrong. `?`
is an answer that could not be checked.
"""
function grade(answer::GiacExpr, integrand::GiacExpr, x::GiacExpr, optimal_size, optimal_special, optimal_complex)
    occursin(r"\bintegrate\(", string(answer)) && return ("F", "unevaluated")
    ok = verify(answer, integrand, x)
    ok === nothing && return ("?", "not checked: too few finite points")
    ok || return ("F", "wrong: the derivative differs from the integrand")
    special = special_functions(answer)
    !isempty(special) && !optimal_special && return ("C", "special functions the optimal has not: " * join(special, ", "))
    has_complex(answer) && !optimal_complex && !has_complex(integrand) && return ("C", "complex numbers the optimal has not")
    (optimal_size === nothing || optimal_size <= 0) && return ("A", "")
    size = leaf_count(answer)
    size <= 2optimal_size ? ("A", "") : ("B", "size $size, over twice the optimal $optimal_size")
end

"""
    optimal_size(optimal_giac, optimal_mma_size)

The size of an integral's optimal antiderivative by `leaf_count`, so that it compares with an
answer's; the suite's own size (Mathematica's) when its GIAC form can't be read.
"""
function optimal_size(optimal_giac, optimal_mma_size)
    (optimal_giac === nothing || isempty(optimal_giac)) && return optimal_mma_size
    try
        leaf_count(giac_eval(optimal_giac))
    catch
        optimal_mma_size
    end
end

# ---------------------------------------------------------------------------------------------
# An integral as a unit of a sweep
# ---------------------------------------------------------------------------------------------

const _WARMED = Ref(false)

# integrals of each kind first, once in a process, so that no timed integral includes compiling
# the code they share
function warm_up!()
    _WARMED[] && return nothing
    x = Giac.giac_identifier("x")
    for f in ("x^2*sin(x)", "1/(x^2+1)", "sqrt(1+x^2)", "exp(x)*cos(x)", "log(x)/x", "(a+b*x)^m")
        g = giac_eval(f)
        Giac.with_timeout(() -> grade(C.integrate(g, x), g, x, 10, false, false), 60)
    end
    _WARMED[] = true
    return nothing
end

"""
    integrate_unit(p; limit = 30, check_limit = 60)

Integrate the integral `p` (its `integrand` and `variable` in GIAC's syntax, and what is known of
its optimal antiderivative) with Giac.jl and grade the answer: the grade, a note on it, the
seconds the integration took, the answer and its size. An integration still running after
`limit` seconds is stopped (F(-1)), and so is a check after `check_limit` (?).
"""
function integrate_unit(p; limit = 30, check_limit = 60)
    warm_up!()
    integrand, x = giac_eval(p.integrand), Giac.giac_identifier(p.variable)
    answer, error = nothing, ""
    seconds = @elapsed try
        answer = Giac.with_timeout(() -> C.integrate(integrand, x), limit)
    catch e
        error = e isa Giac.GiacTimeout ? "" : first(sprint(showerror, e), 300)
    end
    answer === nothing && return isempty(error) ?
        (grade = "F(-1)", note = "no answer within $limit s", seconds, answer = missing, size = missing) :
        (grade = "F(-2)", note = error, seconds, answer = missing, size = missing)
    g, note = try
        Giac.with_timeout(() -> grade(answer, integrand, x, optimal_size(p.optimal_giac, p.optimal_mma_size),
                                      p.optimal_special, p.optimal_complex), check_limit)
    catch e
        ("?", e isa Giac.GiacTimeout ? "not checked within $check_limit s" : "not checked: " * first(sprint(showerror, e), 200))
    end
    size = try leaf_count(answer) catch; missing end
    (grade = g, note, seconds, answer = string(answer), size)
end

end
