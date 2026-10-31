# Tests for slots, segments and `where` conditions. Those about signs ask Giac, so they use what
# was assumed about the variables (`@giac_var k::positive`).

"""
    is_free(e, x)
    is_free(x)

Whether `e` does not involve the variable `x` (or any of the variables in an array `x`). With one
argument, the test as a function, for a slot or segment: `~~consts::is_free(x)`.
"""
function is_free(e, x)
    names = Set(string(v) for v in (x isa AbstractArray ? vec(x) : (x,)) for v in _symbols(v))
    return !any(v -> string(v) in names, _symbols(e))
end
is_free(x) = e -> is_free(e, x)
_symbols(e::GiacExpr) = Giac.free_symbols(e)
_symbols(e::AbstractArray) = (s for x in e for s in _symbols(x))
_symbols(::Number) = ()

"""
    depends_on(e, x)
    depends_on(x)

Whether `e` involves the variable `x` (or one of an array of them); the opposite of
[`is_free`](@ref). With one argument, the test as a function.
"""
depends_on(e, x) = !is_free(e, x)
depends_on(x) = e -> depends_on(e, x)

"""
    of_degree(n, x)

A test for a term of total degree `n` in the variables `x` (a variable or an array of them): the
sum of the powers of the variables among its factors. A term with a factor that depends on `x`
other than as a power (`sin(x1)`) has no degree and does not pass. For a segment:
`+(~~linear::of_degree(1, x)) + ~~rest`.
"""
of_degree(n::Integer, x) = e -> _term_degree(e, x) == n

function _term_degree(t, x)
    t isa GiacExpr || return 0
    names = Set(string(v) for v in (x isa AbstractArray ? vec(x) : (x,)))
    d = 0
    for f in Giac._product_factors(t)
        any(s -> string(s) in names, Giac.free_symbols(f)) || continue
        base, n = something(_power_view(f), (f, convert(GiacExpr, 1)))
        string(base) in names && Giac.is_integer(n) || return nothing
        d += Giac.to_julia(n)
    end
    return d
end

_number(e) = e isa Real ? e : e isa GiacExpr && _is_number(e) ? Giac.to_julia(e) : nothing
_sign(e::GiacExpr) = _key(Giac.invoke_cmd(:sign, e))

"""
    is_positive(e)
    is_negative(e)
    is_nonnegative(e)
    is_real(e)

Whether Giac can show `e` is positive, negative, nonnegative or real, using what is assumed
about its variables: with `@giac_var k::positive`, `is_positive(k + 1)` is `true`. Without enough
known they are `false`, so a rule conditioned on them does not apply.
"""
is_positive(e) = (n = _number(e)) === nothing ? _sign(e) == "1" : n > 0
"See [`is_positive`](@ref)."
is_negative(e) = (n = _number(e)) === nothing ? _sign(e) == "-1" : n < 0
"See [`is_positive`](@ref)."
is_nonnegative(e) = (n = _number(e)) === nothing ? _key(Giac.invoke_cmd(:abs, e)) == _key(e) : n >= 0
"See [`is_positive`](@ref)."
is_real(e) = e isa Real || e isa GiacExpr && (_number(e) !== nothing || _key(Giac.invoke_cmd(:im, e)) == "0")

"""
    is_integer_valued(e)

Whether `e` is an integer for every value of its variables: an integer, a variable declared
`::integer`, or a sum, product or nonnegative integer power of such.
"""
function is_integer_valued(e)
    e isa Integer && return true
    e isa GiacExpr || return false
    is_integer(e) && return true
    Giac.is_identifier(e) && return Giac._declared(e) === :integer
    h = _head(e)
    h in ("+", "*", "-") && return all(is_integer_valued, arguments(e))
    h == "^" && return is_integer_valued(arguments(e)[1]) && is_integer(arguments(e)[2]) && Giac.to_julia(arguments(e)[2]) >= 0
    return false
end

"""
    in_interval(lo, hi)
    in_interval(e, lo, hi)

A test for an expression known to lie in `[lo, hi]`: a number in it, a variable declared in a
narrower interval (`@giac_var u::(0, 1)`), or anything Giac can place there from the
assumptions. For a slot: `asin(sin(~x::in_interval(-π/2, π/2))) => ~x`.
"""
struct InInterval
    lo::GiacExpr
    hi::GiacExpr
end
in_interval(lo, hi) = InInterval(convert(GiacExpr, lo), convert(GiacExpr, hi))
in_interval(e, lo, hi) = in_interval(lo, hi)(e)
function (p::InInterval)(e)
    lo, hi = _float(p.lo), _float(p.hi)
    n = _number(e)
    n === nothing || return lo <= n <= hi
    if e isa GiacExpr && Giac.is_identifier(e)
        tag = Giac._declared(e)
        tag isa Tuple && return lo <= tag[1] && tag[2] <= hi
    end
    return e isa GiacExpr && is_nonnegative(e - p.lo) && is_nonnegative(p.hi - e)
end
_float(e::GiacExpr) = Float64(Giac.to_julia(Giac.Commands.evalf(e)))
