# Domains to factor over, given to `factor` in place of an extension: `factor(p, ℝ)`

"""
    Giac.Domain

A domain `factor(p, K)` factors over, given in place of an extension:

- `ℚ`: into polynomials irreducible over the rationals, as `factor(p)` does
- `ℤ`: the same, with the integer content as a product of primes
- `ℝ`: into linear factors and quadratics with no real roots
- `ℂ`: into linear factors
- `Giac.GF(p)`: modulo the prime `p`; with Nemo loaded, its `ZZ`, `QQ` and prime fields `GF(p)` too

`ℝ` and `ℂ` take polynomials in one variable with rational coefficients, and find roots in square
roots: GIAC factors with square roots joined to the rationals, and each quadratic left is split by
its roots. A factor of degree three or more left by that is an `ArgumentError`. A factorization over
an extension is asked for with the extension, as in `factor(x^4 + 1, sqrt(2))`.

```julia
julia> factor(x^4 - 2, ℝ)
(x-2^(1/4))*(x+2^(1/4))*(x^2+sqrt(2))

julia> factor(6x^2 - 12, ℤ)
(2*3)*(x^2-2)
```
"""
abstract type Domain end
struct Integers <: Domain end
struct Rationals <: Domain end
struct Reals <: Domain end
struct Complexes <: Domain end
struct PrimeField <: Domain
    p::BigInt
end

"The integers, to factor over: `factor(p, ℤ)` writes the content as a product of primes. See [`Giac.Domain`](@ref)."
const ℤ = Integers()
"The rationals, to factor over: `factor(p, ℚ)` is `factor(p)`. See [`Giac.Domain`](@ref)."
const ℚ = Rationals()
"The reals, to factor over: `factor(p, ℝ)` gives linear factors and quadratics with no real roots. See [`Giac.Domain`](@ref)."
const ℝ = Reals()
"The complex numbers, to factor over: `factor(p, ℂ)` gives linear factors. See [`Giac.Domain`](@ref)."
const ℂ = Complexes()

"""
    Giac.GF(p)

The integers modulo the prime `p`, to factor over: `factor(x^4 + 1, Giac.GF(5))`.
"""
function GF(p::Integer)
    (p > 1 && string(_isprime_cmd(p)) == "true") || throw(ArgumentError("GF(p) needs a prime p, not $p"))
    return PrimeField(big(p))
end

Base.show(io::IO, ::Integers) = print(io, "ℤ")
Base.show(io::IO, ::Rationals) = print(io, "ℚ")
Base.show(io::IO, ::Reals) = print(io, "ℝ")
Base.show(io::IO, ::Complexes) = print(io, "ℂ")
Base.show(io::IO, K::PrimeField) = print(io, "GF(", K.p, ")")

# the domain a value names, or nothing; the Nemo extension adds Nemo's rings
_factor_domain(K) = nothing
_factor_domain(K::Domain) = K

_factor_over(p::GiacExpr, ::Rationals) = _factor_cmd(p)

function _factor_over(p::GiacExpr, ::Integers)
    _is_polynomial(_node(p)) || throw(ArgumentError("factoring over ℤ: $p is not a polynomial with rational coefficients"))
    F = _factor_cmd(p)
    n = _node(F)
    c = n isa MulNode ? n.coef : n isa NumNode ? n.v : 1
    (c isa Integer || c isa Rational && isone(denominator(c))) ||
        throw(ArgumentError("factoring over ℤ: $p has coefficients that are not integers"))
    c = Integer(c)
    abs(c) <= 1 && return F
    primitive = n isa MulNode ? GiacExpr(_mul((_pow(b, e) for (b, e) in n.factors)...)) : one(GiacExpr)
    return sign(c) * invoke_cmd(:ifactor, abs(c)) * primitive
end

# p with its rational coefficients taken modulo K.p: numerator over a denominator prime to K.p
function _factor_over(p::GiacExpr, K::PrimeField)
    _is_polynomial(_node(p)) || throw(ArgumentError("factoring over $K: $p is not a polynomial with rational coefficients"))
    q = invoke_cmd(:normal, p)
    d = Integer(to_julia(invoke_cmd(:denom, q)))
    gcd(d, K.p) == 1 || throw(ArgumentError("factoring over $K: $p has a denominator divisible by $(K.p)"))
    F = invoke_cmd(:factor, invoke_cmd(:numer, q) * giac_eval("1 % $(K.p)"))
    return isone(d) ? F : giac_eval("$(invmod(d, K.p)) % $(K.p)") * F
end

function _factor_over(p::GiacExpr, K::Union{Reals, Complexes})
    names = _free_names(p)
    (_is_polynomial(_node(p)) && length(names) <= 1) ||
        throw(ArgumentError("factoring over $K: $p is not a polynomial in one variable with rational coefficients"))
    isempty(names) && return p
    x = giac_identifier(only(names))
    F = _wrapper_overrides() ? with_settings(() -> _invoke_cmd_direct(:factor, (p,)), :withsqrt => true) :
                               _invoke_cmd_direct(:factor, (p,))
    n = _node(F)
    out = n isa MulNode ? GiacExpr(NumNode(n.coef)) : one(GiacExpr)
    for (b, e) in (n isa MulNode ? n.factors : [n => _ONE])
        out *= _split_over(GiacExpr(b), x, K)^GiacExpr(e)
    end
    return out
end

# b, irreducible over the rationals with square roots joined, split by its roots where they lie in K:
# a quadratic a(x - r)(x - r̄) with r = (-β ± √D)/2a, over ℝ where D > 0
function _split_over(b::GiacExpr, x::GiacExpr, K::Union{Reals, Complexes})
    d = Int(to_julia(invoke_cmd(:degree, b, x)))
    d <= 1 && return b
    d == 2 || throw(ArgumentError("factoring over $K: $b, of degree $d, is left by factoring with square roots"))
    a, β, c = (invoke_cmd(:coeff, b, x, k) for k in (2, 1, 0))
    D = invoke_cmd(:simplify, β^2 - 4a * c)
    s = _sign_of_nonzero(D)
    K isa Reals && s < 0 && return b
    re = invoke_cmd(:simplify, -β / (2a))
    w = s > 0 ? invoke_cmd(:simplify, sqrt(D) / (2a)) : convert(GiacExpr, im) * invoke_cmd(:simplify, sqrt(-D) / (2a))
    return a * (x - (re + w)) * (x - (re - w))
end

# the sign of a real constant known not to be 0, as the discriminant of an irreducible quadratic,
# from its value to 30 digits
function _sign_of_nonzero(D::GiacExpr)
    v = _approx_value(invoke_cmd(:evalf, D, 30))
    (v isa Real && !iszero(v)) || throw(ArgumentError("the sign of $D is not decided"))
    return sign(v)
end
