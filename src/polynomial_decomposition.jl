# Functional decomposition of a polynomial, f = g₁(g₂(⋯ gₖ(x))), by the method of Kozen and
# Landau. A right factor h of degree r (r dividing n = deg f, s = n/r) is fixed by the leading
# coefficients of f: with h monic, hˢ agrees with f/lc(f) in its r top coefficients, so the
# reversed h is the sᵗʰ root of the reversed f as a power series, to order r. f is a composition
# with h when its expansion in powers of h has constant coefficients.

"""
    polynomial_decomposition(f, x)

The polynomials `[g₁, g₂, …, gₖ]` in `x`, each of degree at least 2, with f = g₁(g₂(⋯ gₖ(x))),
for a polynomial `f` in `x` with rational coefficients; `[f]` when it is not a composition. The
inner polynomials are monic and have no constant term.

```julia
@giac_var x
Giac.polynomial_decomposition(x^4 + 2x^2 + 3, x)   # [x^2 + 2x + 3, x^2]
```
"""
function polynomial_decomposition(f::GiacExpr, x::GiacExpr)
    c = _rational_coefficients(f, x)
    return [_polynomial_in(p, x) for p in _decompose_coefficients(c)]
end

# the coefficients of `f` in `x`, constant term first, as rationals
function _rational_coefficients(f::GiacExpr, x::GiacExpr)
    l = invoke_cmd(:symb2poly, f, x)
    cs = [to_julia(l[i]) for i in length(l):-1:1]
    all(c -> c isa Union{Integer, Rational}, cs) ||
        throw(ArgumentError("polynomial_decomposition: $f is not a polynomial in $x with rational coefficients"))
    return Rational{BigInt}.(cs)
end

# the polynomial with coefficients `c` (constant term first), written from the highest power down
_polynomial_in(c::Vector{Rational{BigInt}}, x::GiacExpr) =
    sum((GiacExpr(NumNode(_canonical(c[k]))) * x^(k - 1) for k in length(c):-1:1 if !iszero(c[k])); init = zero(GiacExpr))

function _decompose_coefficients(f::Vector{Rational{BigInt}})
    n = length(f) - 1
    for r in 2:(n ÷ 2)
        n % r == 0 || continue
        h = _right_composition_factor(f, r)
        g = _left_composition_factor(f, h)
        g === nothing || return [_decompose_coefficients(g); _decompose_coefficients(h)]
    end
    return [f]
end

# the monic h of degree r with h(0) = 0 whose (n/r)th power agrees with f/lc(f) at the top
function _right_composition_factor(f, r)
    n = length(f) - 1
    α = 1 // big(n ÷ r)
    Q = [f[n + 1 - k] / f[n + 1] for k in 0:(r - 1)]       # the reversed f/lc(f), to order r
    # P = Q^α as a power series: k Pₖ = Σⱼ ((α+1) j − k) Qⱼ Pₖ₋ⱼ
    P = zeros(Rational{BigInt}, r)
    P[1] = 1
    for k in 1:(r - 1)
        P[k + 1] = sum(((α + 1) * j - k) * Q[j + 1] * P[k - j + 1] for j in 1:k) / k
    end
    h = zeros(Rational{BigInt}, r + 1)
    for k in 0:(r - 1)
        h[r + 1 - k] = P[k + 1]
    end
    h[1] = 0
    return h
end

# g with f = g(h), when every coefficient of f in powers of h is a constant; nothing otherwise
function _left_composition_factor(f, h)
    g = Rational{BigInt}[]
    q = copy(f)
    while !(length(q) == 1 && iszero(q[1]))
        q, rem = _divrem_monic(q, h)
        all(iszero, rem[2:end]) || return nothing
        push!(g, rem[1])
    end
    return g
end

# quotient and remainder of a by the monic h, coefficients constant term first
function _divrem_monic(a, h)
    r = length(h) - 1
    length(a) <= r && return Rational{BigInt}[0], [a; zeros(Rational{BigInt}, r - length(a))]
    a = copy(a)
    q = zeros(Rational{BigInt}, length(a) - r)
    for k in length(q):-1:1
        c = a[k + r]
        q[k] = c
        for j in 0:r
            a[k + j] -= c * h[j + 1]
        end
    end
    return q, a[1:r]
end
