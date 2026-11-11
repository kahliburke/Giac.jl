# ∫ xᵐ (a + b xⁿ)ᵖ dx, a binomial differential, with m, n and p rational. By Chebyshev's theorem it
# is elementary exactly when p, (m+1)/n or (m+1)/n + p is an integer, and then a substitution makes
# it the integral of a rational function: x = t^N when p is an integer, N the denominator common to
# m and n, and t = (a + b xⁿ)^(1/s) when (m+1)/n is one, s the denominator of p. The third case is
# the second for xᵐ⁺ⁿᵖ (b + a x⁻ⁿ)ᵖ, the same function for x > 0. A sum of such terms is integrated
# term by term, after the factors other than the binomial are expanded. An answer is kept only if
# its derivative is f.

_rational(n::Node) = n isa NumNode && n.v isa Union{Integer, Rational} ? Rational{BigInt}(n.v) : nothing

_num(q::Rational) = NumNode(isone(denominator(q)) ? Int(numerator(q)) : Rational{Int}(q))
_giac_number(q::Rational) = GiacExpr(_num(q))

_is_symbol(n::Node, x::SymNode) = n isa SymNode && n.name == x.name

# a node k xᵠ, k free of x and q rational: (k, q), or nothing
function _power_of(n::Node, x::SymNode)
    _is_symbol(n, x) && return (_ONE, 1 // 1)
    n isa MulNode || return nothing
    k = Node[NumNode(n.coef)]
    q = nothing
    for (b, e) in n.factors
        if _free_of(b, x) && _free_of(e, x)
            push!(k, _pow(b, e))
        elseif _is_symbol(b, x) && q === nothing && (r = _rational(e)) !== nothing
            q = r
        else
            return nothing
        end
    end
    return q === nothing ? nothing : (_mul(k...), q)
end

# a + b xⁿ, a and b free of x and n rational: (a, b, n), or nothing
function _binomial(n::Node, x::SymNode)
    n isa AddNode && length(n.terms) == 2 || return nothing
    a = b = q = nothing
    for (m, c) in n.terms
        if _free_of(m, x)
            a = _scale(m, c)
        elseif (k = _power_of(m, x)) !== nothing && b === nothing
            b, q = _scale(k[1], c), k[2]
        else
            return nothing
        end
    end
    return a === nothing || b === nothing || iszero(q) ? nothing : (a, b, q)
end

# a term c xᵐ (a + b xⁿ)ᵖ: (c, m, (a, b, n) or nothing, p), or nothing
function _binomial_term(t::Node, x::SymNode)
    factors = t isa MulNode ? t.factors : [t => _ONE]
    coef = Node[NumNode(t isa MulNode ? t.coef : 1)]
    m, p = 0 // 1, 0 // 1
    binomial = nothing
    for (b, e) in factors
        if _free_of(b, x) && _free_of(e, x)
            push!(coef, _pow(b, e))
            continue
        end
        r = _rational(e)
        r === nothing && return nothing
        # √u as u^(1/2), and (uᵏ)ʳ as u^(k r)
        while true
            if b isa CallNode && b.f == "sqrt" && length(b.args) == 1
                b, r = b.args[1], r / 2
            elseif b isa MulNode && isone(b.coef) && length(b.factors) == 1 && (k = _rational(b.factors[1][2])) !== nothing
                b, r = b.factors[1][1], r * k
            else
                break
            end
        end
        if (k = _power_of(b, x)) !== nothing
            push!(coef, _pow(k[1], _num(r)))
            m += k[2] * r
        elseif (B = _binomial(b, x)) !== nothing
            if binomial === nothing
                binomial = B
            elseif !all(isequal.(B, binomial))
                return nothing
            end
            p += r
        else
            return nothing
        end
    end
    return (_mul(coef...), m, binomial, p)
end

# whether `n` is made of x and expressions free of it by sums, products and rational powers
_algebraic_in(n::Node, x::SymNode) = _free_of(n, x) || _is_symbol(n, x) ||
    (n isa AddNode && all(((m, c),) -> _algebraic_in(m, x), n.terms)) ||
    (n isa MulNode && all(((b, e),) -> _algebraic_in(b, x) && _rational(e) !== nothing, n.factors)) ||
    (n isa CallNode && n.f == "sqrt" && length(n.args) == 1 && _algebraic_in(n.args[1], x))

# whether one of Chebyshev's cases holds for xᵐ (a + b xⁿ)ᵖ
_chebyshev_case(m, binomial, p) = binomial === nothing || isinteger(p) || isinteger((m + 1) / binomial[3]) ||
                                  isinteger((m + 1) / binomial[3] + p)

# f as a sum of binomial terms, each in one of Chebyshev's cases and at least one with a fractional
# power; or nothing
function _binomial_terms(f::Node, x::SymNode)
    _algebraic_in(f, x) || return nothing
    one_term = _binomial_term(f, x)
    terms = if one_term !== nothing
        [one_term]
    else
        e = _expand(f)
        parts = e isa AddNode ? [_scale(m, c) for (m, c) in e.terms] : [e]
        ts = [_binomial_term(t, x) for t in parts]
        any(isnothing, ts) && return nothing
        ts
    end
    any(((c, m, B, p),) -> !isinteger(m) || !isinteger(p) || (B !== nothing && !isinteger(B[3])), terms) || return nothing
    all(((c, m, B, p),) -> _chebyshev_case(m, B, p), terms) || return nothing
    return terms
end

# ∫ c xᵐ (a + b xⁿ)ᵖ dx by Chebyshev's substitutions, or nothing. `flipped` is the third case's
# call, for xᵐ (a + b xⁿ)ᵖ written from the original x^m' (b + a x⁻ⁿ)ᵖ: its t = (a + b xⁿ)^(1/s) is
# written back as (b + a x⁻ⁿ)^(1/s) xⁿ⸍ˢ, which equals it for x > 0 and, unlike it, gives an
# antiderivative on both sides of 0 (x and the substitution change sign together)
function _chebyshev(c::Node, m::Rational, binomial, p::Rational, x::GiacExpr; flipped::Bool = false)
    C = GiacExpr(c)
    if binomial === nothing
        return m == -1 ? C * log(x) : C * x^_giac_number(m + 1) / _giac_number(m + 1)
    end
    a, b, n = binomial
    A, B = GiacExpr(a), GiacExpr(b)
    k = (m + 1) / n
    t = _fresh_symbol(_mul(c, a, b, _node(x)), "t")
    q = _giac_number
    if isinteger(p)
        N = lcm(denominator(m), denominator(n))
        N == 1 && return _integrate_or_nothing(C * x^q(m) * (A + B * x^q(n))^q(p), x)
        g = C * q(N // 1) * t^q(N * m + N - 1) * (A + B * t^q(N * n))^q(p)
        back = x^q(1 // N)
    elseif isinteger(k)
        s = denominator(p)
        g = C * q(s // 1) / (q(n) * B) * ((t^q(s // 1) - A) / B)^q(k - 1) * t^q(s * p + s - 1)
        back = flipped ? (B + A * x^q(-n))^q(1 // s) * x^q(n / s) : (A + B * x^q(n))^q(1 // s)
    elseif isinteger(k + p)
        return _chebyshev(c, m + n * p, (b, a, -n), p, x; flipped = true)
    else
        return nothing
    end
    r = _integrate_or_nothing(g, t)
    return r === nothing ? nothing : substitute(r, t => back)
end

"""
    _binomial_substitution(f, x) -> antiderivative or nothing

`∫ f dx` for a sum of binomial differentials c xᵐ (a + b xⁿ)ᵖ, by Chebyshev's substitutions.
"""
function _binomial_substitution(f::GiacExpr, x::GiacExpr)
    xn = _node(x)
    xn isa SymNode || return nothing
    terms = _binomial_terms(_node(f), xn)
    terms === nothing && return nothing
    total = zero(GiacExpr)
    for (c, m, binomial, p) in terms
        g = _chebyshev(c, m, binomial, p, x)
        g === nothing && return nothing
        total += g
    end
    best = nothing
    for answer in (total, _simplified_within(total, 2.0))
        best !== nothing && _size(_node(answer)) >= _size(_node(best)) && continue
        _is_antiderivative(answer, f, x) && (best = answer)
    end
    return best
end

# e simplified, or e where GIAC fails or takes longer than `seconds` (the simplify of nested roots
# can run on for minutes)
function _simplified_within(e::GiacExpr, seconds::Real)
    try
        return with_timeout(() -> invoke_cmd(:simplify, e), seconds)
    catch err
        (_share_spent(err) || !_stopped(err)) && return e
        rethrow()
    end
end

# ∫ f dx by u = x + σ/x (σ = ±1), where f = (1 - σ/x²) h with h² unchanged by x → σ/x, so that h is
# ±g(u): ∫ (x²-1)/((x²+1)√(x⁴+1)) dx = ∫ du/(u√(u²-2)). h² is rational in x, and is written in u by
# eliminating x from w = h² and x² - ux + σ = 0 with a resultant; g is its square root. h/g(u) is ±1,
# constant on each side of 0 where h has no zero or pole: with s₊ and s₋ its values there, F is
# s₊ G(u) where they agree and s₊ sign(x) G(u) where they do not. The answer is kept only if its
# derivative is f.
function _reciprocal_substitution(f::GiacExpr, x::GiacExpr)
    xn = _node(x)
    (xn isa SymNode && _algebraic_in(_node(f), xn)) || return nothing
    for σ in (1, -1)
        u, w = _fresh_symbol(_node(f), "u"), _fresh_symbol(_node(f), "w")
        h = invoke_cmd(:normal, f / (1 - σ / x^2))
        H = invoke_cmd(:normal, h^2)
        _is_rational_in(_node(H), xn.name) || continue
        _is_zero_expr(invoke_cmd(:normal, H - substitute(H, x => σ / x))) || continue
        R = invoke_cmd(:factor, invoke_cmd(:resultant, w * invoke_cmd(:denom, H) - invoke_cmd(:numer, H), x^2 - u * x + σ, x))
        g2 = _linear_root(R, w)
        (g2 === nothing || _depends_on(g2, x)) && continue
        g = _root_of_factored(invoke_cmd(:factor, g2))
        gx = substitute(g, u => x + σ / x)
        s₊, s₋ = _keeping_epsilon(() -> (_side_sign(h, gx, x, 1), _side_sign(h, gx, x, -1)))
        (s₊ === nothing || s₋ === nothing) && continue
        G = _integrate_or_nothing(g, u)
        G === nothing && continue
        F = s₊ * substitute(G, u => x + σ / x) * (s₊ == s₋ ? one(GiacExpr) : invoke_cmd(:sign, x))
        for A in (_simplified_within(F, 2.0), F)
            _is_antiderivative(A, f, x) && return A
        end
    end
    return nothing
end

# h/g, ±1, on the side of 0 of the sign `side`: its value at a generic point there where both have
# values, or nothing where it is not ±1 there
function _side_sign(h::GiacExpr, g::GiacExpr, x::GiacExpr, side::Int)
    for k in 1:_CHECK_POINTS
        point = Dict{GiacExpr, Rational{Int}}(x => side * abs(_generic_point(k, 1)[1]))
        hv, gv = _value_at(h, point), _value_at(g, point)
        (hv === nothing || gv === nothing || iszero(gv)) && continue
        r = hv / gv
        return abs(r - 1) <= _CHECK_TOLERANCE ? 1 : abs(r + 1) <= _CHECK_TOLERANCE ? -1 : nothing
    end
    return nothing
end

# the root in w of the factor of R linear in w, or nothing
function _linear_root(R::GiacExpr, w::GiacExpr)
    n = _node(R)
    for (b, k) in (n isa MulNode ? n.factors : [n => _ONE])
        B = GiacExpr(b)
        d = _approx_value(invoke_cmd(:degree, B, w))
        d == 1 || continue
        c1, c0 = invoke_cmd(:coeff, B, w, 1), invoke_cmd(:coeff, B, w, 0)
        return invoke_cmd(:normal, -c0 / c1)
    end
    return nothing
end

# √e for e a product of powers, the even powers taken out of the root: 1/(u √(u²-2)) for
# 1/(u²(u²-2))
function _root_of_factored(e::GiacExpr)
    n = _node(e)
    n isa MulNode || return sqrt(e)
    outside, inside = Node[], Node[NumNode(n.coef)]
    for (b, k) in n.factors
        if k isa NumNode && k.v isa Integer
            q, r = divrem(k.v, 2)
            iszero(q) || push!(outside, _pow(b, NumNode(q)))
            iszero(r) || push!(inside, _pow(b, NumNode(r)))
        else
            push!(inside, _pow(b, k))
        end
    end
    return GiacExpr(_mul(outside...)) * sqrt(GiacExpr(_mul(inside...)))
end

# the methods of the integration search, in the order they are tried
const _INDEFINITE_METHODS = Function[_with_inverse_hyperbolics, _inverse_substitution, _binomial_substitution,
                                     _pseudo_elliptic, _reciprocal_substitution, _parts_antiderivative]
