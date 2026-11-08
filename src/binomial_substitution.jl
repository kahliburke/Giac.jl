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

# f as a sum of binomial terms, at least one with a fractional power; or nothing
function _binomial_terms(f::Node, x::SymNode)
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
    return terms
end

# ∫ c xᵐ (a + b xⁿ)ᵖ dx by Chebyshev's substitutions, or nothing
function _chebyshev(c::Node, m::Rational, binomial, p::Rational, x::GiacExpr)
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
        back = (A + B * x^q(n))^q(1 // s)
    elseif isinteger(k + p)
        return _chebyshev(c, m + n * p, (b, a, -n), p, x)
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
    for answer in (total, try invoke_cmd(:simplify, total) catch e; _stopped(e) && rethrow(); total end)
        best !== nothing && _size(_node(answer)) >= _size(_node(best)) && continue
        _is_antiderivative(answer, f, x) && (best = answer)
    end
    return best
end

# the methods of the integration search, in the order they are tried
const _INDEFINITE_METHODS = Function[_with_inverse_hyperbolics, _inverse_substitution, _binomial_substitution]
