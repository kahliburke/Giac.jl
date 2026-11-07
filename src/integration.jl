# GIAC's indefinite integral returns infinity for some exact derivatives of expressions in unknown
# functions, ∫(u′v + uv′) dt among them. Those are found here, a derivative term at a time: a term
# V·u′·uᵖ comes from V·uᵖ⁺¹/(p+1), whose derivative is taken off what is left.

# `Commands.integrate(f, x)`
function _integrate(f::GiacExpr, x::GiacExpr)
    r = invoke_cmd(:integrate, f, x)
    (_is_infinite(r) && occursin("diff(", string(f))) || return r
    g = _antiderivative(f, x)
    return g === nothing ? r : g
end

# `Commands.integrate(f, x, a, b)`: GIAC's, and where it leaves the integral unevaluated, the
# integral as derivatives of a known one in its parameters (parametric_integrals.jl), term by term
# with sines and cosines written as exponentials, or as the mean of f and its image under an
# involution of the interval
function _integrate(f::GiacExpr, x::GiacExpr, a, b)
    r = invoke_cmd(:integrate, f, x, a, b)
    _integral_left(r) || return r
    for method in (_parametric_integral, _exponential_integral, _reflected_integral)
        v = method(f, x, a, b)
        v === nothing || return v
    end
    return r
end

_integral_left(r::GiacExpr) = _is_infinite(r) || _any_node(n -> n isa CallNode && n.f == "integrate", _node(r))

# ∫ₐᵇ f with each sin u and cos u written as exponentials of ±iu, term by term: a Gaussian with
# a cosine is a sum of Gaussians with complex centres
function _exponential_integral(f::GiacExpr, x::GiacExpr, a, b)
    n, xn = _node(f), _node(x)
    xn isa SymNode || return nothing
    _any_node(m -> m isa CallNode && m.f in ("sin", "cos") && !_free_of(m, xn), n) || return nothing
    terms = _sum_terms(GiacExpr(_expand(_trig_as_exponentials(n))))
    total = zero(GiacExpr)
    for t in terms
        v = invoke_cmd(:integrate, t, x, a, b)
        _integral_left(v) && return nothing
        total += v
    end
    return invoke_cmd(:simplify, total)
end

function _trig_as_exponentials(n::Node)
    i = NumNode(im)
    if n isa CallNode
        args = Node[_trig_as_exponentials(a) for a in n.args]
        if n.f in ("sin", "cos") && length(args) == 1
            u = _mul(i, args[1])
            p, m = _exp(u), _exp(_neg(u))
            return n.f == "cos" ? _mul(_HALF, _add(p, m)) : _mul(NumNode(-im // 2), _sub(p, m))
        end
        return _call(n.f, args)
    elseif n isa AddNode
        return _add((_scale(_trig_as_exponentials(m), c) for (m, c) in n.terms)...)
    elseif n isa MulNode
        return _mul(NumNode(n.coef), (_pow(_trig_as_exponentials(b), _trig_as_exponentials(e)) for (b, e) in n.factors)...)
    end
    return n
end

# ∫ₐᵇ f = ½ ∫ₐᵇ (f + (f∘φ)·|φ′|) for a decreasing involution φ of [a, b]: a+b−x on a finite
# interval, (1−x)/(1+x) on [0, 1] and 1/x on [0, ∞). Taken when the sum is no longer than f and
# GIAC integrates it: x g(x) with g symmetric about the midpoint gives (a+b) g(x), and
# ln(1+x)/(1+x²) on [0, 1] gives ln 2/(1+x²).
function _reflected_integral(f::GiacExpr, x::GiacExpr, a, b)
    for φ in _involutions(x, convert(GiacExpr, a), convert(GiacExpr, b))
        g = invoke_cmd(:simplify, f - substitute(f, x => φ) * invoke_cmd(:diff, φ, x))
        _size(_node(g)) <= _size(_node(f)) || continue
        v = invoke_cmd(:integrate, g, x, a, b)
        _integral_left(v) || return invoke_cmd(:simplify, v / 2)
    end
    return nothing
end

function _involutions(x, a, b)
    infinite(e) = _any_node(n -> n isa SymNode && n.name in ("inf", "infinity"), _node(e))
    φs = GiacExpr[]
    infinite(a) || infinite(b) || push!(φs, a + b - x)
    if _is_zero_expr(a)
        _is_zero_expr(b - 1) && push!(φs, (1 - x) / (1 + x))
        _is_plus_infinity(_node(b)) && push!(φs, 1 / x)
    end
    return φs
end

_is_infinite(r::GiacExpr) = string(r) in ("infinity", "+infinity", "-infinity", "inf", "-inf", "undef")
_is_zero_expr(e::GiacExpr) = (n = _node(e); n isa NumNode && iszero(n.v))

function _antiderivative(f::GiacExpr, x::GiacExpr)
    G = zero(GiacExpr)
    R = GiacExpr(_expand(_node(f)))
    for _ in 1:12
        _is_zero_expr(R) && return G
        g = nothing
        for t in _sum_terms(R)
            g = _term_antiderivative(t, x)
            g === nothing || break
        end
        if g === nothing
            # no derivative term left to take off: GIAC integrates the rest
            rest = invoke_cmd(:integrate, R, x)
            return _is_infinite(rest) ? nothing : G + rest
        end
        G += g
        R = GiacExpr(_expand(_node(R - Commands.diff(g, x))))
    end
    return nothing
end

# For a term V·u⁽ᵏ⁾ (k = 1: V·u′·uᵖ), the expression whose derivative gives it with the rest of
# the product rule left over; nothing if it has no such factor
function _term_antiderivative(t::GiacExpr, x::GiacExpr)
    fs = _product_factors(t)
    i = findfirst(f -> _derivative_in(f, x) !== nothing, fs)
    i === nothing && return nothing
    u, k = _derivative_in(fs[i], x)
    others = fs[[j for j in eachindex(fs) if j != i]]
    k > 1 && return _product(others) * Commands.diff(u, x, k - 1)
    # the powers of u among the other factors join it: V u′ uᵖ is (V uᵖ⁺¹/(p+1))′ less V′ uᵖ⁺¹/(p+1)
    p = 0
    V = GiacExpr[]
    for f in others
        if isequal(f, u)
            p += 1
        elseif iscall(f) && symb_funcname(f) == "^" && isequal(arguments(f)[1], u) && is_integer(arguments(f)[2])
            p += Int(to_julia(arguments(f)[2]))
        else
            push!(V, f)
        end
    end
    return _product(V) * u^(p + 1) / (p + 1)
end

_product(fs) = isempty(fs) ? one(GiacExpr) : reduce(*, fs)

# (u, k) if `f` is the kth derivative in `x` of an unknown function u(x)
function _derivative_in(f::GiacExpr, x::GiacExpr)
    (iscall(f) && symb_funcname(f) == "diff") || return nothing
    a = arguments(f)
    (length(a) >= 2 && isequal(a[2], x) && iscall(a[1]) && symb_funcname(a[1]) == "of") || return nothing
    k = 1
    if length(a) >= 3
        is_integer(a[3]) || return nothing
        k = Int(to_julia(a[3]))
    end
    return (a[1], k)
end
