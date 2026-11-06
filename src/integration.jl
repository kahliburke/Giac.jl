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
# integral as derivatives of a known one in its parameters (parametric_integrals.jl)
function _integrate(f::GiacExpr, x::GiacExpr, a, b)
    r = invoke_cmd(:integrate, f, x, a, b)
    _any_node(n -> n isa CallNode && n.f == "integrate", _node(r)) || return r
    v = _parametric_integral(f, x, a, b)
    return v === nothing ? r : v
end

_is_infinite(r::GiacExpr) = string(r) in ("infinity", "+infinity", "-infinity", "undef")
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
