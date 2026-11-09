# ∫ A/√P dx for polynomials A and P, P of even degree with a square leading coefficient, when it is
# a logarithm (a pseudo-elliptic integral, Chebyshev's among them). A unit U + V√P of the function
# field, U² - PV² constant, has d/dx ln(U + V√P) = (U'/V)/√P, with U'/V a polynomial; such units
# are the convergents of the continued fraction of √P, which is periodic exactly when one exists
# (Abel). Where A is a constant multiple c of U'/V, the integral is c ln(U + V√P).

const _PELL_STEPS = 24

function _pseudo_elliptic(f::GiacExpr, x::GiacExpr)
    parts = _over_square_root(f, x)
    parts === nothing && return nothing
    A, P = parts
    n2 = _polynomial_degree(P, x)
    (n2 !== nothing && n2 >= 4 && iseven(n2)) || return nothing
    unit = _pell_unit(P, x)
    unit === nothing && return nothing
    U, V = unit
    c = invoke_cmd(:normal, A * V / Commands.diff(U, x))
    string(x) in _free_names(c) && return nothing
    F = c * log(U + V * sqrt(P))
    return _is_antiderivative(F, f, x) ? F : nothing
end

# (A, P) for f = A/√P with A and P polynomials in x; nothing for other f
function _over_square_root(f::GiacExpr, x::GiacExpr)
    n = _node(f)
    factors = n isa MulNode ? [(b, e) for (b, e) in n.factors] : n isa CallNode || n isa SymNode || n isa AddNode ? [(n, _ONE)] : Tuple{Node, Node}[]
    radicand, rest = Node[], Node[_node(convert(GiacExpr, n isa MulNode ? n.coef : 1))]
    for (b, e) in factors
        if e isa NumNode && e.v == -1 // 2
            push!(radicand, b)
        else
            push!(rest, _pow(b, e))
        end
    end
    isempty(radicand) && return nothing
    A, P = GiacExpr(_mul(rest...)), invoke_cmd(:expand, GiacExpr(_mul(radicand...)))
    (_is_polynomial_in(A, x) && _is_polynomial_in(P, x)) || return nothing
    return (A, P)
end

_is_polynomial_in(p::GiacExpr, x::GiacExpr) =
    !(string(x) in _free_names(invoke_cmd(:denom, invoke_cmd(:normal, p)))) && _polynomial_degree(p, x) !== nothing &&
    _rational_in(_node(p), string(x)) && !_has_float(_node(p))

# (U, V) with U² - PV² a constant, from the continued fraction of √P: aₖ the polynomial parts,
# mₖ₊₁ = dₖaₖ - mₖ, dₖ₊₁ = (P - mₖ₊₁²)/dₖ, aₖ₊₁ = the polynomial part of (a₀ + mₖ₊₁)/dₖ₊₁, and the
# convergents hₖ/kₖ satisfy hₖ² - Pkₖ² = ±dₖ₊₁; nothing if no dₖ is a constant within the steps
function _pell_unit(P::GiacExpr, x::GiacExpr)
    a0 = _polynomial_square_root_part(P, x)
    a0 === nothing && return nothing
    N(e) = invoke_cmd(:normal, e)
    m, d, a = convert(GiacExpr, 0), convert(GiacExpr, 1), a0
    h1, h0, k1, k0 = convert(GiacExpr, 1), a0, convert(GiacExpr, 0), convert(GiacExpr, 1)
    for _ in 1:_PELL_STEPS
        m = N(d * a - m)
        d = N(invoke_cmd(:quo, P - m^2, d, x))
        _is_zero_expr(d) && return nothing
        string(x) in _free_names(d) || return _integral_unit(h0, k0)
        a = invoke_cmd(:quo, a0 + m, d, x)
        h1, h0 = h0, N(a * h0 + h1)
        k1, k0 = k0, N(a * k0 + k1)
    end
    return nothing
end

# the unit with integer coefficients and U's leading coefficient positive
function _integral_unit(U::GiacExpr, V::GiacExpr)
    l = to_julia(invoke_cmd(:lcm, invoke_cmd(:denom, invoke_cmd(:normal, U)), invoke_cmd(:denom, invoke_cmd(:normal, V))))
    l isa Number || return (U, V)
    s = to_julia(invoke_cmd(:sign, invoke_cmd(:lcoeff, U)))
    λ = (s isa Number && s < 0 ? -1 : 1) * l
    return (invoke_cmd(:normal, λ * U), invoke_cmd(:normal, λ * V))
end

# the polynomial a with a² agreeing with P in its upper half of the degrees, a's leading
# coefficient the square root of P's; nothing where that is not rational
function _polynomial_square_root_part(P::GiacExpr, x::GiacExpr)
    n = _polynomial_degree(P, x) ÷ 2
    r = invoke_cmd(:sqrt, invoke_cmd(:lcoeff, P, x))
    _is_rational_number(r) || return nothing
    a = r * x^n
    # the next coefficient of a from the coefficient of x^(2n-j) of P - a²
    for j in 1:n
        c = invoke_cmd(:coeff, invoke_cmd(:expand, P - a^2), x, 2n - j) / (2r)
        a += c * x^(n - j)
    end
    return invoke_cmd(:normal, a)
end

_is_rational_number(e::GiacExpr) = (v = _node(e); v isa NumNode && v.v isa Union{Integer, Rational})
