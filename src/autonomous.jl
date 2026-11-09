# Autonomous equations y'' + g(y) = 0, which GIAC does not solve, through the energy integral
# y'²/2 + V(y) = E, V' = g: x + c = ∫ dy/√(2(E - V(y))) is elliptic for the pendulum, g = k sin y,
# and for Duffing's oscillator, g = αy + βy³, and is inverted with Jacobi's functions (jacobi.jl).
# General solutions, the pendulum's in its energy c₀ and Duffing's in its amplitude c₀:
#     y = 2 am(√((c₀ + k)/2) (x + c₁) | 2k/(c₀ + k)),     y = c₀ cn(ω (x + c₁) | m),
# ω² = α + βc₀², m = βc₀²/(2ω²). From y(x₀) = a, y'(x₀) = b the pendulum swings, κ² < 1, or turns
# over, κ² > 1, κ² = b²/(4k) + sin²(a/2):
#     y = 2 asin(κ sn(√k (x - x₀) + u₀ | κ²)),   sn(u₀|κ²) = sin(a/2)/κ, cn(u₀) of the sign of b,
#     y = 2 am(±√k κ (x - x₀) + F(a/2 | 1/κ²) | 1/κ²),   the sign of b;
# at rest at a, y = 2 asin(sin(a/2) cd(√k (x - x₀) | sin²(a/2))), cd = cn/dn. Duffing's from rest
# at a is a cn(ω (x - x₀) | m) with c₀ = a.

# (:pendulum, k) or (:duffing, α, β) for an equation y'' + g(y) = 0 of those forms; nothing else
function _autonomous_form(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    S = _derivative_symbols(eq, x, y)
    (S === nothing || length(S.d) != 3) && return nothing
    d0, d1, d2 = S.d
    E = S.E
    names(e) = _free_names(e)
    (string(d1) in names(E) || string(x) in names(E)) && return nothing
    q = invoke_cmd(:diff, E, d2)
    (isempty(intersect(names(q), string.((d0, d1, d2, x)))) && !_is_zero_expr(q)) || return nothing
    g = invoke_cmd(:normal, (E - q * d2) / q)
    string(d2) in names(g) && return nothing
    k = invoke_cmd(:normal, g / sin(d0))
    string(d0) in names(k) || return (:pendulum, k)
    _is_polynomial_in(g, d0) || return nothing
    _polynomial_degree(g, d0) == 3 || return nothing
    c = [invoke_cmd(:coeff, g, d0, j) for j in 0:3]
    (_is_zero_expr(c[1]) && _is_zero_expr(c[3])) || return nothing
    return (:duffing, c[2], c[4])
end

# the general solution of an autonomous equation of the forms above, or nothing
function _autonomous(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    form = _autonomous_form(eq, x, y)
    form === nothing && return nothing
    C0, C1 = GiacExpr(SymNode("c_0")), GiacExpr(SymNode("c_1"))
    if form[1] === :pendulum
        k = form[2]
        return 2jacobiam(sqrt((C0 + k) / 2) * (x + C1), invoke_cmd(:normal, 2k / (C0 + k)))
    end
    α, β = form[2], form[3]
    ω2 = α + β * C0^2
    return C0 * jacobicn(sqrt(ω2) * (x + C1), invoke_cmd(:normal, β * C0^2 / (2ω2)))
end

# the solution of y'' + g(y) = 0 with y(x₀) = a and y'(x₀) = b, for the forms above, checked in
# the equation and at x₀; nothing where the cases are not decided by the numbers
function _autonomous_initial(eq::GiacExpr, conditions::Vector, x::GiacExpr, y::GiacExpr)
    form = _autonomous_form(eq, x, y)
    form === nothing && return nothing
    v = _initial_values(conditions, y)
    v === nothing && return nothing
    x0, a, b = v
    s = x - x0
    sol = if form[1] === :pendulum
        _pendulum(form[2], a, b, s)
    elseif _is_zero_expr(b)
        α, β = form[2], form[3]
        ω2 = invoke_cmd(:normal, α + β * a^2)
        a * jacobicn(sqrt(ω2) * s, invoke_cmd(:normal, β * a^2 / (2ω2)))
    end
    sol === nothing && return nothing
    sol = _smaller_form(sol)
    # checked with the parameters given values
    names = setdiff(_free_names(eq + sol), [string(x), "c_0", "c_1"])
    at = Dict{GiacExpr, Any}(giac_identifier(n) => 1 // 2 + j // 7 for (j, n) in enumerate(sort(collect(names))))
    x00 = _approx_value(substitute(x0, at))
    x00 isa Real || return nothing
    _residual(eq, sol, x, y; values = at, points = (x00 + 0.31, x00 + 0.77)) === true || return nothing
    for (f, w) in ((sol, a), (invoke_cmd(:diff, sol, x), b))
        d = _approx_value(substitute(f - w, merge(at, Dict(x => x0))))
        (d isa Number && abs(d) < 1e-10) || return nothing
    end
    return sol
end

function _pendulum(k::GiacExpr, a::GiacExpr, b::GiacExpr, s::GiacExpr)
    r = sqrt(k)
    if _is_zero_expr(b)                       # at rest at a
        h = sin(a / 2)
        m = h^2
        return 2asin(h * jacobicn(r * s, m) / jacobidn(r * s, m))
    end
    sb = _approx_value(b)
    sb isa Real || return nothing
    κ2 = invoke_cmd(:normal, b^2 / (4k) + sin(a / 2)^2)
    v = _approx_value(κ2)
    v isa Real || return nothing
    if v < 1                                  # it swings
        κ = sqrt(κ2)
        u0 = ellipf(asin(sin(a / 2) / κ), κ2)
        sb < 0 && (u0 = 2ellipk(κ2) - u0)
        return 2asin(κ * jacobisn(r * s + u0, κ2))
    elseif v > 1                              # it turns over
        m = invoke_cmd(:normal, 1 / κ2)
        return 2jacobiam((sb > 0 ? 1 : -1) * r * sqrt(κ2) * s + ellipf(a / 2, m), m)
    end
    return nothing
end

# (x₀, a, b) for conditions y(x₀) = a and y'(x₀) = b, in either order; nothing for others
function _initial_values(conditions::Vector, y::GiacExpr)
    length(conditions) == 2 || return nothing
    parsed = _function_call(y)
    parsed === nothing && return nothing
    name = String(parsed[1])
    found = Dict{Int, Tuple{GiacExpr, GiacExpr}}()
    for q in conditions
        parts = _condition_parts(q)
        parts === nothing && return nothing
        m = match(r"^(\w+)('*)\((.*)\)$", parts[1])
        (m === nothing || m[1] != name) && return nothing
        found[length(m[2])] = (giac_eval(m[3]), parts[2])
    end
    (haskey(found, 0) && haskey(found, 1)) || return nothing
    _is_zero_expr(found[0][1] - found[1][1]) || return nothing
    return (found[0][1], found[0][2], found[1][2])
end
