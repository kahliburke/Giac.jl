# Second-order linear equations y'' + a y' + b y = 0 with rational coefficients, by the first case
# of Kovacic's algorithm (J. Symbolic Computation 2, 1986): with y = e^(-∫a/2) z the equation is
# z'' = r z, r = a²/4 + a'/2 - b, and a solution z = P e^(∫ω), P a polynomial and ω rational, is
# found from the poles of r and its behaviour at infinity, where there is one. The second
# solution is y₁ ∫ e^(-∫a)/y₁². Riccati equations y' = q₀ + q₁y + q₂y² are solved through the
# linear equation of y = -u'/(q₂u).

function _kovacic(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    c = _linear_coefficients(eq, x, y)
    (c === nothing || length(c.q) != 3 || !_is_zero_expr(c.f)) && return nothing
    a = invoke_cmd(:normal, c.q[2] / c.q[3])
    b = invoke_cmd(:normal, c.q[1] / c.q[3])
    all(q -> _is_rational_in(_node(q), string(x)), (a, b)) || return nothing
    r = invoke_cmd(:normal, a^2 / 4 + invoke_cmd(:diff, a, x) / 2 - b)
    z = _kovacic_case1(r, x)
    z === nothing && return nothing
    half = _integrate_or_nothing(a / 2, x)
    half === nothing && return nothing
    y1 = invoke_cmd(:simplify, exp(-half) * z)
    w = _integrate_or_nothing(invoke_cmd(:simplify, exp(-2half) / y1^2), x)
    second = w === nothing ? Commands.integrate(invoke_cmd(:simplify, exp(-2half) / y1^2), x) : w
    C0, C1 = GiacExpr(SymNode("c_0")), GiacExpr(SymNode("c_1"))
    return C0 * y1 + C1 * _smaller_form(_erfi_form(y1 * second))
end

# a solution P e^(∫ω) of z'' = r z, r rational in x, or nothing where there is none of that form
function _kovacic_case1(r::GiacExpr, x::GiacExpr)
    s, t = invoke_cmd(:numer, r), invoke_cmd(:denom, r)
    _is_zero_expr(s) && return one(GiacExpr)
    choices = Vector{Vector{Tuple{GiacExpr, GiacExpr}}}()          # per point: ([√r] term, α) pairs
    poles = GiacExpr[]
    R = invoke_cmd(:roots, t)
    for i in 1:length(R)
        c, m = R[i][1], to_julia(R[i][2])
        push!(poles, c)
        if m == 1
            push!(choices, [(zero(GiacExpr), one(GiacExpr))])
        elseif m == 2
            β = first(_laurent_coefficients(r, x, c, 2, 1))
            q = sqrt(1 // 4 + β)
            push!(choices, [(zero(GiacExpr), 1 // 2 + q), (zero(GiacExpr), 1 // 2 - q)])
        elseif iseven(m)
            ν = m ÷ 2
            σ = _series_root(_laurent_coefficients(r, x, c, m, ν))
            S = sum(σ[j + 1] * (x - c)^(j - ν) for j in 0:(ν - 2))
            push!(choices, [(S, (ν + 2σ[ν]) / 2), (-S, (ν - 2σ[ν]) / 2)])
        else
            return nothing                       # a pole of odd order past 1: no solution of this form
        end
    end
    o = to_julia(invoke_cmd(:degree, t, x)) - to_julia(invoke_cmd(:degree, s, x))
    if o > 2
        push!(choices, [(zero(GiacExpr), zero(GiacExpr)), (zero(GiacExpr), one(GiacExpr))])
    elseif o == 2
        β = invoke_cmd(:lcoeff, s, x) / invoke_cmd(:lcoeff, t, x)
        q = sqrt(1 // 4 + β)
        push!(choices, [(zero(GiacExpr), 1 // 2 + q), (zero(GiacExpr), 1 // 2 - q)])
    elseif iseven(o)
        ν = -o ÷ 2
        σ = _series_root(_coefficients_at_infinity(r, x, 2ν, ν + 2))
        S = sum(σ[j + 1] * x^(ν - j) for j in 0:ν)
        push!(choices, [(S, (2σ[ν + 2] - ν) / 2), (-S, (-2σ[ν + 2] - ν) / 2)])
    else
        return nothing
    end
    for pick in Iterators.product(choices...)
        α∞ = last(pick)[2]
        d = _whole_number(invoke_cmd(:simplify, α∞ - sum((p[2] for p in pick[1:(end - 1)]); init = zero(GiacExpr))))
        (d === nothing || d < 0) && continue
        ω = last(pick)[1] + sum((pick[i][1] + pick[i][2] / (x - poles[i]) for i in eachindex(poles)); init = zero(GiacExpr))
        P = _kovacic_polynomial(ω, r, x, d)
        P === nothing && continue
        ∫ω = _integrate_or_nothing(ω, x)
        ∫ω === nothing && continue
        return invoke_cmd(:simplify, P * exp(∫ω))
    end
    return nothing
end

# the monic polynomial P of degree d with P'' + 2ωP' + (ω' + ω² - r)P = 0, or nothing
function _kovacic_polynomial(ω::GiacExpr, r::GiacExpr, x::GiacExpr, d::Integer)
    p = [_fresh_symbol(_node(ω + r), "p$(k)__") for k in 0:(d - 1)]
    P = x^d + sum((p[k + 1] * x^k for k in 0:(d - 1)); init = zero(GiacExpr))
    D(f) = invoke_cmd(:diff, f, x)
    E = invoke_cmd(:numer, invoke_cmd(:normal, D(D(P)) + 2ω * D(P) + (D(ω) + ω^2 - r) * P))
    _is_zero_expr(E) && return P
    d == 0 && return nothing
    n = to_julia(invoke_cmd(:degree, E, x))
    eqs = [invoke_cmd(:coeff, E, x, k) for k in 0:n]
    sol = try
        invoke_cmd(:linsolve, eqs, p)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (giac_type(sol) == GenTypes.VECT && length(sol) == d) || return nothing
    Q = invoke_cmd(:normal, substitute(P, Dict(zip(p, collect(sol)))))
    _is_zero_expr(invoke_cmd(:normal, D(D(Q)) + 2ω * D(Q) + (D(ω) + ω^2 - r) * Q)) || return nothing
    return Q
end

# the coefficients ρ₀, …, ρₙ₋₁ of r at its pole c of order m: r = Σ ρⱼ (x - c)^(j - m)
function _laurent_coefficients(r::GiacExpr, x::GiacExpr, c::GiacExpr, m::Integer, n::Integer)
    h = _fresh_symbol(_node(r), "h")
    return _taylor_coefficients(invoke_cmd(:normal, substitute(r, x => c + h) * h^m), h, n)
end

# the coefficients ρ₀, …, ρₙ₋₁ of r at infinity where it grows as x^m: r = Σ ρⱼ x^(m - j)
function _coefficients_at_infinity(r::GiacExpr, x::GiacExpr, m::Integer, n::Integer)
    h = _fresh_symbol(_node(r), "h")
    return _taylor_coefficients(invoke_cmd(:normal, substitute(r, x => 1 / h) * h^m), h, n)
end

function _taylor_coefficients(f::GiacExpr, h::GiacExpr, n::Integer)
    out = GiacExpr[]
    g = f
    for j in 0:(n - 1)
        push!(out, invoke_cmd(:simplify, substitute(g, h => 0) / factorial(big(j))))
        g = invoke_cmd(:diff, g, h)
    end
    return out
end

# the coefficients σ of the square root of the series Σ ρⱼ hʲ, σ₀ = √ρ₀ (a for a², not |a|: either
# sign is one of the choices)
function _series_root(ρ::Vector{GiacExpr})
    σ = GiacExpr[_square_root(ρ[1])]
    for j in 2:length(ρ)
        push!(σ, invoke_cmd(:simplify, (ρ[j] - sum((σ[k] * σ[j - k + 1] for k in 2:(j - 1)); init = zero(GiacExpr))) / (2σ[1])))
    end
    return σ
end

function _whole_number(e::GiacExpr)
    n = _node(e)
    return n isa NumNode && n.v isa Integer ? Int(n.v) : nothing
end

# ---------------------------------------------------------------------------------------------
# Riccati equations
# ---------------------------------------------------------------------------------------------

# y' = q₀ + q₁y + q₂y² with q₂ ≠ 0: y = -u'/(q₂u) for u'' - (q₁ + q₂'/q₂) u' + q₀q₂ u = 0, solved
# by the whole `desolve`; the general solution, its constant c_0, or nothing
function _riccati(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    S = _derivative_symbols(eq, x, y)
    (S === nothing || length(S.d) != 2) && return nothing
    w, p = S.d
    A = invoke_cmd(:diff, S.E, p)
    names = Set(string.(S.d))
    isempty(intersect(_free_names(A), names)) || return nothing
    F = invoke_cmd(:normal, -substitute(S.E, p => 0) / A)
    D(f, v = w) = invoke_cmd(:diff, f, v)
    _is_zero_expr(invoke_cmd(:normal, D(D(D(F))))) || return nothing
    q2 = invoke_cmd(:normal, D(D(F)) / 2)
    _is_zero_expr(q2) && return nothing
    q1 = substitute(D(F), w => 0)
    q0 = substitute(F, w => 0)
    isempty(intersect(union(_free_names(q0), _free_names(q1), _free_names(q2)), names)) || return nothing
    u = giac_eval(string(_fresh_symbol(S.e, "u")) * "(" * string(x) * ")")
    Dx(f, k = 1) = Commands.diff(f, x, k)
    linear = invoke_cmd(:normal, q1 + Dx(q2) / q2)
    U = try
        Commands.desolve(Dx(u, 2) - linear * Dx(u) + invoke_cmd(:normal, q0 * q2) * u ~ 0, u)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (_gave_up(U) || !occursin("c_1", string(U))) && return nothing
    C0, C1 = GiacExpr(SymNode("c_0")), GiacExpr(SymNode("c_1"))
    U = substitute(U, Dict(C0 => one(GiacExpr), C1 => C0))
    return _exp_powers(invoke_cmd(:simplify, _erfi_form(-Dx(U) / (q2 * U))))
end

# e with each whole power of an exponential, (e^u)ᵏ, written e^(ku)
_exp_powers(e::GiacExpr) = GiacExpr(_exp_powers(_node(e)))
_exp_powers(n::Node) = _replace_nodes(n) do m
    m isa MulNode || return nothing
    whole(b, k) = b isa CallNode && b.f == "exp" && k isa NumNode && k.v isa Integer && k.v != 1
    any(((b, k),) -> whole(b, k), m.factors) || return nothing
    return _mul(NumNode(m.coef), (whole(b, k) ? _exp(_mul(k, _exp_powers(only(b.args)))) : _pow(_exp_powers(b), _exp_powers(k))
                                  for (b, k) in m.factors)...)
end

# ---------------------------------------------------------------------------------------------
# Forcing terms and conditions
# ---------------------------------------------------------------------------------------------

const _DESOLVE_METHODS = (:_euler_cauchy, :_bessel_equation, :_kovacic, :_riccati)

# the solution of `eqs`, an equation or a vector of one and conditions on y and its derivatives at
# points, by the methods after GIAC's: the equation's general solution, with a forcing term by
# variation of parameters, and its constants fitted to the conditions; nothing where none applies
function _desolve_methods(eqs, x::GiacExpr, y::GiacExpr)
    eq, conditions = eqs isa GiacExpr ? (eqs, Any[]) : (first(eqs), collect(Any, eqs[2:end]))
    eq isa GiacExpr || return nothing
    S = _general_solution(eq, x, y)
    S === nothing && return nothing
    return isempty(conditions) ? S : _fit_conditions(S, conditions, x, y)
end

function _general_solution(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    for m in _DESOLVE_METHODS
        v = getfield(@__MODULE__, m)(eq, x, y)
        v === nothing || return v
    end
    return _forced_solution(eq, x, y)
end

# q₂y'' + q₁y' + q₀y = f with f ≠ 0: the homogeneous equation's solutions y₁, y₂ (GIAC's, or from
# the methods above) and y = c₀y₁ + c₁y₂ - y₁ ∫ y₂g/W + y₂ ∫ y₁g/W, g = f/q₂, W = y₁y₂' - y₁'y₂
function _forced_solution(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    c = _linear_coefficients(eq, x, y)
    (c === nothing || length(c.q) != 3 || _is_zero_expr(c.f)) && return nothing
    D(f, k = 1) = Commands.diff(f, x, k)
    homogeneous = c.q[3] * D(y, 2) + c.q[2] * D(y) + c.q[1] * y ~ 0
    H = try
        Commands.desolve(homogeneous, y)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (H isa GiacExpr && !_gave_up(H)) || return nothing
    C0, C1 = GiacExpr(SymNode("c_0")), GiacExpr(SymNode("c_1"))
    y1, y2 = substitute(H, Dict(C0 => 1, C1 => 0)), substitute(H, Dict(C0 => 0, C1 => 1))
    W = invoke_cmd(:simplify, y1 * D(y2) - D(y1) * y2)
    _is_zero_expr(W) && return nothing
    g = c.f / c.q[3]
    I1 = Commands.integrate(invoke_cmd(:simplify, y2 * g / W), x)
    I2 = Commands.integrate(invoke_cmd(:simplify, y1 * g / W), x)
    return H + _exp_powers(_smaller_form(_erfi_form(y2 * I2 - y1 * I1)))
end

# S with its constants c_0, c_1 fitted to conditions y⁽ᵏ⁾(x₀) = v, written y(x₀), y'(x₀), …
function _fit_conditions(S::GiacExpr, conditions::Vector, x::GiacExpr, y::GiacExpr)
    parsed = _function_call(y)
    parsed === nothing && return nothing
    name = String(parsed[1])
    constants = [GiacExpr(SymNode(c)) for c in ("c_0", "c_1") if c in _free_names(S)]
    length(constants) == length(conditions) || return nothing
    equations = GiacExpr[]
    for q in conditions
        parts = _condition_parts(q)
        parts === nothing && return nothing
        m = match(r"^(\w+)('*)\((.*)\)$", parts[1])
        (m === nothing || m[1] != name) && return nothing
        k, x0 = length(m[2]), giac_eval(m[3])
        v = substitute(Commands.diff(S, x, k), x => x0)
        _is_finite_value(v) || return nothing
        push!(equations, v - parts[2])
    end
    sol = try
        invoke_cmd(:solve, equations, constants)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (giac_type(sol) == GenTypes.VECT && length(sol) == 1) || return nothing
    values = sol[1]
    values = giac_type(values) == GenTypes.VECT ? collect(values) : [values]
    length(values) == length(constants) || return nothing
    return _exp_powers(invoke_cmd(:simplify, substitute(S, Dict(zip(constants, values)))))
end

_function_call(y::GiacExpr) = _parse_function_args(string(y))

# a condition's sides, the left as written (y(0), y'(1)): from an equation, or from a derivative
# condition's string
_condition_parts(q) = nothing
_condition_parts(q::GiacExpr) = _is_equation(q) ? (string(lhs(q)), rhs(q)) : nothing
function _condition_parts(q::DerivativeCondition)
    i = findfirst('=', q.condition_str)
    i === nothing && return nothing
    return (strip(q.condition_str[1:(i - 1)]), giac_eval(q.condition_str[(i + 1):end]))
end
