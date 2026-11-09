# Elliptic integrals in the parameter m (the square of the modulus), as DLMF §19.2 and Mathematica
# write them:
#     F(φ|m) = ∫₀^φ dθ/√(1 - m sin²θ),   E(φ|m) = ∫₀^φ √(1 - m sin²θ) dθ,
#     Π(n; φ|m) = ∫₀^φ dθ/((1 - n sin²θ)√(1 - m sin²θ)),
# and the complete K(m) = F(π/2|m), E(m) = E(π/2|m), Π(n|m) = Π(n; π/2|m). GIAC has none of them;
# they are calls it does not interpret, as the Bessel functions are (special_functions.jl):
# ellipk(m), ellipe(m), ellipe(φ, m), ellipf(φ, m), ellippi(n, m), ellippi(n, φ, m). Their values
# come from Carlson's symmetric integrals, at any precision and complex arguments.

const _ELLIPTIC = ("ellipk", "ellipe", "ellipf", "ellippi")

# the numbers of arguments each takes
_elliptic_arity(f::AbstractString, k::Integer) =
    f == "ellipk" ? k == 1 : f == "ellipe" ? k in (1, 2) : f == "ellipf" ? k == 2 : f == "ellippi" && k in (2, 3)

"""
    ellipk(m)
    ellipe(m)
    ellipe(φ, m)
    ellipf(φ, m)
    ellippi(n, m)
    ellippi(n, φ, m)

The elliptic integrals in the parameter m: the complete K(m) and E(m), the incomplete F(φ|m) and
E(φ|m) of the first and second kinds, and Π(n|m), Π(n; φ|m) of the third,
`Π(n; φ|m) = ∫₀^φ dθ/((1 - n sin²θ)√(1 - m sin²θ))`. At numbers they are numbers, real or
complex, at the precision of the arguments; with a Giac expression they are symbolic, with their
derivatives, series and values at 0, π/2 and m = 0, 1. `SpecialFunctions.ellipk(x)` and
`SpecialFunctions.ellipe(x)` of a Giac expression are these. `integrate` writes some integrals
with them:

```julia
@giac_var x m a
integrate(1 / sqrt(1 - m * sin(x)^2), x)                   # ellipf(x, m)
integrate(1 / sqrt(cos(x) - cos(a)), x, 0, a)              # sqrt(2)*ellipk(sin(a/2)^2), a pendulum's period
```
"""
function ellipk end
@doc (@doc ellipk) function ellipe end
@doc (@doc ellipk) function ellipf end
@doc (@doc ellipk) function ellippi end

ellipk(m::Number) = _elliptic_number("ellipk", m)
ellipe(m::Number) = _elliptic_number("ellipe", m)
ellipe(φ::Number, m::Number) = _elliptic_number("ellipe", φ, m)
ellipf(φ::Number, m::Number) = _elliptic_number("ellipf", φ, m)
ellippi(n::Number, m::Number) = _elliptic_number("ellippi", n, m)
ellippi(n::Number, φ::Number, m::Number) = _elliptic_number("ellippi", n, φ, m)
for (f, k) in (("ellipk", 1), ("ellipe", 1), ("ellipe", 2), ("ellipf", 2), ("ellippi", 2), ("ellippi", 3))
    args = [Symbol(:a, i) for i in 1:k]
    @eval function $(Symbol(f))($(args...))
        any(a -> a isa GiacExpr, ($(args...),)) || throw(MethodError($(Symbol(f)), ($(args...),)))
        return GiacExpr(_elliptic($f, (_node(convert(GiacExpr, a)) for a in ($(args...),))...))
    end
end
SpecialFunctions.ellipk(m::GiacExpr) = ellipk(m)
SpecialFunctions.ellipe(m::GiacExpr) = ellipe(m)

# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------

# the call, with its known values: K(0) = E(0) = π/2, E(1) = 1, the incomplete integrals at φ = 0,
# at m = 0 (φ), at φ = kπ/2 (k times the complete ones) and where m sin²φ = 1 (from the reciprocal
# parameter, F(φ|m) = K(1/m)/√m and E(φ|m) = √m (E(1/m) - (1 - 1/m) K(1/m))), Π at n = 0, and the
# value at floats
function _elliptic(f::String, args::Node...)
    if all(a -> a isa NumNode, args) && any(_inexact, args)
        v = _elliptic_number(f, (a.v for a in args)...)
        v === nothing || return NumNode(v)
    end
    G = map(GiacExpr, args)
    π_ = convert(GiacExpr, π)
    zero_(e) = _is_zero_node(_node(e))
    whole(e, v) = (n = _node(e); n isa NumNode && n.v == v)
    # k for φ = kπ/2, k whole: F(kπ/2|m) = kK(m), and so for E and Π
    quarter(e) = _whole_number(invoke_cmd(:simplify, 2e / π_))
    if f == "ellipk"
        zero_(G[1]) && return _node(π_ / 2)
    elseif f == "ellipe" && length(G) == 1
        zero_(G[1]) && return _node(π_ / 2)
        whole(G[1], 1) && return _ONE
    elseif f in ("ellipf", "ellipe")
        φ, m = G
        zero_(φ) && return _ZERO
        zero_(m) && return _node(φ)
        k = quarter(φ)
        k === nothing || return _node(k * GiacExpr(_elliptic(f == "ellipf" ? "ellipk" : "ellipe", _node(m))))
        if f == "ellipe" && whole(m, 1) && _is_constant_real_between(φ, -π_ / 2, π_ / 2)
            return _node(sin(φ))
        end
        if zero_(invoke_cmd(:simplify, m * sin(φ)^2 - 1))
            # 1/m = sin²φ, and 1/√m = |sin φ|
            s2 = sin(φ)^2
            K = GiacExpr(_elliptic("ellipk", _node(s2)))
            f == "ellipf" && return _node(K * abs(sin(φ)))
            return _node((GiacExpr(_elliptic("ellipe", _node(s2))) - (1 - s2) * K) / abs(sin(φ)))
        end
    elseif f == "ellippi"
        n = G[1]
        rest = args[2:end]
        zero_(n) && return _elliptic(length(rest) == 1 ? "ellipk" : "ellipf", rest...)
        if length(G) == 3
            φ, m = G[2], G[3]
            zero_(φ) && return _ZERO
            k = quarter(φ)
            k === nothing || return _node(k * GiacExpr(_elliptic("ellippi", args[1], args[3])))
        end
    end
    return _special(f, args...)
end

# whether e is a real number between lo and hi, decided by its value
function _is_constant_real_between(e::GiacExpr, lo::GiacExpr, hi::GiacExpr)
    isempty(_free_names(e)) || return false
    v, l, h = _approx_value(e), _approx_value(lo), _approx_value(hi)
    return v isa Real && l isa Real && h isa Real && l <= v <= h
end

# d/dx of the call (DLMF 19.4): in m,
#     K' = (E - (1-m)K)/(2m(1-m)),   E' = (E - K)/(2m),
#     ∂F/∂m = E(φ|m)/(2m(1-m)) - F(φ|m)/(2m) - sin 2φ/(4(1-m)Δ),   ∂E(φ|m)/∂m = (E(φ|m) - F(φ|m))/(2m),
# Δ = √(1 - m sin²φ); in φ, the integrands; and Π's from those of Carlson's R_J
function _elliptic_derivative(f::String, n::CallNode, x::String)
    args = n.args[2:end]
    G = map(GiacExpr, args)
    total = zero(GiacExpr)
    for (i, a) in enumerate(args)
        da = _derivative(a, x)
        da === nothing && return nothing
        _is_zero_node(da) && continue
        total += GiacExpr(da) * _elliptic_partial(f, G, i)
    end
    return _node(total)
end

function _elliptic_partial(f::String, G::Vector{GiacExpr}, i::Integer)
    call(g, a...) = GiacExpr(_elliptic(g, map(_node, a)...))
    if f == "ellipk"
        m = G[1]
        return (call("ellipe", m) - (1 - m) * call("ellipk", m)) / (2m * (1 - m))
    elseif f == "ellipe" && length(G) == 1
        m = G[1]
        return (call("ellipe", m) - call("ellipk", m)) / (2m)
    elseif f in ("ellipf", "ellipe")
        φ, m = G
        Δ = sqrt(1 - m * sin(φ)^2)
        i == 1 && return f == "ellipf" ? 1 / Δ : Δ
        F, E = call("ellipf", φ, m), call("ellipe", φ, m)
        f == "ellipe" && return (E - F) / (2m)
        return E / (2m * (1 - m)) - F / (2m) - sin(2φ) / (4(1 - m) * Δ)
    end
    # Π(n; φ|m) and Π(n|m) (DLMF 19.4.4 and its incomplete form)
    complete = length(G) == 2
    n, m = G[1], G[end]
    φ = complete ? convert(GiacExpr, π) / 2 : G[2]
    Δ = sqrt(1 - m * sin(φ)^2)
    s2 = complete ? zero(GiacExpr) : sin(2φ)
    F = complete ? call("ellipk", m) : call("ellipf", φ, m)
    E = complete ? call("ellipe", m) : call("ellipe", φ, m)
    P = complete ? call("ellippi", n, m) : call("ellippi", n, φ, m)
    if !complete && i == 2
        return 1 / ((1 - n * sin(φ)^2) * Δ)
    elseif i == 1
        return (E + (m - n) * F / n + (n^2 - m) * P / n - n * Δ * s2 / (2(1 - n * sin(φ)^2))) / (2(m - n) * (n - 1))
    end
    return (E / (m - 1) + P - m * s2 / (2(m - 1) * Δ)) / (2(n - m))
end

# ---------------------------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------------------------

# the value at numbers, real where the integral is: F, E and Π from Carlson's R_F, R_D and R_J
# (DLMF 19.25.5, 19.25.7, 19.25.14), a real φ reduced to [-π/2, π/2] by F(φ + kπ) = F(φ) + 2kK;
# nothing where the arguments are not numbers
function _elliptic_number(f::String, args::Number...)
    all(a -> a isa Number && isfinite(a), args) || return nothing
    T = float(real(promote_type(map(typeof, args)...)))
    T <: AbstractFloat || (T = Float64)
    a = map(v -> v isa Complex ? Complex{T}(v) : T(v), args)
    v = if f == "ellipk"
        _carlson_rf(zero(T), 1 - a[1], one(T))
    elseif f == "ellipe" && length(a) == 1
        m = a[1]
        _carlson_rf(zero(T), 1 - m, one(T)) - m * _carlson_rd(zero(T), 1 - m, one(T)) / 3
    elseif f == "ellippi" && length(a) == 2
        n, m = a
        _carlson_rf(zero(T), 1 - m, one(T)) + n * _carlson_rj(zero(T), 1 - m, one(T), 1 - n) / 3
    else
        φ = f == "ellippi" ? a[2] : a[1]
        m = a[end]
        n = f == "ellippi" ? a[1] : zero(T)
        if φ isa Real && abs(φ) > T(π) / 2
            k = round(φ / T(π))
            whole = k == 0 ? zero(T) : 2k * _elliptic_number(f == "ellippi" ? "ellippi" : f == "ellipf" ? "ellipk" : "ellipe",
                                                             (f == "ellippi" ? (n, m) : (m,))...)
            return whole + _elliptic_number(f, (f == "ellippi" ? (n, φ - k * T(π), m) : (φ - k * T(π), m))...)
        end
        s, c = sin(φ), cos(φ)
        Δ2 = 1 - m * s^2
        if f == "ellipf"
            s * _carlson_rf(c^2, Δ2, one(T))
        elseif f == "ellipe"
            s * _carlson_rf(c^2, Δ2, one(T)) - m * s^3 * _carlson_rd(c^2, Δ2, one(T)) / 3
        else
            s * _carlson_rf(c^2, Δ2, one(T)) + n * s^3 * _carlson_rj(c^2, Δ2, one(T), 1 - n * s^2) / 3
        end
    end
    v === nothing && return nothing
    return v isa Complex && abs(imag(v)) <= 8eps(T) * abs(v) && all(x -> !(x isa Complex), a) ? real(v) : v
end

# Carlson's duplication (Numerical Algorithms 10, 1995), with the arguments complex where one is
# negative, on the side of the cut its +0 imaginary part puts it
_carlson_complex(args...) = any(v -> v isa Real && v < 0, args)

function _carlson_rf(x, y, z)
    _carlson_complex(x, y, z) && return _carlson_rf(complex(x), complex(y), complex(z))
    T = real(typeof(float(x + y + z)))
    x0, y0, z0 = x, y, z
    A0 = (x + y + z) / 3
    Q = (3eps(T))^(-1 / 6) * maximum(abs, (A0 - x, A0 - y, A0 - z))
    A, f = A0, one(T)
    while Q * f >= abs(A)
        sx, sy, sz = sqrt(x), sqrt(y), sqrt(z)
        λ = sx * sy + sx * sz + sy * sz
        x, y, z, A, f = (x + λ) / 4, (y + λ) / 4, (z + λ) / 4, (A + λ) / 4, f / 4
    end
    X, Y = f * (A0 - x0) / A, f * (A0 - y0) / A
    Z = -(X + Y)
    E2, E3 = X * Y - Z^2, X * Y * Z
    return (1 - E2 / 10 + E3 / 14 + E2^2 / 24 - 3 * E2 * E3 / 44) / sqrt(A)
end

function _carlson_rd(x, y, z)
    _carlson_complex(x, y, z) && return _carlson_rd(complex(x), complex(y), complex(z))
    T = real(typeof(float(x + y + z)))
    x0, y0, z0 = x, y, z
    A0 = (x + y + 3z) / 5
    Q = (eps(T) / 4)^(-1 / 6) * maximum(abs, (A0 - x, A0 - y, A0 - z))
    A, f, total = A0, one(T), zero(A0)
    while Q * f >= abs(A)
        sx, sy, sz = sqrt(x), sqrt(y), sqrt(z)
        λ = sx * sy + sx * sz + sy * sz
        total += f / (sz * (z + λ))
        x, y, z, A, f = (x + λ) / 4, (y + λ) / 4, (z + λ) / 4, (A + λ) / 4, f / 4
    end
    X, Y = f * (A0 - x0) / A, f * (A0 - y0) / A
    Z = -(X + Y) / 3
    E2, E3 = X * Y - 6Z^2, (3X * Y - 8Z^2) * Z
    E4, E5 = 3(X * Y - Z^2) * Z^2, X * Y * Z^3
    return f * A^(-3 / 2) * (1 - 3 * E2 / 14 + E3 / 6 + 9 * E2^2 / 88 - 3 * E4 / 22 - 9 * E2 * E3 / 52 + 3 * E5 / 26) + 3total
end

_carlson_rc(x, y) = _carlson_rf(x, y, y)

# R_J(x, y, z, p); for real arguments with p < 0 its Cauchy principal value (DLMF 19.20.14)
function _carlson_rj(x, y, z, p)
    if all(v -> v isa Real, (x, y, z, p)) && p < 0 && min(x, y, z) >= 0
        x, y, z = sort([x, y, z])
        q = y + (z - y) * (y - x) / (y - p)
        return ((q - y) * _carlson_rj(x, y, z, q) - 3_carlson_rf(x, y, z) +
                3sqrt(x * y * z / (x * z + p * q)) * _carlson_rc(x * z + p * q, p * q)) / (y - p)
    end
    _carlson_complex(x, y, z, p) && return _carlson_rj(complex(x), complex(y), complex(z), complex(p))
    T = real(typeof(float(x + y + z + p)))
    x0, y0, z0, p0 = x, y, z, p
    A0 = (x + y + z + 2p) / 5
    δ = (p - x) * (p - y) * (p - z)
    Q = (eps(T) / 4)^(-1 / 6) * maximum(abs, (A0 - x, A0 - y, A0 - z, A0 - p))
    A, f, total = A0, one(T), zero(A0)
    while Q * f >= abs(A)
        sx, sy, sz, sp = sqrt(x), sqrt(y), sqrt(z), sqrt(p)
        λ = sx * sy + sx * sz + sy * sz
        d = (sp + sx) * (sp + sy) * (sp + sz)
        e = f^3 * δ / d^2
        total += f / d * _carlson_rc(one(e), 1 + e)
        x, y, z, p, A, f = (x + λ) / 4, (y + λ) / 4, (z + λ) / 4, (p + λ) / 4, (A + λ) / 4, f / 4
    end
    X, Y, Z = f * (A0 - x0) / A, f * (A0 - y0) / A, f * (A0 - z0) / A
    P = -(X + Y + Z) / 2
    E2 = X * Y + X * Z + Y * Z - 3P^2
    E3 = X * Y * Z + 2 * E2 * P + 4P^3
    E4 = (2X * Y * Z + E2 * P + 3P^3) * P
    E5 = X * Y * Z * P^2
    return f * A^(-3 / 2) * (1 - 3 * E2 / 14 + E3 / 6 + 9 * E2^2 / 88 - 3 * E4 / 22 - 9 * E2 * E3 / 52 + 3 * E5 / 26) + 6total
end

# ---------------------------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------------------------

# K and E are analytic at m = 0, where their derivatives in m are 0/0; the incomplete integrals
# are not analytic in m where m sin²φ = 1, nor K and E at m = 1
function _elliptic_singular(m::CallNode)
    f, args = m.args[1].name, m.args[2:end]
    f in ("ellipk", "ellipe") && length(args) == 1 && return isequal(args[1], _ONE)
    return false
end

# whether the call is K or E of a parameter that vanishes at x = a, where they are written as
# their series
function _elliptic_vanishes_at(m::CallNode, x::GiacExpr, a)
    (m.args[1].name in ("ellipk", "ellipe") && length(m.args) == 2) || return false
    return _is_zero_node(_node(substitute(GiacExpr(m.args[2]), x => a)))
end

# K(u) = (π/2) Σₖ cₖ² uᵏ and E(u) = (π/2) Σₖ cₖ² uᵏ/(1 - 2k), cₖ = (2k)!/(4ᵏ k!²), to uᴺ
function _elliptic_expansion(f::String, u::GiacExpr, N::Integer)
    c(k) = binomial(big(2k), big(k)) // big(4)^k
    s = sum(GiacExpr(NumNode(f == "ellipk" ? c(k)^2 : c(k)^2 // (1 - 2k))) * u^k for k in 0:N)
    return convert(GiacExpr, π) / 2 * s
end

# ---------------------------------------------------------------------------------------------
# Integrals
# ---------------------------------------------------------------------------------------------

# ∫ f dx for f elliptic in one of the forms
#     (a + b sin²u)^(-1/2) (α + β sin²u),   (a + b sin²u)^(1/2),   (a + b sin²u)^(-1/2) (c + d sin²u)^(-1),
# u linear in x, written with any sines and cosines of u and 2u that make them; and
#     ((a + bx²)(c + dx²))^(-1/2),   (a + bx²)^(1/2) (c + dx²)^(-1/2),
# a quartic in x² taken as the product of its two quadratics. Each is checked by differentiating;
# nothing for other integrands
function _elliptic_integral(f::GiacExpr, x::GiacExpr)
    _node(x) isa SymNode || return nothing
    F = _trig_elliptic_integral(f, x)
    F === nothing && (F = _algebraic_elliptic_integral(f, x))
    F === nothing && return nothing
    return _is_antiderivative(F, f, x) ? F : nothing
end

# (c, [(base, exponent)]) for f as c times a product of powers, products inside bases with whole
# exponents spread over their factors
function _power_factors(f::GiacExpr)
    c, out = Ref(one(GiacExpr)), Tuple{GiacExpr, GiacExpr}[]
    function spread(b::Node, k::Node)
        if b isa MulNode && k isa NumNode && k.v isa Union{Integer, Rational}
            isone(b.coef) || (c[] *= GiacExpr(NumNode(b.coef))^GiacExpr(k))
            foreach(((bb, kk),) -> spread(bb, _mul(kk, k)), b.factors)
        else
            push!(out, (GiacExpr(b), GiacExpr(k)))
        end
    end
    spread(_node(f), _ONE)
    return (c[], out)
end

_half_odd(k::GiacExpr) = (v = _node(k); v isa NumNode && v.v isa Rational && denominator(v.v) == 2)

# (a, b, u) with e = a + b sin²u, u linear in x, a and b free of x; nothing where e is not
function _in_sine_squared(e::GiacExpr, x::GiacExpr)
    L = invoke_cmd(:tlin, e)
    trig = Node[]
    _any_node(_node(L)) do m
        m isa CallNode && m.f in ("sin", "cos", "tan") && !_free_of(m, _node(x)) && (m in trig || push!(trig, m))
        false
    end
    (length(trig) == 1 && trig[1].f == "cos") || return nothing
    w = GiacExpr(only(trig[1].args))
    dw = invoke_cmd(:diff, w, x)
    string(x) in _free_names(dw) && return nothing
    C = _fresh_symbol(_node(L), "cw")
    P = invoke_cmd(:normal, GiacExpr(_replace_node(_node(L), trig[1], _node(C))))
    b1 = invoke_cmd(:diff, P, C)
    b0 = invoke_cmd(:normal, P - b1 * C)
    any(v -> string(x) in _free_names(v) || string(C) in _free_names(v), (b0, b1)) && return nothing
    # a' + b' cos w = (a' + b') - 2b' sin²(w/2)
    return (invoke_cmd(:normal, b0 + b1), invoke_cmd(:normal, -2b1), w / 2)
end

function _trig_elliptic_integral(f::GiacExpr, x::GiacExpr)
    _any_node(m -> m isa CallNode && m.f in ("sin", "cos"), _node(f)) || return nothing
    c, factors = _power_factors(f)
    roots = [i for (i, (b, k)) in enumerate(factors) if _half_odd(k) && string(x) in _free_names(b)]
    length(roots) == 1 || return nothing
    B, p = factors[only(roots)]
    v = _node(p).v
    v in (-1 // 2, 1 // 2) || return nothing
    s = _in_sine_squared(B, x)
    s === nothing && return nothing
    a, b, u = s
    du = invoke_cmd(:diff, u, x)
    m = invoke_cmd(:normal, -b / a)
    rest = c * prod((b_^k for (i, (b_, k)) in enumerate(factors) if i != only(roots)); init = one(GiacExpr))
    if !(string(x) in _free_names(rest))
        v == 1 // 2 && return rest * sqrt(a) * ellipe(u, m) / du
        return rest * ellipf(u, m) / (sqrt(a) * du)
    end
    v == -1 // 2 || return nothing
    # α + β sin²u over B, or 1/(c + d sin²u) for Π
    r = _in_sine_squared(rest, x)
    if r !== nothing && _same_angle(r[3], u)
        α, β = r[1], r[2]
        # (α + βS)/√B = (α - βa/b)/√B + (β/b)√B for S = (B - a)/b
        return ((α - β * a / b) * ellipf(u, m) / sqrt(a) + β / b * sqrt(a) * ellipe(u, m)) / du
    end
    q = _in_sine_squared(invoke_cmd(:normal, 1 / rest), x)
    if q !== nothing && _same_angle(q[3], u)
        C0, D0 = q[1], q[2]
        return ellippi(invoke_cmd(:normal, -D0 / C0), u, m) / (C0 * sqrt(a) * du)
    end
    return nothing
end

_same_angle(u::GiacExpr, v::GiacExpr) = _is_zero_expr(invoke_cmd(:normal, u - v)) || _is_zero_expr(invoke_cmd(:normal, u + v))

# ((a + bx²)(c + dx²))^(-1/2) = F(asin(kx), bc/(ad))/(√a √c k) and
# (a + bx²)^(1/2)/(c + dx²)^(1/2) = √a E(asin(kx), bc/(ad))/(√c k), k = √(-d/c) (Byrd and Friedman
# 214.00, 219.00); the quadratic c + dx² taken as the one with a real zero where there is one
function _algebraic_elliptic_integral(f::GiacExpr, x::GiacExpr)
    c, factors = _power_factors(f)
    quadratics = Tuple{GiacExpr, GiacExpr, GiacExpr}[]          # (a, b, exponent)
    for (B, k) in factors
        string(x) in _free_names(B) || (c *= B^k; continue)
        _half_odd(k) || return nothing
        q = _even_quadratics(B, x)
        q === nothing && return nothing
        for (a, b) in q
            push!(quadratics, (a, b, k))
        end
    end
    length(quadratics) == 2 || return nothing
    ks = [_node(q[3]).v for q in quadratics]
    if ks == [-1 // 2, -1 // 2]
        for (i, j) in ((1, 2), (2, 1))
            (a, b, _), (cc, d, _) = quadratics[i], quadratics[j]
            _real_zero(cc, d) || (i, j) == (2, 1) || continue
            k, φ = _elliptic_amplitude(invoke_cmd(:normal, -d / cc), x)
            return c * ellipf(φ, invoke_cmd(:normal, b * cc / (a * d))) / (sqrt(a) * sqrt(cc) * k)
        end
    elseif sort(ks) == [-1 // 2, 1 // 2]
        (a, b, _) = quadratics[findfirst(==(1 // 2), ks)]
        (cc, d, _) = quadratics[findfirst(==(-1 // 2), ks)]
        k, φ = _elliptic_amplitude(invoke_cmd(:normal, -d / cc), x)
        return c * sqrt(a) * ellipe(φ, invoke_cmd(:normal, b * cc / (a * d))) / (sqrt(cc) * k)
    end
    return nothing
end

# k = √k² and asin(kx), written i asinh(κx) for k = iκ where k² is a negative number
function _elliptic_amplitude(k2::GiacExpr, x::GiacExpr)
    v = _approx_value(k2)
    if v isa Real && v < 0
        κ = sqrt(-k2)
        return (convert(GiacExpr, im) * κ, convert(GiacExpr, im) * asinh(κ * x))
    end
    k = sqrt(k2)
    return (k, asin(k * x))
end

# whether c + dx² has a real zero, as far as its numbers decide
function _real_zero(c::GiacExpr, d::GiacExpr)
    v = _approx_value(invoke_cmd(:normal, -c / d))
    return v isa Real && v > 0
end

# B as a product of even quadratics a + bx²: one, [(a, b)], or a quartic in x² as two with real
# coefficients; nothing for anything else
function _even_quadratics(B::GiacExpr, x::GiacExpr)
    P = invoke_cmd(:normal, B)
    _is_polynomial_in(P, x) || return nothing
    d = _polynomial_degree(P, x)
    coeff(k) = invoke_cmd(:coeff, P, x, k)
    all(k -> _is_zero_expr(coeff(k)), 1:2:d) || return nothing
    d == 2 && return [(coeff(0), coeff(2))]
    d == 4 || return nothing
    p0, p2, p4 = coeff(0), coeff(2), coeff(4)
    disc = invoke_cmd(:normal, p2^2 - 4p4 * p0)
    v = _approx_value(disc)
    (v isa Real && v > 0) || return nothing
    r = sqrt(disc)
    X1, X2 = invoke_cmd(:normal, (-p2 + r) / (2p4)), invoke_cmd(:normal, (-p2 - r) / (2p4))
    # p4 (x² - X1)(x² - X2) = (-X1 + x²)(-p4 X2 + p4 x²)
    return [(-X1, one(GiacExpr)), (invoke_cmd(:normal, -p4 * X2), p4)]
end
