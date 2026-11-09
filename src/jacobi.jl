# Jacobi's elliptic functions in the parameter m, as DLMF §22 and Mathematica write them: the
# amplitude am(u|m), the inverse of u = F(φ|m), and sn = sin am, cn = cos am, dn = √(1 - m sn²).
# They are calls GIAC does not interpret, kept with the elliptic integrals (elliptic.jl):
# jacobiam(u, m), jacobisn(u, m), jacobicn(u, m), jacobidn(u, m).

const _JACOBI = ("jacobiam", "jacobisn", "jacobicn", "jacobidn")

"""
    jacobiam(u, m)
    jacobisn(u, m)
    jacobicn(u, m)
    jacobidn(u, m)

Jacobi's amplitude am(u|m), the φ with F(φ|m) = u, and his elliptic functions sn(u|m) = sin φ,
cn(u|m) = cos φ and dn(u|m) = √(1 - m sin²φ). At numbers they are numbers, at the precision of
the arguments; with a Giac expression they are symbolic, with their derivatives, their values at
u = 0 and K(m) and at m = 0 and 1, and am(F(φ|m)|m) = φ.

```julia
@giac_var u m
diff(Giac.jacobisn(u, m), u)          # jacobicn(u,m)*jacobidn(u,m)
Giac.jacobisn(u, 0)                   # sin(u)
```
"""
function jacobiam end
@doc (@doc jacobiam) function jacobisn end
@doc (@doc jacobiam) function jacobicn end
@doc (@doc jacobiam) function jacobidn end

for f in _JACOBI
    F = Symbol(f)
    @eval $F(u::Number, m::Number) = _jacobi_number($f, u, m)
    @eval $F(u::GiacExpr, m::GiacExpr) = GiacExpr(_jacobi($f, _node(u), _node(m)))
    @eval $F(u::Number, m::GiacExpr) = $F(convert(GiacExpr, u), m)
    @eval $F(u::GiacExpr, m::Number) = $F(u, convert(GiacExpr, m))
end

# the call, with its known values and symmetries: am and sn odd in u, cn and dn even; at u = 0;
# at m = 0 (am = u) and m = 1 (sn = tanh u, cn = dn = sech u); at u = K(m) (sn = 1); and at
# u = F(φ|m), where am is φ
function _jacobi(f::String, u::Node, m::Node)
    if u isa NumNode && m isa NumNode && (_inexact(u) || _inexact(m))
        v = _jacobi_number(f, u.v, m.v)
        v === nothing || return NumNode(v)
    end
    c = u isa NumNode ? u.v : u isa MulNode ? u.coef : 1
    if c isa Real && c < 0
        v = _jacobi(f, _neg(u), m)
        return f in ("jacobiam", "jacobisn") ? _neg(v) : v
    end
    U, M = GiacExpr(u), GiacExpr(m)
    if _is_zero_node(u)
        return f in ("jacobiam", "jacobisn") ? _ZERO : _ONE
    elseif _is_zero_node(m)
        return _node(f == "jacobiam" ? U : f == "jacobisn" ? sin(U) : f == "jacobicn" ? cos(U) : one(GiacExpr))
    elseif m isa NumNode && isone(m.v)
        return _node(f == "jacobiam" ? 2atan(tanh(U / 2)) : f == "jacobisn" ? tanh(U) : 1 / cosh(U))
    end
    if u isa CallNode && _special_name(u) == "ellipk" && isequal(u.args[2], m)
        return _node(f == "jacobiam" ? convert(GiacExpr, π) / 2 : f == "jacobisn" ? one(GiacExpr) :
                     f == "jacobicn" ? zero(GiacExpr) : sqrt(1 - M))
    end
    if u isa CallNode && _special_name(u) == "ellipf" && isequal(u.args[3], m)
        φ = GiacExpr(u.args[2])
        return _node(f == "jacobiam" ? φ : f == "jacobisn" ? sin(φ) : f == "jacobicn" ? cos(φ) : sqrt(1 - M * sin(φ)^2))
    end
    return _special(f, u, m)
end

# ∂/∂u: am' = dn, sn' = cn dn, cn' = -sn dn, dn' = -m sn cn; ∂/∂m from F(am|m) = u held fixed,
#     ∂am/∂m = -dn E(am|m)/(2m(1-m)) + dn u/(2m) + sin(2am)/(4(1-m)),
# and through it for sn, cn and dn (dn² = 1 - m sn²)
function _jacobi_partial(f::String, G::Vector{GiacExpr}, i::Integer)
    u, m = G
    sn, cn, dn = (GiacExpr(_jacobi(g, _node(u), _node(m))) for g in ("jacobisn", "jacobicn", "jacobidn"))
    if i == 1
        return f == "jacobiam" ? dn : f == "jacobisn" ? cn * dn : f == "jacobicn" ? -sn * dn : -m * sn * cn
    end
    am = GiacExpr(_jacobi("jacobiam", _node(u), _node(m)))
    E = GiacExpr(_elliptic("ellipe", _node(am), _node(m)))
    dam = -dn * E / (2m * (1 - m)) + dn * u / (2m) + sin(2am) / (4(1 - m))
    f == "jacobiam" && return dam
    f == "jacobisn" && return cn * dam
    f == "jacobicn" && return -sn * dam
    return -(sn^2 + 2m * sn * cn * dam) / (2dn)
end

# the value at numbers: am by Newton's method on F(φ|m) = u, after u is brought within [-K, K]
# by am(u + 2kK) = am(u) + kπ, for m ≤ 1; for m > 1 sn, cn and dn from the parameter 1/m,
#     sn(u|m) = sn(√m u|1/m)/√m,   cn(u|m) = dn(√m u|1/m),   dn(u|m) = cn(√m u|1/m)
function _jacobi_number(f::String, u::Number, m::Number)
    (isfinite(u) && isfinite(m)) || return nothing
    T = float(real(promote_type(typeof(u), typeof(m))))
    T <: AbstractFloat || (T = Float64)
    U = u isa Complex ? Complex{T}(u) : T(u)
    M = m isa Complex ? Complex{T}(m) : T(m)
    if M isa Real && M > 1
        f == "jacobiam" && return nothing
        r = sqrt(M)
        g = f == "jacobisn" ? "jacobisn" : f == "jacobicn" ? "jacobidn" : "jacobicn"
        v = _jacobi_number(g, r * U, 1 / M)
        return v === nothing ? nothing : f == "jacobisn" ? v / r : v
    end
    if M isa Real && M == 1
        f == "jacobiam" && return 2atan(tanh(U / 2))
        return f == "jacobisn" ? tanh(U) : 1 / cosh(U)
    end
    φ = _jacobi_amplitude(U, M)
    φ === nothing && return nothing
    f == "jacobiam" && return φ
    s = sin(φ)
    f == "jacobisn" && return s
    f == "jacobicn" && return cos(φ)
    return sqrt(1 - M * s^2)
end

function _jacobi_amplitude(u, m)
    T = real(typeof(float(u + m)))
    K = _elliptic_number("ellipk", m)
    turns = real(u) isa Real && K isa Real ? round(real(u) / (2K)) : zero(T)
    r = u - 2turns * K
    φ = r * T(π) / (2K)
    for _ in 1:100
        Δ = sqrt(1 - m * sin(φ)^2)
        step = (_elliptic_number("ellipf", φ, m) - r) * Δ
        φ -= step
        abs(step) <= 4eps(T) * max(1, abs(φ)) && return φ + turns * T(π)
    end
    return nothing
end
