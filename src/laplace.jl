# Laplace transforms. GIAC's `laplace` returns 0 for some transforms it does not find (√t, 1/√t),
# and writes erf where erfc belongs (e^(-t²), erf t); its `ilaplace` returns a transform it cannot
# invert times Dirac(t). So each answer is checked against the integral that defines it,
#
#     F(s₀) = ∫₀^∞ e^(-s₀t) f(t) dt,
#
# computed numerically at two points s₀ (an exp-sinh quadrature, with f evaluated by GIAC in one
# call). Where GIAC has no answer or a wrong one, the transform is found a term at a time: factors
# free of t come out, a factor e^(at) shifts the transform to s - a, and the rest is GIAC's
# transform where it passes, else one from a table of pairs in √t and 1/t, from the theorems
#
#     L{tⁿg} = (-1)ⁿ dⁿG/dsⁿ,   L{g/t} = ∫ₛ^∞ G,   L{g} = (g(0⁺) + L{g'})/s,
#
# or from the integral itself with s > 0. An inverse is found a term at a time from powers
# (s + a)^(-p) and delays e^(-bs), GIAC's inverse of the term where it passes, the table, or
#
#     L⁻¹{G/s} = ∫₀ᵗ g,   L⁻¹{G} = -L⁻¹{G'}/t,   L⁻¹{AB} = ∫₀ᵗ a(τ) b(t-τ) dτ.
#
# A transform that is not found stays held.

function _laplace_cmd(f, t, s)
    f, t, s = convert(GiacExpr, f), convert(GiacExpr, t), convert(GiacExpr, s)
    (_node(t) isa SymNode && _node(s) isa SymNode) || return _giac_cmd(:laplace, f, t, s)
    r = _giac_or_nothing(:laplace, f, t, s)
    if _is_laplace_answer(r, f, t)
        _exponential_polynomial(_node(f), string(t)) && return r
        _laplace_verdict(f, t, r, s) !== :disagree && return r
    end
    g = _laplace_by_terms(f, t, s)
    return g === nothing ? _hold(:laplace, (f, t, s)) : g
end

function _ilaplace_cmd(F, s, t)
    F, s, t = convert(GiacExpr, F), convert(GiacExpr, s), convert(GiacExpr, t)
    (_node(t) isa SymNode && _node(s) isa SymNode) || return _giac_cmd(:ilaplace, F, s, t)
    r = _giac_or_nothing(:ilaplace, F, s, t)
    if _is_inverse_answer(r, s)
        _is_rational_in(_node(F), string(s)) && return r
        _laplace_verdict(r, t, F, s) !== :disagree && return r
    end
    g = _ilaplace_by_terms(F, s, t)
    return g === nothing ? _hold(:ilaplace, (F, s, t)) : g
end

function _giac_or_nothing(cmd::Symbol, args...)
    try
        return _giac_cmd(cmd, args...)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
end

const _TRANSFORM_HEADS = ("laplace", "ilaplace", "invlaplace", "integrate")
_has_transform_call(n::Node) = _any_node(m -> m isa CallNode && m.f in _TRANSFORM_HEADS, n)

# GIAC's transforms of these are taken as they are: sums of products of whole powers of t and
# exponentials, sines and cosines of a linear argument; and inverses of rational functions
_exponential_polynomial(n::Node, x::String) = !(x in _symbol_names(n))
_exponential_polynomial(n::SymNode, x::String) = true
_exponential_polynomial(n::AddNode, x::String) = all(((m, c),) -> _exponential_polynomial(m, x), n.terms)
_exponential_polynomial(n::MulNode, x::String) =
    all(((b, e),) -> !(x in _symbol_names(b)) && !(x in _symbol_names(e)) ||
                     (e isa NumNode && e.v isa Integer && e.v >= 0 ||
                      b isa CallNode && b.f == "exp" && !(x in _symbol_names(e))) && _exponential_polynomial(b, x), n.factors)
_exponential_polynomial(n::CallNode, x::String) =
    !(x in _symbol_names(n)) || n.f in ("exp", "sin", "cos", "sinh", "cosh") && length(n.args) == 1 && _linear_in(n.args[1], x)

_is_rational_in(n::Node, x::String) = !(x in _symbol_names(n))
_is_rational_in(n::SymNode, x::String) = true
_is_rational_in(n::AddNode, x::String) = all(((m, c),) -> _is_rational_in(m, x), n.terms)
_is_rational_in(n::MulNode, x::String) =
    all(((b, e),) -> !(x in _symbol_names(b)) && !(x in _symbol_names(e)) ||
                     e isa NumNode && e.v isa Integer && _is_rational_in(b, x), n.factors)

# a finite answer in s alone, and not 0 for a function that is not 0
function _is_laplace_answer(r, f::GiacExpr, t::GiacExpr)
    r === nothing && return false
    n = _node(r)
    (!_is_finite_value(r) || _has_transform_call(n) || string(t) in _free_names(r)) && return false
    return !(_is_zero_node(n) && !(f == 0))
end

_is_inverse_answer(r, s::GiacExpr) = r !== nothing && _is_finite_value(r) && !_has_transform_call(_node(r)) &&
                                     !(string(s) in _free_names(r))

# ---------------------------------------------------------------------------------------------
# Term by term
# ---------------------------------------------------------------------------------------------

# how deeply the theorems may call on one another: the transform of t Si(t) is the derivative of
# that of Si(t), which comes from that of Si'(t) = sin(t)/t
const _TRANSFORM_DEPTH = 4

_depends_on(f::GiacExpr, x::GiacExpr) = string(x) in _symbol_names(_node(f))

function _laplace_by_terms(f::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int = 0)
    depth > _TRANSFORM_DEPTH && return nothing
    parts = GiacExpr[]
    for m in _terms(_node(f))
        F = _laplace_term(GiacExpr(m), t, s, depth)
        F === nothing && return nothing
        push!(parts, F)
    end
    return _tidied(parts)
end

# the sum of the terms' transforms, over a common denominator where that is smaller
function _tidied(parts::Vector{GiacExpr})
    F = sum(parts)
    length(parts) > 1 || return F
    G = _invoke_cmd_direct(:normal, (F,))
    return Rules.expression_size(G) < Rules.expression_size(F) ? G : F
end

function _laplace_term(f::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    c, a, g = _exponential_factor(f, t)
    G = _laplace_core(g, t, s, depth)
    G === nothing && return nothing
    return c * (_is_zero_node(_node(a)) ? G : substitute(G, s => s - a))
end

# f as c e^(at) g: c the factors free of t, a the coefficient of t in the exponentials' exponents
function _exponential_factor(f::GiacExpr, x::GiacExpr)
    name = string(x)
    n = _node(f)
    factors = n isa MulNode ? n.factors : Pair{Node, Node}[n => _ONE]
    c, a, rest = n isa MulNode ? GiacExpr(NumNode(n.coef)) : one(GiacExpr), zero(GiacExpr), Node[]
    for (b, e) in factors
        factor = _pow(b, e)
        if !(name in _symbol_names(factor))
            c *= GiacExpr(factor)
        elseif b isa CallNode && b.f == "exp" && _linear_in(_mul(only(b.args), e), name)
            u = GiacExpr(_mul(only(b.args), e))
            k = GiacExpr(_derivative(_node(u), name))
            a += k
            c *= exp(substitute(u, x => 0))
        else
            push!(rest, factor)
        end
    end
    return c, a, isempty(rest) ? one(GiacExpr) : GiacExpr(_mul(rest...))
end

function _linear_in(u::Node, name::String)
    d = _derivative(u, name)
    return d !== nothing && !(name in _symbol_names(d))
end

function _laplace_core(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    r = _giac_or_nothing(:laplace, g, t, s)
    _is_laplace_answer(r, g, t) && _laplace_verdict(g, t, r, s) !== :disagree && return r
    for rule in (_laplace_pair, _laplace_bessel, _laplace_times_power, _laplace_over_t, _laplace_from_derivative, _laplace_integral)
        r = rule(g, t, s, depth)
        r !== nothing && _is_laplace_answer(r, g, t) && _laplace_verdict(g, t, r, s) !== :disagree && return r
    end
    return nothing
end

# L{tⁿh} = (-1)ⁿ dⁿH/dsⁿ, for a whole n > 0
function _laplace_times_power(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    n, h = _power_factor(g, t)
    (n isa Integer && n > 0 && _depends_on(h, t)) || return nothing
    H = _laplace_by_terms(h, t, s, depth + 1)
    H === nothing && return nothing
    for _ in 1:n
        H = -invoke_cmd(:diff, H, s)
    end
    return H
end

# L{h/t} = ∫ₛ^∞ H(u) du
function _laplace_over_t(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    n, h = _power_factor(g, t)
    (n == -1 && _depends_on(h, t)) || return nothing
    H = _laplace_by_terms(h, t, s, depth + 1)
    H === nothing && return nothing
    u = _fresh_symbol(_node(H), "u")
    return _integral_or_nothing(substitute(H, s => u), u, s, GiacExpr(_PLUS_INFINITY), s > 0)
end

# L{g} = (g(0⁺) + L{g'})/s, for g one function of t such as Si(t) or erf(√t)
function _laplace_from_derivative(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    _node(g) isa CallNode || return nothing
    g0 = _giac_or_nothing(:limit, g, t, 0, 1)
    (g0 === nothing || !_is_finite_value(g0)) && return nothing
    D = _laplace_by_terms(invoke_cmd(:diff, g, t), t, s, depth + 1)
    return D === nothing ? nothing : (g0 + D) / s
end

# ∫₀^∞ e^(-st) g(t) dt with s > 0, through the integration search
_laplace_integral(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int) =
    _integral_or_nothing(g * exp(-s * t), t, 0, GiacExpr(_PLUS_INFINITY), s > 0)

function _integral_or_nothing(f::GiacExpr, x::GiacExpr, a, b, condition::GiacExpr)
    r = try
        assuming(condition) do
            Commands.integrate(f, x, a, b)
        end
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (string(r) == "undef" || _has_transform_call(_node(r)) || !_is_finite_value(r)) && return nothing
    return r
end

_is_finite_value(r::GiacExpr) = string(r) != "undef" &&
    !_any_node(n -> n isa SymNode && n.name in ("inf", "infinity", "undef"), _node(r))

# g as xⁿh: the exponent n of the factor x in g, and the rest h
function _power_factor(g::GiacExpr, x::GiacExpr)
    n = _node(g)
    n isa SymNode && n.name == string(x) && return 1, one(GiacExpr)
    n isa MulNode || return 0, g
    for (b, e) in n.factors
        b isa SymNode && b.name == string(x) && e isa NumNode && e.v isa Union{Integer, Rational} &&
            return e.v, g / GiacExpr(_pow(b, e))
    end
    return 0, g
end

# g as xᵖ h(u), h a function of one argument u, or e^u: p, h's name and u
function _power_and_call(g::GiacExpr, x::GiacExpr)
    p, h = _power_factor(g, x)
    n = _node(h)
    if n isa MulNode && isone(n.coef) && length(n.factors) == 1
        b, e = only(n.factors)
        (b isa CallNode && b.f == "exp") || return nothing
        return p, "exp", GiacExpr(_mul(only(b.args), e))
    end
    (n isa CallNode && length(n.args) == 1) || return nothing
    return p, n.f, GiacExpr(only(n.args))
end

# u as kx, k√x, k/√x or k/x: :linear, :sqrt, :inv_sqrt or :inv, and k
function _argument_shape(u::GiacExpr, x::GiacExpr)
    for (shape, w) in ((:linear, x), (:sqrt, sqrt(x)), (:inv_sqrt, 1 / sqrt(x)), (:inv, 1 / x))
        k = u / w
        _depends_on(k, x) || return shape, k
    end
    return nothing
end

_gpi() = convert(GiacExpr, π)

# transforms of tᵖ h(k w(t)), keyed by h, the shape of w and p
const _LAPLACE_PAIRS = Dict{Tuple{String, Symbol, Rational{Int}}, Function}(
    ("sin", :sqrt, 0 // 1) => (k, s) -> k * sqrt(_gpi()) / (2 * s^(3 // 2)) * exp(-k^2 / (4s)),
    ("cos", :sqrt, -1 // 2) => (k, s) -> sqrt(_gpi()) / sqrt(s) * exp(-k^2 / (4s)),
    ("sinh", :sqrt, 0 // 1) => (k, s) -> k * sqrt(_gpi()) / (2 * s^(3 // 2)) * exp(k^2 / (4s)),
    ("cosh", :sqrt, -1 // 2) => (k, s) -> sqrt(_gpi()) / sqrt(s) * exp(k^2 / (4s)),
    ("exp", :inv, -1 // 2) => (k, s) -> sqrt(_gpi()) / sqrt(s) * exp(-2 * sqrt(-k * s)),
    ("exp", :inv, -3 // 2) => (k, s) -> sqrt(_gpi()) / sqrt(-k) * exp(-2 * sqrt(-k * s)),
    ("erfc", :inv_sqrt, 0 // 1) => (k, s) -> exp(-2k * sqrt(s)) / s,
    ("erf", :inv_sqrt, 0 // 1) => (k, s) -> (1 - exp(-2k * sqrt(s))) / s,
)

function _laplace_pair(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    m = _power_and_call(g, t)
    m === nothing && return nothing
    p, h, u = m
    shape = _argument_shape(u, t)
    shape === nothing && return nothing
    F = get(_LAPLACE_PAIRS, (h, shape[1], Rational{Int}(p)), nothing)
    return F === nothing ? nothing : F(shape[2], s)
end

# g as xᵖ f(ν, u) for a Bessel function f: p, f's name, ν and u
function _power_and_bessel(g::GiacExpr, x::GiacExpr)
    p, h = _power_factor(g, x)
    f = _special_name(_node(h))
    (f !== nothing && f in _BESSEL) || return nothing
    return p, f, GiacExpr(_node(h).args[2]), GiacExpr(_node(h).args[3])
end

# transforms of J_ν(kt) and I_ν(kt), alone, over t and times t^ν; of t^(ν/2) J_ν(k√t) and
# t^(ν/2) I_ν(k√t); and of K₀(kt) and Y₀(kt)
function _laplace_bessel(g::GiacExpr, t::GiacExpr, s::GiacExpr, depth::Int)
    m = _power_and_bessel(g, t)
    m === nothing && return nothing
    p, f, ν, u = m
    _depends_on(ν, t) && return nothing
    v = _node(ν)
    v isa NumNode && v.v isa Real && v.v <= -1 && return nothing
    shape = _argument_shape(u, t)
    shape === nothing && return nothing
    form, k = shape
    P = GiacExpr(NumNode(p))
    π_ = convert(GiacExpr, π)
    if form === :linear && f in ("besselj", "besseli")
        r = f == "besselj" ? sqrt(s^2 + k^2) : sqrt(s^2 - k^2)
        w = f == "besselj" ? r - s : s - r
        iszero(p) && return w^ν / (k^ν * r)
        p == -1 && return w^ν / (ν * k^ν)
        _is_zero_node(_node(P - ν)) && return (2k)^ν * invoke_cmd(:Gamma, ν + 1 // 2) / (sqrt(π_) * r^(2ν + 1))
    elseif form === :linear && iszero(p) && _is_zero_node(v) && f == "besselk"
        r = sqrt(s^2 - k^2)
        return log((s + r) / k) / r
    elseif form === :linear && iszero(p) && _is_zero_node(v) && f == "bessely"
        r = sqrt(s^2 + k^2)
        return -2 * asinh(s / k) / (π_ * r)
    elseif form === :sqrt && f in ("besselj", "besseli") && _is_zero_node(_node(P - ν / 2))
        return (k / 2)^ν * s^(-ν - 1) * exp((f == "besselj" ? -1 : 1) * k^2 / (4s))
    end
    return nothing
end

# inverses of (s² + c)^(-ν-1/2), in J_ν or I_ν by the sign of c, and of e^(k/s)/s^(ν+1); a
# quadratic s² + bs + c is (s + b/2)² + c - b²/4, its inverse that in s times e^(-bt/2)
function _ilaplace_bessel(G::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    n = _node(G)
    π_ = convert(GiacExpr, π)
    if n isa MulNode && isone(n.coef) && length(n.factors) == 1
        base, e = only(n.factors)
        if e isa NumNode && e.v isa Rational && e.v < 0 && denominator(e.v) == 2
            c = GiacExpr(base) - s^2
            if _depends_on(c, s)
                _linear_in(_node(c), string(s)) || return nothing
                b = GiacExpr(_derivative(_node(c), string(s)))
                h = _ilaplace_bessel((s^2 + substitute(c, s => 0) - b^2 / 4)^GiacExpr(e), s, t, depth)
                return h === nothing ? nothing : exp(-b * t / 2) * h
            end
            ν = GiacExpr(NumNode(-e.v - 1 // 2))
            positive = _positive_coefficient(c)
            k = sqrt(positive ? c : -c)
            f = positive ? SpecialFunctions.besselj : SpecialFunctions.besseli
            return sqrt(π_) * t^ν * f(ν, k * t) / ((2k)^ν * invoke_cmd(:Gamma, ν + 1 // 2))
        end
    end
    m = _power_and_call(G, s)
    (m === nothing || m[2] != "exp" || !(m[1] < 0)) && return nothing
    p, _, u = m
    shape = _argument_shape(u, s)
    (shape === nothing || shape[1] !== :inv) && return nothing
    k = shape[2]
    ν = GiacExpr(NumNode(-p - 1))
    positive = _positive_coefficient(k)
    a = positive ? k : -k
    f = positive ? SpecialFunctions.besseli : SpecialFunctions.besselj
    return (t / a)^(ν / 2) * f(ν, 2 * sqrt(a * t))
end

function _ilaplace_by_terms(F::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int = 0)
    depth > _TRANSFORM_DEPTH && return nothing
    parts = GiacExpr[]
    for m in _terms(_node(F))
        f = _ilaplace_term(GiacExpr(m), s, t, depth)
        f === nothing && return nothing
        push!(parts, f)
    end
    f = _tidied(parts)
    return _laplace_verdict(f, t, F, s) === :disagree ? nothing : f
end

# c e^(-bs) G(s), the delay b moving the inverse of G to t - b, from t = b on
function _ilaplace_term(F::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    c, nb, G = _exponential_factor(F, s)
    b = -nb
    g = _ilaplace_core(G, s, t, depth)
    g === nothing && return nothing
    _is_zero_node(_node(b)) && return c * g
    return c * substitute(g, t => t - b) * invoke_cmd(:Heaviside, t - b)
end

# the inverse as a power, from GIAC where it passes, from the table or from the theorems; the
# last are checked as a whole by the caller, as a term alone such as log(s + 1) has no inverse
function _ilaplace_core(G::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    g = _inverse_power(G, s, t)
    g === nothing || return g
    r = _giac_or_nothing(:ilaplace, G, s, t)
    _is_inverse_answer(r, s) && _laplace_verdict(r, t, G, s) !== :disagree && return r
    for rule in (_ilaplace_pair, _ilaplace_bessel, _ilaplace_over_s, _ilaplace_by_derivative, _ilaplace_convolution)
        r = rule(G, s, t, depth)
        r !== nothing && _is_inverse_answer(r, s) && return r
    end
    return nothing
end

# the inverse of (ks + a)^(-p), p > 0: t^(p-1) e^(-at/k) / (k^p Γ(p))
function _inverse_power(G::GiacExpr, s::GiacExpr, t::GiacExpr)
    n = _node(G)
    (n isa MulNode && isone(n.coef) && length(n.factors) == 1) || return nothing
    base, e = only(n.factors)
    (e isa NumNode && e.v isa Real && e.v < 0) || return nothing
    _linear_in(base, string(s)) || return nothing
    k = GiacExpr(_derivative(base, string(s)))
    a = substitute(GiacExpr(base), s => 0)
    p = GiacExpr(NumNode(-e.v))
    return t^(p - 1) * exp(-a * t / k) / (k^p * invoke_cmd(:Gamma, p))
end

# inverses of s^(-q) e^(k w(s)), keyed by the shape of w and q; e^(k/s) is written with cos and
# sin where k is negative, and with cosh and sinh where it is positive
function _inverse_pair(shape::Symbol, q::Rational{Int}, k::GiacExpr, t::GiacExpr)
    rπ = sqrt(_gpi())
    if shape === :sqrt
        q == 0 && return -k / (2rπ * t^(3 // 2)) * exp(-k^2 / (4t))
        q == 1 // 2 && return exp(-k^2 / (4t)) / (rπ * sqrt(t))
        q == 1 && return invoke_cmd(:erfc, -k / (2 * sqrt(t)))
    elseif shape === :inv
        positive = _positive_coefficient(k)
        a = positive ? k : -k
        w = 2 * sqrt(a * t)
        q == 1 // 2 && return (positive ? cosh(w) : cos(w)) / (rπ * sqrt(t))
        q == 3 // 2 && return (positive ? sinh(w) : sin(w)) / (rπ * sqrt(a))
    end
    return nothing
end

function _positive_coefficient(k::GiacExpr)
    n = _node(k)
    n isa NumNode && return n.v isa Real && n.v > 0
    n isa MulNode && return n.coef isa Real && n.coef > 0
    return n isa SymNode
end

function _ilaplace_pair(G::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    m = _power_and_call(G, s)
    (m === nothing || m[2] != "exp") && return nothing
    p, _, u = m
    shape = _argument_shape(u, s)
    shape === nothing && return nothing
    return _inverse_pair(shape[1], -Rational{Int}(p), shape[2], t)
end

# L⁻¹{H/s} = ∫₀ᵗ h
function _ilaplace_over_s(G::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    n, _ = _power_factor(G, s)
    (n isa Integer && n <= -1) || return nothing
    H = G * s
    _depends_on(H, s) || return nothing
    h = _ilaplace_by_terms(H, s, t, depth + 1)
    h === nothing && return nothing
    τ = _fresh_symbol(_node(h), "tau")
    return _integral_or_nothing(substitute(h, t => τ), τ, 0, t, t > 0)
end

# L⁻¹{G} = -L⁻¹{G'}/t, for G one function of s such as atan(1/s) or log((s+1)/(s+2))
function _ilaplace_by_derivative(G::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    _node(G) isa CallNode || return nothing
    h = _ilaplace_by_terms(invoke_cmd(:diff, G, s), s, t, depth + 1)
    return h === nothing ? nothing : -h / t
end

# L⁻¹{AB} = ∫₀ᵗ a(τ) b(t-τ) dτ, A the first factor of G in s
function _ilaplace_convolution(G::GiacExpr, s::GiacExpr, t::GiacExpr, depth::Int)
    n = _node(G)
    n isa MulNode || return nothing
    factors = [GiacExpr(_pow(b, e)) for (b, e) in n.factors]
    filter!(A -> _depends_on(A, s), factors)
    length(factors) >= 2 || return nothing
    A = first(factors)
    a = _ilaplace_by_terms(A, s, t, depth + 1)
    a === nothing && return nothing
    b = _ilaplace_by_terms(G / A, s, t, depth + 1)
    b === nothing && return nothing
    τ = _fresh_symbol(_mul(_node(a), _node(b)), "tau")
    return _integral_or_nothing(substitute(a, t => τ) * substitute(b, t => t - τ), τ, 0, t, t > 0)
end

# ---------------------------------------------------------------------------------------------
# The numeric check
# ---------------------------------------------------------------------------------------------

# :agree, :disagree or :unknown: whether F(s) is the transform of f(t), from the defining integral
# at two points, with the other names given sample values
function _laplace_verdict(f::GiacExpr, t::GiacExpr, F::GiacExpr, s::GiacExpr)
    names = setdiff(union(_free_names(f), _free_names(F)), (string(t), string(s)))
    values = [giac_identifier(n) => _sample_value(n, i) for (i, n) in enumerate(sort!(collect(names)))]
    fp = isempty(values) ? f : substitute(f, values...)
    Fp = isempty(values) ? F : substitute(F, values...)
    _any_node(m -> m isa CallNode && m.f == "Dirac", _node(fp)) && return :unknown
    for scale in (1, 4, 16)
        verdicts = Symbol[]
        for s0 in (2.5scale, 3.7scale)
            q = _numeric_laplace(fp, t, s0)
            q === :diverges && break
            q === nothing && return :unknown
            v = _approx_value(substitute(Fp, s => s0))
            v === nothing && return :unknown
            d, m = abs(v - q.value), max(abs(q.value), abs(v), 1e-300)
            push!(verdicts, d <= 1e-6m + 10q.err ? :agree : d > 1e-3m + 100q.err ? :disagree : :unknown)
        end
        length(verdicts) == 2 || continue
        :disagree in verdicts && return :disagree
        return all(==(:agree), verdicts) ? :agree : :unknown
    end
    return :unknown
end

# generic values for the other names, with the sign or integrality they are declared with
const _SAMPLES = (0.6180339887498949, 0.7071067811865476, 1.324717957244746, 0.41421356237309503, 1.2599210498948732)
function _sample_value(name::AbstractString, i::Int)
    tag = _declared_tag(name)
    tag === :integer && return i + 1
    v = _SAMPLES[mod1(i, length(_SAMPLES))]
    tag in (:negative, :nonpositive) && return -v
    tag isa Tuple && length(tag) == 2 && all(x -> x isa Real && isfinite(x), tag) && return (tag[1] + tag[2]) / 2
    return v
end

# ∫₀^∞ e^(-s₀t) f(t) dt, split where a Heaviside step in f switches on: tanh-sinh quadrature
# between the steps, and exp-sinh beyond the last one with f shifted to start there. :diverges
# where the integrand has not died away at the far end, nothing where f has no numeric values
function _numeric_laplace(f::GiacExpr, t::GiacExpr, s0::Real)
    steps = _steps(f, t)
    isempty(steps) && return _exp_sinh_laplace(f, t, s0)
    value, err, a = 0.0, 0.0, 0.0
    for (_, b) in steps
        q = _tanh_sinh_laplace(f, t, s0, a, b)
        q === nothing && return nothing
        value, err, a = value + q.value, err + q.err, b
    end
    q = _exp_sinh_laplace(GiacExpr(_node(substitute(f, t => t + steps[end][1]))), t, s0)
    q isa NamedTuple || return q
    w = exp(-s0 * a)
    return (value = value + w * q.value, err = err + w * q.err)
end

# the points b > 0 where a step Heaviside(t - b) in f switches on, as expressions and values
function _steps(f::GiacExpr, t::GiacExpr)
    name = string(t)
    steps = Tuple{GiacExpr, Float64}[]
    _any_node(_node(f)) do n
        if n isa CallNode && n.f == "Heaviside" && length(n.args) == 1 && _linear_in(only(n.args), name)
            u = GiacExpr(only(n.args))
            b = -substitute(u, t => 0) / GiacExpr(_derivative(only(n.args), name))
            v = _approx_value(b)
            v isa Real && v > 0 && !any(x -> isapprox(x[2], v), steps) && push!(steps, (b, v))
        end
        return false
    end
    return sort!(steps; by = last)
end

# ∫ₐᵇ e^(-s₀t) f(t) dt by tanh-sinh quadrature, with steps 1/32 and 1/16 for its error
function _tanh_sinh_laplace(f::GiacExpr, t::GiacExpr, s0::Real, a::Real, b::Real)
    h = 1 / 32
    us = collect(-3.0:h:3.0)
    xs = tanh.(π / 2 .* sinh.(us))
    ts = (a + b) / 2 .+ (b - a) / 2 .* xs
    vals = _values_at(f, t, ts)
    vals === nothing && return nothing
    g = vals .* exp.(-s0 .* ts) .* ((b - a) / 2 .* π / 2 .* cosh.(us) ./ cosh.(π / 2 .* sinh.(us)) .^ 2)
    all(isfinite, g) || return nothing
    q = h * sum(g)
    q2 = 2h * sum(g[1:2:end])
    return (value = q, err = abs(q - q2))
end

# ∫₀^∞ e^(-s₀t) f(t) dt by exp-sinh quadrature, t = exp(π/2 sinh u), with steps 1/32 and 1/16 for
# its error
function _exp_sinh_laplace(f::GiacExpr, t::GiacExpr, s0::Real)
    h = 1 / 32
    umax = asinh(2 / π * log(46 / s0))
    us = collect(-4.5:h:max(umax, -4.0))
    ts = exp.(π / 2 .* sinh.(us))
    vals = _values_at(f, t, ts)
    vals === nothing && return nothing
    g = vals .* exp.(-s0 .* ts) .* (π / 2 .* cosh.(us) .* ts)
    all(isfinite, g) || return nothing
    total = sum(abs, g)
    maximum(abs, g[max(1, end - 3):end]) > 1e-10 * max(total, 1e-300) && return :diverges
    q = h * sum(g)
    q2 = 2h * sum(g[1:2:end])
    return (value = q, err = abs(q - q2))
end

# f at each of the points, from one GIAC call, or compiled where it has functions GIAC has no
# values for
function _values_at(f::GiacExpr, t::GiacExpr, ts::Vector{Float64})
    _has_special(f) && return _compiled_values_at(f, t, ts)
    v = try
        to_julia(invoke_cmd(:evalf, invoke_cmd(:map, giac_eval("[" * join(repr.(ts), ",") * "]"), invoke_cmd(:unapply, f, t))))
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (v isa AbstractVector && length(v) == length(ts) && all(x -> x isa Number, v)) || return nothing
    return any(x -> x isa Complex, v) ? ComplexF64.(v) : Float64.(v)
end

function _compiled_values_at(f::GiacExpr, t::GiacExpr, ts::Vector{Float64})
    g = try
        _compile(f, (t,))
    catch e
        e isa ArgumentError || rethrow()
        return nothing
    end
    v = map(ts) do x
        try
            g(x)
        catch e
            e isa DomainError || rethrow()
            g(complex(x))
        end
    end
    all(x -> x isa Number, v) || return nothing
    return any(x -> x isa Complex, v) ? ComplexF64.(v) : Float64.(v)
end
