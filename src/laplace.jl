# Laplace transforms. GIAC's `laplace` returns 0 for some transforms it does not find (√t, 1/√t),
# and writes erf where erfc belongs (e^(-t²), erf t); its `ilaplace` returns a transform it cannot
# invert times Dirac(t). So each answer is checked against the integral that defines it,
#
#     F(s₀) = ∫₀^∞ e^(-s₀t) f(t) dt,
#
# computed numerically at two points s₀ (an exp-sinh quadrature, with f evaluated by GIAC in one
# call). Where GIAC has no answer or a wrong one, the transform is found a term at a time: factors
# free of t come out, a factor e^(at) shifts the transform to s - a, and the rest is GIAC's
# transform where it passes, else the integral itself with s > 0. An inverse is found a term at a
# time from powers (s + a)^(-p) and delays e^(-bs), or GIAC's inverse of the term where it passes.
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

# an answer in s alone, and not 0 for a function that is not 0
function _is_laplace_answer(r, f::GiacExpr, t::GiacExpr)
    r === nothing && return false
    n = _node(r)
    (string(r) == "undef" || _has_transform_call(n) || string(t) in _free_names(r)) && return false
    return !(_is_zero_node(n) && !(f == 0))
end

_is_inverse_answer(r, s::GiacExpr) = r !== nothing && string(r) != "undef" && !_has_transform_call(_node(r)) &&
                                     !(string(s) in _free_names(r))

# ---------------------------------------------------------------------------------------------
# Term by term
# ---------------------------------------------------------------------------------------------

_terms(n::AddNode) = Node[_scale(m, c) for (m, c) in n.terms]
_terms(n::Node) = Node[n]

function _laplace_by_terms(f::GiacExpr, t::GiacExpr, s::GiacExpr)
    parts = GiacExpr[]
    for m in _terms(_node(f))
        F = _laplace_term(GiacExpr(m), t, s)
        F === nothing && return nothing
        push!(parts, F)
    end
    return sum(parts)
end

function _laplace_term(f::GiacExpr, t::GiacExpr, s::GiacExpr)
    c, a, g = _exponential_factor(f, t)
    G = _laplace_core(g, t, s)
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

function _laplace_core(g::GiacExpr, t::GiacExpr, s::GiacExpr)
    r = _giac_or_nothing(:laplace, g, t, s)
    _is_laplace_answer(r, g, t) && _laplace_verdict(g, t, r, s) !== :disagree && return r
    r = _laplace_integral(g, t, s)
    return r !== nothing && _laplace_verdict(g, t, r, s) !== :disagree ? r : nothing
end

# ∫₀^∞ e^(-st) g(t) dt with s > 0, through the integration search
function _laplace_integral(g::GiacExpr, t::GiacExpr, s::GiacExpr)
    r = try
        assuming(s > 0) do
            Commands.integrate(g * exp(-s * t), t, 0, GiacExpr(_PLUS_INFINITY))
        end
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    return _is_laplace_answer(r, g, t) ? r : nothing
end

function _ilaplace_by_terms(F::GiacExpr, s::GiacExpr, t::GiacExpr)
    parts = GiacExpr[]
    for m in _terms(_node(F))
        f = _ilaplace_term(GiacExpr(m), s, t)
        f === nothing && return nothing
        push!(parts, f)
    end
    f = sum(parts)
    return _laplace_verdict(f, t, F, s) === :disagree ? nothing : f
end

# c e^(-bs) G(s): G a power (ks + a)^(-p), or GIAC's inverse of G where it passes; the delay b
# moves the inverse to t - b, from t = b on
function _ilaplace_term(F::GiacExpr, s::GiacExpr, t::GiacExpr)
    c, nb, G = _exponential_factor(F, s)
    b = -nb
    g = _inverse_power(G, s, t)
    if g === nothing
        r = _giac_or_nothing(:ilaplace, G, s, t)
        (_is_inverse_answer(r, s) && _laplace_verdict(r, t, G, s) !== :disagree) || return nothing
        g = r
    end
    _is_zero_node(_node(b)) && return c * g
    return c * substitute(g, t => t - b) * invoke_cmd(:Heaviside, t - b)
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

# ∫₀^∞ e^(-s₀t) f(t) dt by exp-sinh quadrature, t = exp(π/2 sinh u), with steps 1/32 and 1/16 for
# its error; :diverges where the integrand has not died away at the far end, nothing where f has
# no numeric values
function _numeric_laplace(f::GiacExpr, t::GiacExpr, s0::Real)
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

# f at each of the points, from one GIAC call
function _values_at(f::GiacExpr, t::GiacExpr, ts::Vector{Float64})
    v = try
        to_julia(invoke_cmd(:evalf, invoke_cmd(:map, giac_eval("[" * join(repr.(ts), ",") * "]"), invoke_cmd(:unapply, f, t))))
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    (v isa AbstractVector && length(v) == length(ts) && all(x -> x isa Number, v)) || return nothing
    return any(x -> x isa Complex, v) ? ComplexF64.(v) : Float64.(v)
end
