# Definite integrals as derivatives of known integrals in their parameters. Differentiating xˢ in s
# brings down ln x, so a term
#
#     c xᵅ (1−x)ᵝ (ln x)ᵐ (ln(1−x))ⁿ  on [0, 1]  is  c ∂ᵐ/∂sᵐ ∂ⁿ/∂rⁿ B(s+1, r+1)  at s = α, r = β,
#
# and likewise for the Gamma integral of xᵅ e^(−kx) (ln x)ᵐ and the Beta integral of the second
# kind of xᵅ (1+x)^(−γ) (ln x)ᵐ (ln(1+x))ⁿ, both on [0, ∞), and for sinᵅx cosᵝx (ln sin x)ᵐ
# (ln cos x)ⁿ on [0, π/2], half a Beta function of (α+1)/2 and (β+1)/2. Each term must converge, as decided
# from its exponents; where they leave a parameter at a pole of the closed form, the value is the
# limit there. Used for the definite integrals GIAC leaves unevaluated.

# a term matched to a family: factors free of x, the exponents of x, 1−x, 1+x, sin x and cos x,
# the powers of their logarithms, and the rate k of e^(−kx)
struct _ParametricTerm
    coef::Node
    exponent::Dict{Symbol, Node}
    logs::Dict{Symbol, Int}
    rate::Node
end

# :x, :one_minus, :one_plus, :sin or :cos for x, 1−x, 1+x, sin x, cos x; :minus_one for x−1
function _base_kind(b::Node, x::SymNode)
    isequal(b, x) && return :x
    isequal(b, _sub(_ONE, x)) && return :one_minus
    isequal(b, _add(_ONE, x)) && return :one_plus
    isequal(b, _sub(x, _ONE)) && return :minus_one
    b isa CallNode && b.f in ("sin", "cos") && length(b.args) == 1 && isequal(b.args[1], x) && return Symbol(b.f)
    return nothing
end

const _PARAMETRIC_BASES = (:x, :one_minus, :one_plus, :sin, :cos)

_free_of(n::Node, x::SymNode) = !(x.name in _symbol_names(n))

function _match_parametric(t::Node, x::SymNode)
    m = t isa MulNode ? t : MulNode(1, Pair{Node, Node}[t => _ONE])
    coef = Node[NumNode(m.coef)]
    exponent = Dict{Symbol, Node}(k => _ZERO for k in _PARAMETRIC_BASES)
    logs = Dict(k => 0 for k in _PARAMETRIC_BASES)
    rate = _ZERO
    for (b, e) in m.factors
        if _free_of(b, x) && _free_of(e, x)
            push!(coef, _pow(b, e))
            continue
        end
        _free_of(e, x) || return nothing
        kind = _base_kind(b, x)
        if kind === :minus_one
            # (x−1)ᵏ = (−1)ᵏ (1−x)ᵏ for a whole power k
            (e isa NumNode && e.v isa Integer) || return nothing
            isodd(e.v) && push!(coef, _MINUS_ONE)
            kind = :one_minus
        end
        if kind !== nothing
            exponent[kind] = _add(exponent[kind], e)
        elseif b isa CallNode && b.f == "ln" && length(b.args) == 1 && e isa NumNode && e.v isa Integer && e.v > 0
            kind = _base_kind(b.args[1], x)
            (kind === nothing || kind === :minus_one) && return nothing
            logs[kind] += e.v
        elseif b isa CallNode && b.f == "exp" && length(b.args) == 1
            # e^(c − kx), to a power free of x
            u = b.args[1]
            d = _derivative(u, x.name)
            (d === nothing || !_free_of(d, x)) && return nothing
            c = _sub(u, _mul(d, x))
            _free_of(c, x) || return nothing
            push!(coef, _pow(_exp(c), e))
            rate = _sub(rate, _mul(d, e))
        else
            return nothing
        end
    end
    return _ParametricTerm(_mul(coef...), exponent, logs, rate)
end

_proved_positive(n::Node) = n isa NumNode ? (n.v isa Real && n.v > 0) : sign_of(GiacExpr(n); sos = false) === :positive

# `(closed form, [(parameter, value, order)], sign, converges, pole)` for a term on the bounds
# `kind` (:unit for [0, 1], :half_line for [0, ∞), :quarter for [0, π/2]), or `nothing`. The closed form is in the
# parameters `s`, `r`; `pole` says whether a value sits at a pole of it.
function _parametric_family(p::_ParametricTerm, kind::Symbol, s::Node, r::Node)
    α, β, γ = p.exponent[:x], p.exponent[:one_minus], p.exponent[:one_plus]
    m, n, l = p.logs[:x], p.logs[:one_minus], p.logs[:one_plus]
    gt(a, b) = _proved_positive(_sub(a, b))
    Γ(z) = _call("Gamma", Node[z])
    trig = !isequal(p.exponent[:sin], _ZERO) || !isequal(p.exponent[:cos], _ZERO) || p.logs[:sin] + p.logs[:cos] > 0
    if kind === :quarter
        (trig && all(k -> isequal(p.exponent[k], _ZERO) && p.logs[k] == 0, (:x, :one_minus, :one_plus)) &&
         isequal(p.rate, _ZERO)) || return nothing
        α, β, m, n = p.exponent[:sin], p.exponent[:cos], p.logs[:sin], p.logs[:cos]
        # near 0 like x^(α+2n) (ln x)ᵐ, as ln cos x ≈ −x²/2; near π/2 likewise
        converges = gt(_add(α, NumNode(2n)), _MINUS_ONE) && gt(_add(β, NumNode(2m)), _MINUS_ONE)
        pole = !(gt(α, _MINUS_ONE) && gt(β, _MINUS_ONE))
        form = _div(_mul(Γ(_mul(_HALF, _add(s, _ONE))), Γ(_mul(_HALF, _add(r, _ONE)))),
                    _mul(NumNode(2), Γ(_add(_mul(_HALF, _add(s, r)), _ONE))))
        return form, [(s, α, m), (r, β, n)], 1, converges, pole
    end
    trig && return nothing
    if kind === :unit && isequal(γ, _ZERO) && l == 0 && isequal(p.rate, _ZERO)
        # near 0 the integrand is like x^(α+n) (ln x)ᵐ, as ln(1−x) ≈ −x; near 1 likewise
        converges = gt(_add(α, NumNode(n)), _MINUS_ONE) && gt(_add(β, NumNode(m)), _MINUS_ONE)
        pole = !(gt(α, _MINUS_ONE) && gt(β, _MINUS_ONE))
        form = _div(_mul(Γ(_add(s, _ONE)), Γ(_add(r, _ONE))), Γ(_add(s, r, NumNode(2))))
        return form, [(s, α, m), (r, β, n)], 1, converges, pole
    elseif kind === :half_line && isequal(β, _ZERO) && n == 0 && isequal(γ, _ZERO) && l == 0 && !isequal(p.rate, _ZERO)
        converges = gt(α, _MINUS_ONE) && _proved_positive(p.rate)
        form = _mul(Γ(_add(s, _ONE)), _pow(p.rate, _neg(_add(s, _ONE))))
        return form, [(s, α, m)], 1, converges, false
    elseif kind === :half_line && isequal(β, _ZERO) && n == 0 && isequal(p.rate, _ZERO)
        # xᵅ (1+x)^(−g): near 0 like x^(α+l), as ln(1+x) ≈ x; near ∞ like x^(α−g) times logarithms
        g = _neg(γ)
        converges = gt(_add(α, NumNode(l)), _MINUS_ONE) && gt(_sub(g, α), _ONE)
        pole = !gt(α, _MINUS_ONE)
        form = _div(_mul(Γ(_add(s, _ONE)), Γ(_sub(r, _add(s, _ONE)))), Γ(r))
        # ∂/∂g of (1+x)^(−g) is −ln(1+x)
        return form, [(s, α, m), (r, g, l)], iseven(l) ? 1 : -1, converges, pole
    end
    return nothing
end

# ∫ₐᵇ f dx for the bounds [0, 1], [0, ∞) or [0, π/2], term by term, or `nothing`
function _parametric_integral(f::GiacExpr, x::GiacExpr, a, b)
    xn = _node(x)
    xn isa SymNode || return nothing
    lo, hi = _node(convert(GiacExpr, a)), _node(convert(GiacExpr, b))
    isequal(lo, _ZERO) || return nothing
    kind = isequal(hi, _ONE) ? :unit : _is_plus_infinity(hi) ? :half_line :
           isequal(hi, _mul(_HALF, SymNode("pi"))) ? :quarter : nothing
    kind === nothing && return nothing
    n = _node(f)
    for form in (n, _expand(n))
        v = _parametric_terms(form, xn, kind)
        v === nothing || return v
    end
    return nothing
end

function _parametric_terms(n::Node, x::SymNode, kind::Symbol)
    terms = n isa AddNode ? [_scale(m, c) for (m, c) in n.terms] : [n]
    taken = union(_symbol_names(n), Set([x.name]))
    fresh(name) = (while name in taken; name *= "_"; end; SymNode(name))
    s, r = fresh("param__s"), fresh("param__r")
    total = zero(GiacExpr)
    for t in terms
        p = _match_parametric(t, x)
        p === nothing && return nothing
        fam = _parametric_family(p, kind, s, r)
        fam === nothing && return nothing
        form, params, sign, converges, pole = fam
        converges || return nothing
        D = GiacExpr(form)
        for (q, _, k) in params
            k > 0 && (D = invoke_cmd(:diff, D, GiacExpr(q), k))
        end
        for (q, v, _) in params
            D = pole ? invoke_cmd(:limit, D, GiacExpr(q), GiacExpr(v)) : substitute(D, GiacExpr(q) => GiacExpr(v))
        end
        total += sign * GiacExpr(p.coef) * D
    end
    return invoke_cmd(:simplify, total)
end
