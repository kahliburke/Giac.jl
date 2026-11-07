# Sums and products GIAC leaves unevaluated. An infinite sum or product is the limit of the
# partial one where GIAC has that in closed form, and an alternating sum is taken as a sum of
# pairs of terms. GIAC writes sums of rational functions with polygamma values; those at z and
# 1−z are combined by the reflection formula
#
#     ψ⁽ᵐ⁾(1−z) = (−1)ᵐ (ψ⁽ᵐ⁾(z) + π dᵐ/dzᵐ cot πz).

_giac_cmd(cmd::Symbol, args...) = _invoke_cmd_direct(cmd, map(a -> convert(GiacExpr, a), args))

# GIAC's answer still holds the command, or is undef
_unevaluated(r::GiacExpr, f::String) = string(r) == "undef" || _any_node(n -> n isa CallNode && n.f == f, _node(r))
_is_infinity(b) = _is_plus_infinity(_node(convert(GiacExpr, b)))

# a symbol named after `base` that does not occur in `n`
function _fresh_symbol(n::Node, base::String)
    taken = _symbol_names(n)
    while base in taken
        base *= "_"
    end
    return GiacExpr(SymNode(base))
end

# `invoke_cmd(:sum, f, n, a, b)`
function _sum_cmd(f, n, a, b)
    r = _giac_cmd(:sum, f, n, a, b)
    _unevaluated(r, "sum") || return _reflect_polygamma(r)
    v = _chu_vandermonde(convert(GiacExpr, f), n, a, b)
    v === nothing || return v
    if _is_infinity(b)
        v = _alternating_sum(convert(GiacExpr, f), n, a)
        v === nothing || return v
        v = _limit_of_partial(:sum, f, n, a)
        v === nothing || return v
    end
    return r
end

# `invoke_cmd(:product, f, n, a, b)`
function _product_cmd(f, n, a, b)
    r = _giac_cmd(:product, f, n, a, b)
    (_unevaluated(r, "product") && _is_infinity(b)) || return r
    v = _limit_of_partial(:product, f, n, a)
    return v === nothing ? r : v
end

# the limit of the partial sum or product up to N, when GIAC has that in closed form
function _limit_of_partial(cmd::Symbol, f, n, a)
    N = _fresh_symbol(_node(convert(GiacExpr, f)), "N")
    partial = _giac_cmd(cmd, f, n, a, N)
    _unevaluated(partial, string(cmd)) && return nothing
    v = invoke_cmd(:limit, partial, N, giac_eval("+infinity"))
    _unevaluated(v, "limit") && return nothing
    return _reflect_polygamma(invoke_cmd(:simplify, v))
end

# Σₙ₌ₐ^∞ (−1)ⁿ⁺ᶜ g(n) as (−1)ᵃ⁺ᶜ Σₘ₌₀^∞ (g(a+2m) − g(a+2m+1)), for a whole a and c
function _alternating_sum(f::GiacExpr, n::GiacExpr, a)
    nn, an = _node(n), _node(convert(GiacExpr, a))
    (nn isa SymNode && an isa NumNode && an.v isa Integer) || return nothing
    t = _node(f)
    t isa MulNode || return nothing
    i = findfirst(((b, e),) -> isequal(b, _MINUS_ONE) && (c = _sub(e, nn); c isa NumNode && c.v isa Integer), t.factors)
    i === nothing && return nothing
    c = _sub(t.factors[i][2], nn).v
    g = GiacExpr(MulNode(t.coef, t.factors[[j for j in eachindex(t.factors) if j != i]]))
    m = _fresh_symbol(t, "m")
    pair = substitute(g, n => a + 2m) - substitute(g, n => a + 2m + 1)
    s = _giac_cmd(:sum, invoke_cmd(:normal, pair), m, 0, giac_eval("+infinity"))
    (_unevaluated(s, "sum") || _is_infinite(s)) && return nothing
    return _reflect_polygamma(invoke_cmd(:simplify, (-1)^(an.v + c) * s))
end

# the polygamma values ψ⁽ᵐ⁾(z) at rational z: (z, m) for GIAC's Psi(z, m)
_polygamma_point(n::CallNode) =
    n.f == "Psi" && length(n.args) == 2 && n.args[1] isa NumNode && n.args[1].v isa Rational &&
    n.args[2] isa NumNode && n.args[2].v isa Integer ? (n.args[1].v, Int(n.args[2].v)) : nothing
_polygamma_point(n::Node) = nothing

function _polygamma_points!(s, n::Node)
    _any_node(n) do m
        p = _polygamma_point(m)
        p === nothing || push!(s, p)
        false
    end
    return s
end

# `e` with ψ⁽ᵐ⁾(1−z) written through ψ⁽ᵐ⁾(z) wherever that takes both out
function _reflect_polygamma(e::GiacExpr)
    points = _polygamma_points!(Set{Tuple{Rational, Int}}(), _node(e))
    for (z, m) in points
        z < 1 // 2 && (1 - z, m) in points || continue
        w = _fresh_symbol(_node(e), "w")
        Dcot = substitute(invoke_cmd(:diff, cot(giac_eval("pi") * w), w, m), w => GiacExpr(NumNode(z)))
        reflected = (-1)^m * (GiacExpr(CallNode("Psi", Node[NumNode(z), NumNode(m)])) + giac_eval("pi") * Dcot)
        target = CallNode("Psi", Node[NumNode(1 - z), NumNode(m)])
        r = invoke_cmd(:simplify, GiacExpr(_replace_node(_node(e), target, _node(reflected))))
        if !(_any_node(n -> _polygamma_point(n) in ((z, m), (1 - z, m)), _node(r)))
            e = r
            points = _polygamma_points!(Set{Tuple{Rational, Int}}(), _node(e))
        end
    end
    return e
end

# `n` with each occurrence of the subtree `old` replaced by `new`
function _replace_node(n::Node, old::Node, new::Node)
    isequal(n, old) && return new
    if n isa CallNode
        return _call(n.f, Node[_replace_node(a, old, new) for a in n.args])
    elseif n isa AddNode
        return _add((_scale(_replace_node(m, old, new), c) for (m, c) in n.terms)...)
    elseif n isa MulNode
        return _mul(NumNode(n.coef), (_pow(_replace_node(b, old, new), _replace_node(e, old, new)) for (b, e) in n.factors)...)
    end
    return n
end

"""
    hypergeometric(a, b, z)

The generalized hypergeometric function ₚF_q(a; b; z) = Σₖ (a₁)ₖ⋯(aₚ)ₖ / ((b₁)ₖ⋯(b_q)ₖ) zᵏ/k!, in
closed form where the sum has one: a polynomial when an upper parameter is a whole number ≤ 0,
and otherwise GIAC's sum of the series when each upper parameter pairs with a lower one (or with
the 1 of k!) at a whole-number distance and the lower ones left are 1: the coefficients are then
a rational function of k, over a power of k!.

```julia
@giac_var x
Giac.hypergeometric([1, 1], [2], x)     # -ln(abs(1 - x))/x
Giac.hypergeometric([-2, 1], [1], x)    # 1 - 2x + x^2
Giac.hypergeometric([1], [1], x)        # exp(x)
```
"""
function hypergeometric(a::AbstractVector, b::AbstractVector, z)
    A = Node[_node(convert(GiacExpr, v)) for v in a]
    B = Node[_node(convert(GiacExpr, v)) for v in b]
    Z = convert(GiacExpr, z)
    # a whole number ≤ 0 above ends the series
    m = findfirst(v -> v isa NumNode && v.v isa Integer && v.v <= 0, A)
    m === nothing || return _terminating_hypergeometric(A, B, Z, -Int(A[m].v))
    k = _fresh_symbol(_node(Z), "k")
    R = _pochhammer_ratio(A, [B; _ONE], _node(k))
    R === nothing && throw(ArgumentError("hypergeometric: the parameters $a and $b do not pair at whole-number distances"))
    return invoke_cmd(:simplify, invoke_cmd(:sum, GiacExpr(R) * Z^k, k, 0, giac_eval("+infinity")))
end

function _terminating_hypergeometric(A, B, Z, m)
    total, t = one(GiacExpr), one(GiacExpr)
    for k in 0:(m - 1)
        K = NumNode(k)
        t *= GiacExpr(_div(_mul((_add(v, K) for v in A)...), _mul((_add(v, K) for v in B)...))) * Z / (k + 1)
        total += t
    end
    return invoke_cmd(:simplify, total)
end

# Π (aᵢ)ₖ / Π (bⱼ)ₖ as a rational function of k over a power of k!, each aᵢ taken with a bⱼ at a
# whole distance d: (a)ₖ/(a+d)ₖ = (a)_d/(a+k)_d, and (1)ₖ = k!; nothing when they do not pair up
function _pochhammer_ratio(A, B, k::Node)
    lower = collect(B)
    R = _ONE
    for a in A
        i = findfirst(b -> (d = _sub(b, a); d isa NumNode && d.v isa Integer), lower)
        i === nothing && return nothing
        d = Int(_sub(lower[i], a).v)
        deleteat!(lower, i)
        for j in 0:(abs(d) - 1)
            num, den = _add(a, NumNode(j)), _add(a, k, NumNode(j))
            if d < 0         # (a)ₖ/(a−e)ₖ = (a−e+k)ₑ/(a−e)ₑ
                num, den = _add(a, k, NumNode(d + j)), _add(a, NumNode(d + j))
            end
            R = _mul(R, _div(num, den))
        end
    end
    all(b -> isequal(b, _ONE), lower) || return nothing
    return _mul(R, _pow(_call("factorial", Node[k]), NumNode(-length(lower))))
end

# Σₖ₌₀ᵐ t(k) for a hypergeometric term whose ratio is t(k+1)/t(k) = (k−m)(k+b)/((k+c)(k+1)): t(0)
# times ₂F₁(−m, b; c; 1) = (c−b)ₘ/(c)ₘ (Chu–Vandermonde); nothing for any other sum
function _chu_vandermonde(t::GiacExpr, k::GiacExpr, lo, hi)
    (_is_zero_expr(convert(GiacExpr, lo)) && !_is_infinity(hi)) || return nothing
    m = _node(convert(GiacExpr, hi))
    params = _hypergeometric_parameters(t, k)
    params === nothing && return nothing
    A, B, z = params
    (length(A) == 2 && length(B) == 2 && isequal(z, _ONE)) || return nothing
    i = findfirst(b -> isequal(b, _ONE), B)
    j = findfirst(a -> isequal(a, _neg(m)), A)
    (i === nothing || j === nothing) && return nothing
    b, c = A[3 - j], B[3 - i]
    Γ(x) = _call("Gamma", Node[x])
    ratio = _div(_mul(Γ(_add(c, _neg(b), m)), Γ(c)), _mul(Γ(_sub(c, b)), Γ(_add(c, m))))
    return invoke_cmd(:simplify, substitute(t, k => 0) * GiacExpr(ratio))
end

# (upper parameters, lower parameters, z) of the hypergeometric term t in k: its ratio
# t(k+1)/t(k) factored as z Π(k+aᵢ) / Π(k+bⱼ); nothing when the ratio is not of that form
function _hypergeometric_parameters(t::GiacExpr, k::GiacExpr)
    ratio = invoke_cmd(:simplify, substitute(t, k => k + 1) / t)
    num, den = invoke_cmd(:numer, ratio), invoke_cmd(:denom, ratio)
    kn = _node(k)
    A, ca = _linear_factors(invoke_cmd(:factor, num), kn)
    B, cb = _linear_factors(invoke_cmd(:factor, den), kn)
    (A === nothing || B === nothing) && return nothing
    return A, B, _node(invoke_cmd(:simplify, GiacExpr(_div(ca, cb))))
end

# the roots −aᵢ of a product of factors linear in k, as the aᵢ with multiplicity, and the factor
# free of k; nothing if a factor is not linear in k
function _linear_factors(p::GiacExpr, k::Node)
    n = _node(p)
    factors = n isa MulNode ? n.factors : Pair{Node, Node}[n => _ONE]
    coef = n isa MulNode ? Node[NumNode(n.coef)] : Node[]
    params = Node[]
    for (b, e) in factors
        if !(k.name in _symbol_names(b))
            push!(coef, _pow(b, e))
            continue
        end
        (e isa NumNode && e.v isa Integer && e.v > 0) || return nothing, nothing
        d = _derivative(b, k.name)
        (d === nothing || k.name in _symbol_names(d)) && return nothing, nothing
        a = _div(_sub(b, _mul(d, k)), d)          # b = d (k + a)
        append!(params, fill(a, Int(e.v)))
        push!(coef, _pow(d, e))
    end
    return params, _mul(coef...)
end
