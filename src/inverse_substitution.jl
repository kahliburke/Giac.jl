# ∫ f(x) dx where f holds F(u) for an inverse trigonometric or hyperbolic F of a linear u = c x + d:
# with u = G(t), G the inverse of F, F(u) is t, dx = G′(t)/c dt, and the radicals in f become
# trigonometric or hyperbolic functions of t, since 1 − sin²t = cos²t and so on. The integral in t
# is often one GIAC finds (a polynomial in t times sines and cosines) where the one in x is not.
#
# GIAC leaves √(k(1 − sin²t)) as it is, and then integrates by cases on the sign of cos t, so the
# radicals are rewritten here: on the range of F the root is a function S of t of known sign,
# √(1 − sin²t) = cos t for t in [−π/2, π/2]. An answer is kept only if its derivative is f.

# For each F: its inverse G, the expression Q in G that is a square on F's range, and the root S of
# Q that is not negative there
const _INVERSE_SUBSTITUTIONS = Dict(
    "asin"  => ("sin(t)",  "1-sin(t)^2",  "cos(t)"),
    "acos"  => ("cos(t)",  "1-cos(t)^2",  "sin(t)"),
    "atan"  => ("tan(t)",  "1+tan(t)^2",  "1/cos(t)"),
    "asinh" => ("sinh(t)", "1+sinh(t)^2", "cosh(t)"),
    "acosh" => ("cosh(t)", "cosh(t)^2-1", "sinh(t)"),
    "atanh" => ("tanh(t)", "1-tanh(t)^2", "1/cosh(t)"))

# exp(F(u)) for the inverse hyperbolic functions, without them
const _EXP_OF_INVERSE = Dict(
    "asinh" => u -> u + sqrt(u^2 + 1),
    "acosh" => u -> u + sqrt(u^2 - 1),
    "atanh" => u -> sqrt((1 + u) / (1 - u)))

# acosh(u) and asinh(u) as GIAC writes them, ln(u + √(u²−1)) and ln(u + √(u²+1)): the function's
# name and u, or nothing
function _log_form(m::Node)
    m isa CallNode && m.f in ("ln", "log") && length(m.args) == 1 && m.args[1] isa AddNode || return nothing
    terms = m.args[1].terms
    length(terms) == 2 || return nothing
    for (i, (r, cr)) in enumerate(terms)
        q = _sqrt_arg(r)
        (q === nothing || !isone(cr)) && continue
        um, cu = terms[3 - i]
        u = GiacExpr(_scale(um, cu))
        d = string(invoke_cmd(:simplify, GiacExpr(q) - u^2))
        d == "1" && return ("asinh", _node(u))
        d == "-1" && return ("acosh", _node(u))
    end
    return nothing
end
_sqrt_arg(n::CallNode) = n.f == "sqrt" && length(n.args) == 1 ? n.args[1] : nothing
_sqrt_arg(n::MulNode) = isone(n.coef) && length(n.factors) == 1 && isequal(n.factors[1][2], _HALF) ? n.factors[1][1] : nothing
_sqrt_arg(n::Node) = nothing

_linear_in(u::Node, x::SymNode) = !_free_of(u, x) && _free_of(_node(invoke_cmd(:diff, GiacExpr(u), GiacExpr(x))), x)

# the inverse function F(u) in `f` to substitute for, with u linear in x: the subtree that is
# F(u) (or GIAC's logarithm for it), F's name and u; or nothing
function _inverse_call(n::Node, x::SymNode)
    found = nothing
    _any_node(n) do m
        if m isa CallNode && haskey(_INVERSE_SUBSTITUTIONS, m.f) && length(m.args) == 1 && _linear_in(m.args[1], x)
            found = (m, m.f, m.args[1])
            return true
        end
        lf = _log_form(m)
        if lf !== nothing && _linear_in(lf[2], x)
            found = (m, lf[1], lf[2])
            return true
        end
        return false
    end
    return found
end

"""
    _inverse_substitution(f, x) -> antiderivative or nothing

`∫ f dx` by the substitution that makes an inverse trigonometric or hyperbolic function of a
linear argument in `f` the variable of integration.
"""
function _inverse_substitution(f::GiacExpr, x::GiacExpr)
    xn = _node(x)
    xn isa SymNode || return nothing
    found = _inverse_call(_node(f), xn)
    found === nothing && return nothing
    node, F, un = found
    G, Q, S = _INVERSE_SUBSTITUTIONS[F]
    t = _fresh_symbol(_node(f), "t")
    tsub(s) = substitute(giac_eval(s), giac_eval("t") => t)
    u = GiacExpr(un)
    Fu = GiacExpr(_call(F, Node[un]))
    c = invoke_cmd(:diff, u, x)
    d = invoke_cmd(:simplify, u - c * x)
    xt = (tsub(G) - d) / c
    g = GiacExpr(_replace_node(_node(f), node, _node(t)))
    # every other occurrence of x, and of F(u) written another way, in terms of t
    g = substitute(g, x => xt) * invoke_cmd(:diff, xt, t)
    # not simplified: GIAC's simplify writes sines and cosines as tangents, which `tlin` can't take apart
    g = _square_roots_in_t(g, tsub(Q), tsub(S), t)
    _any_node(m -> m isa SymNode && m.name == xn.name, _node(g)) && return nothing
    # Integrated as it is, and with its powers of sines and cosines made sums of multiple angles
    # first: GIAC integrates the first by tan(t/2), and the second term by term, which back in x is
    # usually shorter once the multiple angles are expanded. The shortest correct answer is kept.
    best = nothing
    deadline = get(task_local_storage(), :giac_integral_deadline, Inf)
    for h in (invoke_cmd(:tlin, g), g)
        time() > deadline && break
        # the integral in t is searched as any other
        r = _integrate_or_nothing(h, t)
        r === nothing && continue
        answer = try
            back = if haskey(_EXP_OF_INVERSE, F)
                # GIAC integrates sinh and cosh as exponentials exp(k t), and would write
                # exp(k F(u)) back. With a symbol E for exp(t) they are powers Eᵏ; then E is
                # exp(F(u)), u + √(u²+1) for asinh, and the t left is F(u)
                E = _fresh_symbol(_node(r), "E")
                rE = invoke_cmd(:exp2pow, substitute(r, t => log(E)))
                rE = GiacExpr(_replace_node(_node(rE), _node(log(E)), _node(t)))
                substitute(substitute(rE, E => _EXP_OF_INVERSE[F](u)), t => Fu)
            else
                substitute(r, t => Fu)
            end
            invoke_cmd(:simplify, invoke_cmd(:texpand, back))
        catch
            continue
        end
        best !== nothing && _size(_node(answer)) >= _size(_node(best)) && continue
        _is_antiderivative(answer, f, x) && (best = answer)
    end
    return best
end

# `g` with each power bᵉ of a fractional exponent whose base is k·Q, k free of t, written kᵉ·S²ᵉ
function _square_roots_in_t(g::GiacExpr, Q::GiacExpr, S::GiacExpr, t::GiacExpr)
    tn = _node(t)::SymNode
    root(b, e) = begin
        k = invoke_cmd(:simplify, GiacExpr(b) / Q)
        _free_of(_node(k), tn) ? _node(k^GiacExpr(e) * S^(2 * GiacExpr(e))) : nothing
    end
    rewrite(n::Node) = n
    function rewrite(n::CallNode)
        if n.f == "sqrt" && length(n.args) == 1 && !_free_of(n.args[1], tn)
            r = root(n.args[1], NumNode(1 // 2))
            r === nothing || return r
        end
        return _call(n.f, Node[rewrite(a) for a in n.args])
    end
    rewrite(n::AddNode) = _add((_scale(rewrite(m), c) for (m, c) in n.terms)...)
    function rewrite(n::MulNode)
        parts = Node[NumNode(n.coef)]
        for (b, e) in n.factors
            r = e isa NumNode && e.v isa Rational && !isinteger(e.v) && !_free_of(b, tn) ? root(b, e) : nothing
            push!(parts, r === nothing ? _pow(rewrite(b), rewrite(e)) : r)
        end
        return _mul(parts...)
    end
    return GiacExpr(rewrite(_node(g)))
end

# whether d/dx F is f: the difference simplifies to 0, or vanishes at two of a few exact points,
# the parameters given values of their own (a difference of branches of a root or a logarithm can
# leave it non-zero on part of the line)
function _is_antiderivative(F::GiacExpr, f::GiacExpr, x::GiacExpr)
    d = invoke_cmd(:diff, F, x) - f
    names = setdiff(string.(to_julia(invoke_cmd(:lname, f + F))), [string(x)])
    params = Dict(giac_identifier(n) => 1 // 2 + k // 7 for (k, n) in enumerate(names))
    matched = 0
    for p in (37 // 100, 83 // 100, 161 // 100)
        v = merge(params, Dict(x => p))
        fv, dv = try
            ComplexF64(to_julia(invoke_cmd(:evalf, substitute(f, v), 30))), ComplexF64(to_julia(invoke_cmd(:evalf, substitute(d, v), 30)))
        catch
            continue
        end
        isfinite(fv) && abs(fv) > 1e-12 && abs(dv) <= 1e-8 * max(abs(fv), 1.0) && (matched += 1)
    end
    matched >= 2 && return true
    return try string(invoke_cmd(:simplify, d)) == "0" catch; false end
end

# ∫ f dx with acosh and asinh as they are: GIAC rewrites them as logarithms when it evaluates an
# integrand, and integrates the logarithms less often than the functions, though its integrator keeps
# them while it works. The logarithms are made acosh and asinh again, and GIAC integrates with them
# kept.
function _with_inverse_hyperbolics(f::GiacExpr, x::GiacExpr)
    n = _node(f)
    kept = _restore_inverse_hyperbolics(n)
    isequal(kept, n) && return nothing
    r = with_settings(:keep_acosh_asinh => true) do
        invoke_cmd(:integrate, GiacExpr(kept), x)
    end
    _integral_left(r) && return nothing
    return _is_antiderivative(r, f, x) ? r : nothing
end

# `n` with each logarithm that is acosh or asinh written as that function
function _restore_inverse_hyperbolics(n::Node)
    lf = _log_form(n)
    lf === nothing || return _call(lf[1], Node[lf[2]])
    n isa CallNode && return _call(n.f, Node[_restore_inverse_hyperbolics(a) for a in n.args])
    n isa AddNode && return _add((_scale(_restore_inverse_hyperbolics(m), c) for (m, c) in n.terms)...)
    n isa MulNode && return _mul(NumNode(n.coef), (_pow(_restore_inverse_hyperbolics(b), _restore_inverse_hyperbolics(e)) for (b, e) in n.factors)...)
    return n
end

# the methods of the integration search, in the order they are tried
const _INDEFINITE_METHODS = Function[_with_inverse_hyperbolics, _inverse_substitution]
