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
# name and u, or nothing. The first is acosh(u) where it is real, u ≥ 1; the second is asinh(u)
function _log_form(m::Node)
    m isa CallNode && m.f in ("ln", "log") && length(m.args) == 1 && m.args[1] isa AddNode || return nothing
    terms = m.args[1].terms
    length(terms) == 2 || return nothing
    for (i, (r, cr)) in enumerate(terms)
        q = _sqrt_arg(r)
        (q === nothing || !isone(cr)) && continue
        um, cu = terms[3 - i]
        u = GiacExpr(_scale(um, cu))
        d = GiacExpr(q) - u^2
        _is_zero_expr(invoke_cmd(:normal, d - 1)) && return ("asinh", _node(u))
        _is_zero_expr(invoke_cmd(:normal, d + 1)) && return ("acosh", _node(u))
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

const _CHECK_SECONDS = 2.0

# The check of an antiderivative evaluates at this many points, to this many digits, and takes a
# value as zero below this tolerance relative to the integrand's. GIAC evaluates some functions in
# doubles whatever the digits asked, and cancellation among the terms of F' can lose several more:
# the tolerance is far above that, and far below the difference a wrong branch or a wrong answer makes,
# which is of the integrand's size
const _CHECK_POINTS = 12
const _CHECK_DRAWS = 8 * _CHECK_POINTS          # points drawn at most, for a domain that is a small part of the space
const _CHECK_DIGITS = 30
const _CHECK_TOLERANCE = 1e-6

# the k-th point of Roberts' R_d sequence, a low-discrepancy sequence in [0, 1)ᵈ
function _sequence_point(k::Int, d::Int)
    g = 2.0                                     # the positive root of g^(d+1) = g + 1
    for _ in 1:100                              # by its contraction, to the last bit
        g = (1 + g)^(1 / (d + 1))
    end
    return [mod(0.5 + k / g^j, 1.0) for j in 1:d]
end

# u in [0, 1) as a point of the intervals: the one u falls in when there are several, and on it
# tan(π(u - 1/2)) for the whole line, tan(πu/2) from the finite end of a half-line, linear on a
# bounded one
function _on_intervals(u::Float64, intervals::Vector{Tuple{Float64, Float64}})
    n = length(intervals)
    i = min(n, floor(Int, u * n) + 1)
    u = u * n - (i - 1)
    lo, hi = intervals[i]
    isinf(lo) && isinf(hi) && return tan(π * (u - 0.5))
    isinf(hi) && return lo + tan(π * u / 2)
    isinf(lo) && return hi - tan(π * u / 2)
    return lo + (hi - lo) * u
end

# the open intervals v is assumed to lie in (by `assuming`, or a declaration), as GIAC's `about`
# gives them, their ends exact (±infinity for none); the whole line where nothing is assumed
function _assumed_bounds(v::GiacExpr)
    line = [(giac_eval("-infinity"), giac_eval("+infinity"))]
    a = try
        to_julia(invoke_cmd(:about, v))
    catch err
        _stopped(err) && rethrow()
        return line
    end
    (a isa AbstractVector && length(a) >= 2) || return line
    items(e) = e isa GiacExpr ? to_julia(e) : e         # numbers come back as Julia's
    out = Tuple{GiacExpr, GiacExpr}[]
    intervals = items(a[2])
    intervals isa AbstractVector || return line
    for iv in intervals
        ends = items(iv)
        (ends isa AbstractVector && length(ends) == 2) &&
            push!(out, (convert(GiacExpr, ends[1]), convert(GiacExpr, ends[2])))
    end
    return isempty(out) ? line : out
end

# the assumed intervals of v in doubles
function _assumed_intervals(v::GiacExpr)
    value(e) = (s = string(e); s in ("+infinity", "infinity") ? Inf : s == "-infinity" ? -Inf :
                               Float64(real(ComplexF64(to_julia(invoke_cmd(:evalf, e, 20))))))
    out = [(value(lo), value(hi)) for (lo, hi) in _assumed_bounds(v)]
    out = filter(((lo, hi),) -> lo < hi, out)
    return isempty(out) ? [(-Inf, Inf)] : out
end

# the k-th generic point for variables with the given intervals: points of both signs and all
# scales where nothing is assumed, exact numbers, the same on each run
_generic_point(k::Int, ranges::Vector{Vector{Tuple{Float64, Float64}}}) =
    [rationalize(_on_intervals(u, r); tol = 1e-6) for (u, r) in zip(_sequence_point(k, length(ranges)), ranges)]
_generic_point(k::Int, d::Int) = _generic_point(k, [[(-Inf, Inf)] for _ in 1:d])

# e's value at the point, or nothing where it has none there
function _value_at(e::GiacExpr, point::Dict)
    v = try
        ComplexF64(to_julia(invoke_cmd(:evalf, substitute(e, point), _CHECK_DIGITS)))
    catch err
        _stopped(err) && rethrow()
        return nothing
    end
    return isfinite(v) ? v : nothing
end

_is_real(v::Complex) = abs(imag(v)) <= _CHECK_TOLERANCE * max(abs(v), 1.0)

# the bases of the fractional powers in e
function _root_bases(e::GiacExpr)
    bases = GiacExpr[]
    _any_node(_node(e)) do m
        m isa MulNode && for (b, k) in m.factors
            k isa NumNode && k.v isa Rational && !isinteger(k.v) && (B = GiacExpr(b); any(isequal(B), bases) || push!(bases, B))
        end
        m isa CallNode && m.f == "sqrt" && length(m.args) == 1 && push!(bases, GiacExpr(m.args[1]))
        false
    end
    return bases
end

# whether d/dx F is f on f's real domain: F' - f, with F' by differentiation, vanishes at each of a
# set of generic points, in x and the parameters within what is assumed of them, where f is a real
# function (real, with the base of each fractional power in it non-negative), drawn until there are
# enough of them, and there are two such points at least. A nonzero F' - f vanishes at few generic
# points, and an F that holds on part of the domain only, as one with the wrong branch of a root,
# differs at points of the rest. Where f is real at fewer than two of them, every point where it has
# a value counts; where it has a value at fewer than two (an unknown function in it), F' - f is
# simplified
function _is_antiderivative(F::GiacExpr, f::GiacExpr, x::GiacExpr)
    d = invoke_cmd(:diff, F, x) - f
    names = [giac_identifier(n) for n in sort!(setdiff(string.(to_julia(invoke_cmd(:lname, f + F))), [string(x)]))]
    ranges = [_assumed_intervals(v) for v in [x; names]]
    bases = _root_bases(f)
    on_line, off_line = 0, 0
    differs = false
    for k in 1:_CHECK_DRAWS
        on_line >= _CHECK_POINTS && break
        values = _generic_point(k, ranges)
        point = Dict{GiacExpr, Rational{Int}}(x => values[1], (n => v for (n, v) in zip(names, values[2:end]))...)
        fv = _value_at(f, point)
        fv === nothing && continue
        on = _is_real(fv) && all(b -> (v = _value_at(b, point); v !== nothing && _is_real(v) && real(v) >= 0), bases)
        on || off_line < _CHECK_POINTS || continue
        dv = _value_at(d, point)
        dv === nothing && continue
        zero_ = abs(dv) <= _CHECK_TOLERANCE * max(abs(fv), 1.0)
        if on
            zero_ || return false
            on_line += 1
        else
            off_line += 1
            differs |= !zero_
        end
    end
    on_line >= 2 && return true
    on_line + off_line >= 2 && return !differs
    return try
        _giac_threaded() && _wrapper_interrupts() ? string(_simplified_within(d, _CHECK_SECONDS)) == "0" :
                                                    string(invoke_cmd(:simplify, d)) == "0"
    catch e
        _stopped(e) && rethrow()
        false
    end
end

# ∫ f dx with acosh and asinh as they are: GIAC rewrites them as logarithms when it evaluates an
# integrand, and integrates the logarithms less often than the functions, though its integrator keeps
# them while it works. The logarithms are made acosh and asinh again, and GIAC integrates with them
# kept.
function _with_inverse_hyperbolics(f::GiacExpr, x::GiacExpr)
    n = _node(f)
    kept = _restore_inverse_hyperbolics(n)
    isequal(kept, n) && return nothing
    r = with_settings(() -> invoke_cmd(:integrate, GiacExpr(kept), x), :keep_acosh_asinh => true)
    _integral_left(r) && return nothing
    r = _without_compositions(r)
    return _is_antiderivative(r, f, x) ? r : nothing
end

# GIAC's answers with acosh and asinh kept hold cosh(acosh(u)) and the like: written without them.
# The forms with √(u−1)√(u+1) hold for every u on the principal branches.
const _COMPOSITIONS = Dict(
    ("cosh", "acosh") => u -> u, ("sinh", "acosh") => u -> sqrt(u - 1) * sqrt(u + 1),
    ("exp", "acosh") => u -> u + sqrt(u - 1) * sqrt(u + 1),
    ("sinh", "asinh") => u -> u, ("cosh", "asinh") => u -> sqrt(1 + u^2), ("exp", "asinh") => u -> u + sqrt(1 + u^2))

function _inverse_compositions(n::Node)
    if n isa CallNode
        args = Node[_inverse_compositions(a) for a in n.args]
        if length(args) == 1 && args[1] isa CallNode && length(args[1].args) == 1
            rule = get(_COMPOSITIONS, (n.f, args[1].f), nothing)
            rule === nothing || return _node(rule(GiacExpr(args[1].args[1])))
        end
        # exp(k F(u)) for an integer k, as exp(F(u))ᵏ
        if n.f == "exp" && length(args) == 1 && args[1] isa MulNode && isinteger(args[1].coef) &&
           length(args[1].factors) == 1 && isequal(args[1].factors[1][2], _ONE)
            F = args[1].factors[1][1]
            if F isa CallNode && length(F.args) == 1 && haskey(_COMPOSITIONS, ("exp", F.f))
                return _node(_COMPOSITIONS[("exp", F.f)](GiacExpr(F.args[1]))^Int(args[1].coef))
            end
        end
        return _call(n.f, args)
    elseif n isa AddNode
        return _add((_scale(_inverse_compositions(m), c) for (m, c) in n.terms)...)
    elseif n isa MulNode
        return _mul(NumNode(n.coef), (_pow(_inverse_compositions(b), _inverse_compositions(e)) for (b, e) in n.factors)...)
    end
    return n
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
