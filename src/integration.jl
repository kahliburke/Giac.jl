# GIAC's indefinite integral returns infinity for some exact derivatives of expressions in unknown
# functions, ∫(u′v + uv′) dt among them. Those are found here, a derivative term at a time: a term
# V·u′·uᵖ comes from V·uᵖ⁺¹/(p+1), whose derivative is taken off what is left.

# Indefinite integration is a search. GIAC integrates first; where it leaves the integral or fails,
# the methods of `_INDEFINITE_METHODS` are tried in turn, each giving an antiderivative it has
# checked by differentiating, or nothing. A method that makes a new integral (a substitution's, in
# t) integrates it by `_integrate` again, so the new one is searched too, to a depth of
# `_SEARCH_DEPTH`.
#
# Time is shared out so that no one path takes it all. The methods are in order of promise, each
# giving up at once where it does not apply, and each that does has half of the time left (the last
# all of it), what it leaves over going to the next; a method's own search has only its share, and
# a method still running at the end of its share is stopped and the next one tried. Under a time
# limit (`with_timeout`) GIAC's first attempt has half of it; without one, the methods after it
# have `_SEARCH_SECONDS`. The search runs only with GIAC on a thread of its own (Julia started with
# more than one thread, or DedicatedThreads loaded) and a wrapper library that interrupts it, as
# only then can GIAC be stopped at any moment.

const _SEARCH_DEPTH = 3
const _SEARCH_SECONDS = 20.0

# a computation stopped from outside (with_timeout, interrupt), which ends the search
_stopped(e) = e isa InterruptException || e isa GiacTimeout ||
              (e isa Union{GiacError, ErrorException} && occursin("user interruption", sprint(showerror, e)))

# a stop of one share of the search, which is not a stop from outside: the time limit around it
# has not been reached
_share_spent(e) = e isa GiacTimeout && time() < get(task_local_storage(), :giac_cancel, Inf)

# `Commands.integrate(f, x)`
function _integrate(f::GiacExpr, x::GiacExpr)
    g = _special_terms_integral(f, x)                  # Bessel and Airy functions (special_functions.jl)
    g === nothing || return g
    limit = get(task_local_storage(), :giac_cancel, nothing)
    r = try
        limit === nothing ? invoke_cmd(:integrate, f, x) :
                            with_timeout(() -> invoke_cmd(:integrate, f, x), (limit - time()) / 2)
    catch e
        # GIAC's half of the time limit spent, or GIAC failing: the search, or GIAC's outcome
        _share_spent(e) || !_stopped(e) || rethrow()
        g = _search_integral(f, x)
        g === nothing ? rethrow() : return g
    end
    if _is_infinite(r) && occursin("diff(", string(f))
        g = _antiderivative(f, x)
        return g === nothing ? r : g
    end
    _integral_left(r) || return _without_compositions(r)
    g = _search_integral(f, x)
    return g === nothing ? r : g
end

# `r` without cosh(acosh(u)) and the like, which GIAC's integrator leaves (it keeps acosh and asinh
# while it works), if that is shorter, as it is, or simplified, with acosh and asinh kept
function _without_compositions(r::GiacExpr)
    n = _node(r)
    _any_node(m -> m isa CallNode && m.f in ("cosh", "sinh", "exp") &&
                   _any_node(k -> k isa CallNode && k.f in ("acosh", "asinh"), m), n) || return r
    return with_settings(:keep_acosh_asinh => true) do
        w = GiacExpr(_inverse_compositions(n))
        s = try invoke_cmd(:simplify, w) catch e; _stopped(e) && rethrow(); w end
        argmin(c -> _size(_node(c)), (r, w, s))
    end
end

# ∫ f dx by the methods after GIAC's, within the depth and time left to the search
function _search_integral(f::GiacExpr, x::GiacExpr)
    # the shares of time are kept by stopping GIAC, which needs GIAC on a thread of its own and a
    # wrapper library that stops it whenever asked: without them a search could run on without end,
    # and there is none
    _giac_threaded() && _wrapper_interrupts() || return nothing
    tls = task_local_storage()
    depth = get(tls, :giac_integral_depth, 0)
    depth >= _SEARCH_DEPTH && return nothing
    deadline = get(tls, :giac_integral_deadline, nothing)
    deadline === nothing && (deadline = min(time() + _SEARCH_SECONDS, get(tls, :giac_cancel, Inf)))
    k = length(_INDEFINITE_METHODS)
    task_local_storage(:giac_integral_depth, depth + 1) do
        for (i, method) in enumerate(_INDEFINITE_METHODS)
            share = (deadline - time()) / (i == k ? 1 : 2)
            share <= 0 && return nothing
            g = try
                with_timeout(share) do
                    task_local_storage(() -> method(f, x), :giac_integral_deadline, time() + share)
                end
            catch e
                _share_spent(e) && continue
                _stopped(e) && rethrow()
                nothing
            end
            g === nothing || return g
        end
        return nothing
    end
end

# ∫ f dx, the whole search, for a method's new integral: the antiderivative, or nothing
function _integrate_or_nothing(f::GiacExpr, x::GiacExpr)
    r = try
        _integrate(f, x)
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    return _integral_left(r) ? nothing : r
end

# `Commands.integrate(f, x, a, b)`: GIAC's, and where it leaves the integral unevaluated, the
# integral as derivatives of a known one in its parameters (parametric_integrals.jl), term by term
# with sines and cosines written as exponentials, or as the mean of f and its image under an
# involution of the interval
function _integrate(f::GiacExpr, x::GiacExpr, a, b)
    v = _special_definite_integral(f, x, a, b)       # Bessel and Airy functions (special_functions.jl)
    v === nothing || return v
    r = invoke_cmd(:integrate, f, x, a, b)
    _integral_left(r) || return r
    for method in (_parametric_integral, _exponential_integral, _reflected_integral, _parts_integral,
                   _antiderivative_integral)
        v = method(f, x, a, b)
        v === nothing || return v
    end
    return r
end

_integral_left(r::GiacExpr) = _is_infinite(r) || _any_node(n -> n isa CallNode && n.f == "integrate", _node(r))

# ∫ₐᵇ f with each sin u and cos u written as exponentials of ±iu, term by term: a Gaussian with
# a cosine is a sum of Gaussians with complex centres
function _exponential_integral(f::GiacExpr, x::GiacExpr, a, b)
    n, xn = _node(f), _node(x)
    xn isa SymNode || return nothing
    _any_node(m -> m isa CallNode && m.f in ("sin", "cos") && !_free_of(m, xn), n) || return nothing
    terms = _sum_terms(GiacExpr(_expand(_trig_as_exponentials(n))))
    total = zero(GiacExpr)
    for t in terms
        v = invoke_cmd(:integrate, t, x, a, b)
        _integral_left(v) && return nothing
        total += v
    end
    return invoke_cmd(:simplify, total)
end

function _trig_as_exponentials(n::Node)
    i = NumNode(im)
    if n isa CallNode
        args = Node[_trig_as_exponentials(a) for a in n.args]
        if n.f in ("sin", "cos") && length(args) == 1
            u = _mul(i, args[1])
            p, m = _exp(u), _exp(_neg(u))
            return n.f == "cos" ? _mul(_HALF, _add(p, m)) : _mul(NumNode(-im // 2), _sub(p, m))
        end
        return _call(n.f, args)
    elseif n isa AddNode
        return _add((_scale(_trig_as_exponentials(m), c) for (m, c) in n.terms)...)
    elseif n isa MulNode
        return _mul(NumNode(n.coef), (_pow(_trig_as_exponentials(b), _trig_as_exponentials(e)) for (b, e) in n.factors)...)
    end
    return n
end

# ∫ₐᵇ f = ½ ∫ₐᵇ (f + (f∘φ)·|φ′|) for a decreasing involution φ of [a, b]: a+b−x on a finite
# interval, (1−x)/(1+x) on [0, 1] and 1/x on [0, ∞). Taken when the sum is no longer than f and
# GIAC integrates it: x g(x) with g symmetric about the midpoint gives (a+b) g(x), and
# ln(1+x)/(1+x²) on [0, 1] gives ln 2/(1+x²).
function _reflected_integral(f::GiacExpr, x::GiacExpr, a, b)
    for φ in _involutions(x, convert(GiacExpr, a), convert(GiacExpr, b))
        g = invoke_cmd(:simplify, f - substitute(f, x => φ) * invoke_cmd(:diff, φ, x))
        _size(_node(g)) <= _size(_node(f)) || continue
        v = invoke_cmd(:integrate, g, x, a, b)
        _integral_left(v) || return invoke_cmd(:simplify, v / 2)
    end
    return nothing
end

function _involutions(x, a, b)
    infinite(e) = _any_node(n -> n isa SymNode && n.name in ("inf", "infinity"), _node(e))
    φs = GiacExpr[]
    infinite(a) || infinite(b) || push!(φs, a + b - x)
    if _is_zero_expr(a)
        _is_zero_expr(b - 1) && push!(φs, (1 - x) / (1 + x))
        _is_plus_infinity(_node(b)) && push!(φs, 1 / x)
    end
    return φs
end

_is_infinite(r::GiacExpr) = string(r) in ("infinity", "+infinity", "-infinity", "inf", "-inf", "undef")
_is_zero_expr(e::GiacExpr) = (n = _node(e); n isa NumNode && iszero(n.v))

function _antiderivative(f::GiacExpr, x::GiacExpr)
    G = zero(GiacExpr)
    R = GiacExpr(_expand(_node(f)))
    for _ in 1:12
        _is_zero_expr(R) && return G
        g = nothing
        for t in _sum_terms(R)
            g = _term_antiderivative(t, x)
            g === nothing || break
        end
        if g === nothing
            # no derivative term left to take off: GIAC integrates the rest
            rest = invoke_cmd(:integrate, R, x)
            return _is_infinite(rest) ? nothing : G + rest
        end
        G += g
        R = GiacExpr(_expand(_node(R - Commands.diff(g, x))))
    end
    return nothing
end

# For a term V·u⁽ᵏ⁾ (k = 1: V·u′·uᵖ), the expression whose derivative gives it with the rest of
# the product rule left over; nothing if it has no such factor
function _term_antiderivative(t::GiacExpr, x::GiacExpr)
    fs = _product_factors(t)
    i = findfirst(f -> _derivative_in(f, x) !== nothing, fs)
    i === nothing && return nothing
    u, k = _derivative_in(fs[i], x)
    others = fs[[j for j in eachindex(fs) if j != i]]
    k > 1 && return _product(others) * Commands.diff(u, x, k - 1)
    # the powers of u among the other factors join it: V u′ uᵖ is (V uᵖ⁺¹/(p+1))′ less V′ uᵖ⁺¹/(p+1)
    p = 0
    V = GiacExpr[]
    for f in others
        if isequal(f, u)
            p += 1
        elseif iscall(f) && symb_funcname(f) == "^" && isequal(arguments(f)[1], u) && is_integer(arguments(f)[2])
            p += Int(to_julia(arguments(f)[2]))
        else
            push!(V, f)
        end
    end
    return _product(V) * u^(p + 1) / (p + 1)
end

_product(fs) = isempty(fs) ? one(GiacExpr) : reduce(*, fs)

# (u, k) if `f` is the kth derivative in `x` of an unknown function u(x)
function _derivative_in(f::GiacExpr, x::GiacExpr)
    (iscall(f) && symb_funcname(f) == "diff") || return nothing
    a = arguments(f)
    (length(a) >= 2 && isequal(a[2], x) && iscall(a[1]) && symb_funcname(a[1]) == "of") || return nothing
    k = 1
    if length(a) >= 3
        is_integer(a[3]) || return nothing
        k = Int(to_julia(a[3]))
    end
    return (a[1], k)
end

# ---------------------------------------------------------------------------------------------
# Integration by parts
# ---------------------------------------------------------------------------------------------

# functions whose derivatives are elementary, taken as u in ∫ u w = uV - ∫ u'V
const _PARTS_HEADS = ("erf", "erfc", "ln", "atan", "asin", "acos", "asinh", "acosh", "atanh", "Ei", "Si", "Ci")
const _PARTS_DEPTH = 3

# f as u w: u a whole power of one of the functions in x, w the rest; or nothing
function _parts_split(f::GiacExpr, x::GiacExpr)
    n, name = _node(f), string(x)
    factors = n isa MulNode ? n.factors : Pair{Node, Node}[n => _ONE]
    for (i, (b, k)) in enumerate(factors)
        (b isa CallNode && b.f in _PARTS_HEADS && name in _symbol_names(b) &&
         k isa NumNode && k.v isa Integer && k.v > 0) || continue
        rest = n isa MulNode ? _mul(NumNode(n.coef), (_pow(c, e) for (j, (c, e)) in enumerate(factors) if j != i)...) : _ONE
        return GiacExpr(_pow(b, k)), GiacExpr(rest)
    end
    return nothing
end

# u, V = ∫ w and u'V for f = u w, within the depth of nested integrations by parts; or nothing
function _parts(f::GiacExpr, x::GiacExpr)
    depth = get(task_local_storage(), :giac_parts_depth, 0)
    depth >= _PARTS_DEPTH && return nothing
    split = _parts_split(f, x)
    split === nothing && return nothing
    u, w = split
    V = _is_zero_node(_node(w - 1)) ? x : _integrate_or_nothing(w, x)
    V === nothing && return nothing
    return u, V, invoke_cmd(:simplify, invoke_cmd(:diff, u, x) * V), depth
end

# ∫ f dx = uV - ∫ u'V, the antiderivative checked by its derivative; a sum term by term, as a
# term such as erf(x)² may need the parts its neighbours do not
function _parts_antiderivative(f::GiacExpr, x::GiacExpr)
    n = _node(f)
    n isa AddNode || _parts_split(f, x) !== nothing || (n = _expand(n))       # (1 - erf(x))²
    if n isa AddNode
        parts = GiacExpr[]
        for m in _terms(n)
            F = _integral_or_parts(GiacExpr(m), x)
            F === nothing && return nothing
            push!(parts, F)
        end
        F = sum(parts)
        return _is_antiderivative(F, f, x) ? F : nothing
    end
    p = _parts(f, x)
    p === nothing && return nothing
    u, V, g, depth = p
    R = task_local_storage(() -> _integral_or_parts(g, x), :giac_parts_depth, depth + 1)
    R === nothing && return nothing
    F = _erfi_form(u * V - R)
    return _is_antiderivative(F, f, x) ? F : nothing
end

# ∫ f dx by the whole integration, else by parts directly, which needs no search (the search runs
# only with GIAC on a thread of its own)
function _integral_or_parts(f::GiacExpr, x::GiacExpr)
    F = _integrate_or_nothing(f, x)
    F === nothing || return F
    depth = get(task_local_storage(), :giac_parts_depth, 0)
    depth >= _PARTS_DEPTH && return nothing
    return task_local_storage(() -> _parts_antiderivative(f, x), :giac_parts_depth, depth + 1)
end

# ∫ₐᵇ f dx = [uV]ₐᵇ - ∫ₐᵇ u'V, the ends by limits
function _parts_integral(f::GiacExpr, x::GiacExpr, a, b)
    p = _parts(f, x)
    p === nothing && return nothing
    u, V, g, depth = p
    UV = u * V
    ends = GiacExpr[]
    for (point, side) in ((b, -1), (a, 1))
        v = try
            _is_infinity(point) || _is_infinity(-convert(GiacExpr, point)) ? invoke_cmd(:limit, UV, x, point) :
                                                                              invoke_cmd(:limit, UV, x, point, side)
        catch e
            _stopped(e) && rethrow()
            return nothing
        end
        (_is_finite_value(v) && !_unevaluated(v, "limit")) || return nothing
        push!(ends, v)
    end
    R = task_local_storage(() -> Commands.integrate(g, x, a, b), :giac_parts_depth, depth + 1)
    _integral_left(R) && return nothing
    return invoke_cmd(:simplify, ends[1] - ends[2] - R)
end

# ∫ₐᵇ f dx as F(b⁻) - F(a⁺) for the antiderivative F the search finds, kept only where it agrees
# with the integral computed numerically: F may jump inside the interval, and an integrand such as
# erfc(x)², which GIAC writes erf(x)² - 2erf(x) + 1, has terms that diverge apart
function _antiderivative_integral(f::GiacExpr, x::GiacExpr, a, b)
    F = _integral_or_parts(f, x)
    F === nothing && return nothing
    ends = GiacExpr[]
    for (point, side) in ((b, -1), (a, 1))
        v = try
            _is_infinity(point) || _is_infinity(-convert(GiacExpr, point)) ? invoke_cmd(:limit, F, x, point) :
                                                                              invoke_cmd(:limit, F, x, point, side)
        catch e
            _stopped(e) && rethrow()
            return nothing
        end
        (_is_finite_value(v) && !_unevaluated(v, "limit")) || return nothing
        push!(ends, v)
    end
    value = invoke_cmd(:simplify, ends[1] - ends[2])
    numeric = try
        _approx_value(invoke_cmd(:evalf, _hold(:integrate, (f, x, convert(GiacExpr, a), convert(GiacExpr, b)))))
    catch e
        _stopped(e) && rethrow()
        nothing
    end
    exact = _approx_value(value)
    (numeric isa Number && exact isa Number && abs(numeric - exact) <= 1e-8 * max(1, abs(exact))) || return nothing
    return value
end
