# Held forms: integrals and evaluations written and worked on as expressions, not evaluated;
# integration by parts. Each is the GIAC command that would evaluate it (integrate, subst, preval),
# left unevaluated, so its value is GIAC's if it is evaluated after all.

"""
    integral(f, t)
    integral(f, t, a, b)

The integral of `f` in `t` (from `a` to `b`), held: written ∫ f dt and not evaluated. It is an
expression like any other, so sums and products can include it, [`substitute`](@ref) works inside
it, `factor`, `simplify`, `expand` and `normal` work on its integrand, and [`by_parts`](@ref)
rewrites it.

```julia
@giac_var t t₁ t₂ L(x, v, s) q(t) q̇
S = integral(L(q, q̇, t), t, t₁, t₂)      # ∫ L(q, q̇, t) dt from t₁ to t₂
```
"""
integral(f, t::GiacExpr, limits...) =
    GiacExpr(_held_integral(_node(convert(GiacExpr, f)), Node[_node(t), (_node(convert(GiacExpr, l)) for l in limits)...]))

"""
    evaluated(f, x => a)
    evaluated(f, x, a, b)

`f` at `x = a`, held: written f|ₓ₌ₐ and not evaluated. With two values, the difference
`f(b) - f(a)`, written f|ₐᵇ. [`substitute`](@ref) takes its value when a substitution changes
that value, as `η(a) => 0` does for `evaluated(p * η, t, a, b)`.

```julia
@giac_var t a b ε p(t) η(t)
evaluated(p * η, t, a, b)            # p η |ₐᵇ
evaluated(cos(ε * t), ε => 0)        # cos(ε t) |_{ε=0}
```
"""
evaluated(f, p::Pair) =
    GiacExpr(CallNode("subst", Node[_node(convert(GiacExpr, f)), CallNode("=", Node[_node(first(p)), _node(convert(GiacExpr, last(p)))])]))
evaluated(f, x::GiacExpr, a, b) =
    GiacExpr(CallNode("preval", Node[_node(convert(GiacExpr, f)), _node(convert(GiacExpr, a)), _node(convert(GiacExpr, b)), _node(x)]))

_is_integral(n::Node) = n isa CallNode && n.f == "integrate" && length(n.args) in (2, 4)
_is_evaluation(n::Node) = n isa CallNode && ((n.f == "subst" && length(n.args) == 2 && n.args[2] isa CallNode && n.args[2].f == "=") ||
                                             (n.f == "preval" && length(n.args) == 4))
_is_held(n::Node) = _is_integral(n) || _is_evaluation(n) || _is_held_call(n)

# Commands held as written: a node with one of these heads is kept from GIAC's evaluation and
# written in its notation (lim, Σ, ℒ, …). A command held with `hold_cmd` joins them.
const _HELD_HEADS = Set(["limit", "sum", "product", "laplace", "ilaplace", "invlaplace", "ztrans", "invztrans",
                         "ztransform", "invztransform", "sum_riemann"])
const _HELD_HEADS_LOCK = ReentrantLock()

# a held call other than an integral or an evaluation; `diff` of an expression is held, while the
# derivatives GIAC returns as values are not
function _is_held_call(n::Node)
    n isa CallNode || return false
    n.f == "diff" && return !_is_derivative_value(n)
    # an integral or an evaluation has its own written form; other shapes of these are held calls
    n.f in ("integrate", "subst", "preval") && return !(_is_integral(n) || _is_evaluation(n))
    return lock(() -> n.f in _HELD_HEADS, _HELD_HEADS_LOCK)
end

# a derivative GIAC leaves as it is: of an unknown function, `diff(q(t), t)`, of such a derivative,
# or of a function in one of its arguments, the `diff(F, 0)` of `∂F/∂x`
_is_derivative_value(n::CallNode) =
    length(n.args) == 2 && (n.args[2] isa NumNode ||
                            (n.args[1] isa CallNode && (n.args[1].f == "of" || n.args[1].f == "diff" && _is_derivative_value(n.args[1]))))

# `hold_cmd(cmd, args...)`: the command as a held node, its arguments as given
function _hold(cmd::Symbol, args::Tuple)
    nodes = Node[_held_arg(a) for a in args]
    cmd === :integrate && length(nodes) in (2, 4) && return GiacExpr(_held_integral(nodes[1], nodes[2:end]))
    cmd === :diff || lock(() -> push!(_HELD_HEADS, string(cmd)), _HELD_HEADS_LOCK)
    return GiacExpr(CallNode(string(cmd), nodes))
end

_held_arg(a::GiacExpr) = _node(a)
_held_arg(a::Symbol) = _node(giac_identifier(string(a)))
_held_arg(a::AbstractString) = _node(giac_eval(a))
_held_arg(a::AbstractVector) = _node(giac_eval(_arg_to_giac_string(a)))        # a list
_held_arg(a::GiacMatrix) = _node(GiacExpr(_make_gen_ptr(_get_gen(a.ptr))))     # a handle of its own
_held_arg(a) = _node(convert(GiacExpr, a))

# `release(e)` (Commands.jl): the held forms in `e` evaluated, from the inside out
# A held form alone gives GIAC's result as GIAC returns it (ifactor(120) stays 2^3*3*5); inside
# an expression each one's result joins the tree.
function _release_expr(e::GiacExpr)
    n = _node(e)
    return _is_held(n) ? _release_call(n) : GiacExpr(_release(n))
end

_release(n::Node) = _replace_nodes(m -> _is_held(m) ? _node(_release_call(m)) : nothing, n)

function _release_call(m::CallNode)
    args = GiacExpr[GiacExpr(_release(a)) for a in m.args]
    return m.f == "integrate" ? Commands.integrate(args...) : invoke_cmd(Symbol(m.f), args...)
end

"""
    integrand(I)

The integrand of the held integral `I`, with a number in front of the integral taken in.
"""
function integrand(I::GiacExpr)
    n = _node(I)
    _is_integral(n) && return GiacExpr(n.args[1])
    if n isa MulNode && length(n.factors) == 1 && _is_integral(first(n.factors[1])) && isequal(last(n.factors[1]), _ONE)
        return GiacExpr(_scale(first(n.factors[1]).args[1], n.coef))
    end
    throw(ArgumentError("$I is not a held integral; make one with integral(f, t, a, b)"))
end

# whether `pred` holds for a subtree of `n`
_any_node(pred, n::Node) = pred(n) ||
    (n isa CallNode && any(a -> _any_node(pred, a), n.args)) ||
    (n isa AddNode && any(t -> _any_node(pred, first(t)), n.terms)) ||
    (n isa MulNode && any(((b, e),) -> _any_node(pred, b) || _any_node(pred, e), n.factors))

"""
    by_parts(I, u)

Integrate the held integrals in `I` (from [`integral`](@ref)) by parts, differentiating `u`: each
term `u v′` of an integrand becomes `u v` evaluated between the limits, less the integral of `u′ v`.
Terms without the factor `u` stay under the integral.

```julia
@giac_var t a b f(t) g(t)
by_parts(integral(f * diff(g, t), t, a, b), f)
# f g |ₐᵇ - ∫ f′ g dt
```
"""
function by_parts(I::GiacExpr, u::GiacExpr)
    n = _node(I)
    _is_integral(n) && return _by_parts(n, u)
    _any_node(_is_integral, n) || throw(ArgumentError("by_parts works on held integrals; make one with integral(f, t, a, b)"))
    # (`_add` distributes a number over the sum that replaces an integral: -(f|ₐᵇ - ∫) is -f|ₐᵇ + ∫)
    return GiacExpr(_add(_replace_nodes(m -> _is_integral(m) ? _node(_by_parts(m, u)) : nothing, n)))
end

function _by_parts(n::CallNode, u::GiacExpr)
    t, limits = GiacExpr(n.args[2]), GiacExpr.(n.args[3:end])
    f, un = n.args[1], _node(u)
    uvs, inside = zero(GiacExpr), zero(GiacExpr)
    for term in (f isa AddNode ? Node[_scale(m, c) for (m, c) in f.terms] : Node[f])
        if _has_factor(term, un)
            uv, rest = to_julia(invoke_cmd(:ibpu, GiacExpr(term), u, t))
            uvs += uv
            inside += rest
        else
            inside += GiacExpr(term)
        end
    end
    boundary = isempty(limits) ? uvs : evaluated(uvs, t, limits...)
    return iszero(inside) ? boundary : boundary + integral(inside, t, limits...)
end

# whether `u` divides `term`: the quotient has no negative power of a base of `u` (x divides x² sin x)
function _has_factor(term::Node, u::Node)
    isequal(term, u) && return true
    bases = u isa MulNode ? Node[b for (b, _) in u.factors] : Node[u]
    q = _div(term, u)
    q isa MulNode || return true
    return !any(((b, e),) -> any(isequal(b), bases) && !(e isa NumNode && e.v isa Real && e.v >= 0), q.factors)
end

# The integral node, with a negative integrand's sign outside: -∫ f dt rather than ∫ -f dt
function _held_integral(f::Node, args::Vector{Node})
    neg = f isa MulNode && _negative(f.coef)
    I = CallNode("integrate", Node[neg ? _neg(f) : f; args])
    return neg ? _neg(I) : I
end

# A held form with `f` applied to what it holds; the variable it is in is left as it is
function _held_with(f, m::CallNode)
    g(a) = _node(f(GiacExpr(a)))
    if m.f == "integrate"
        return _held_integral(g(m.args[1]), Node[k == 1 ? a : g(a) for (k, a) in enumerate(m.args[2:end])])
    elseif m.f == "subst"
        x, a = m.args[2].args
        return CallNode("subst", Node[g(m.args[1]), CallNode("=", Node[x, g(a)])])
    elseif m.f == "preval"
        return CallNode("preval", Node[g(m.args[1]), g(m.args[2]), g(m.args[3]), m.args[4]])
    end
    # another held command: what it acts on, the first argument
    return isempty(m.args) ? m : CallNode(m.f, Node[g(m.args[1]); m.args[2:end]])
end

# The value of a held evaluation: f at x = a, or f(b) - f(a)
function _evaluation_value(h::CallNode)
    if h.f == "subst"
        x, a = h.args[2].args
        return _value_at(h.args[1], x, a)
    end
    f, a, b, x = h.args
    return _value_at(f, x, b) - _value_at(f, x, a)
end

# `f` at `x = a`. A derivative of a function of `x` is first written as the derivative function at
# `x`, φ′(x), which GIAC evaluates at a point; diff(φ(x), x) at x = 0 it cannot.
function _value_at(f::Node, x::Node, a::Node)
    g = _replace_nodes(f) do m
        (m isa CallNode && m.f == "diff" && length(m.args) in (2, 3) && isequal(m.args[2], x) &&
         m.args[1] isa CallNode && m.args[1].f == "of" && length(m.args[1].args) == 2 && isequal(m.args[1].args[2], x)) || return nothing
        k = length(m.args) == 3 && m.args[3] isa NumNode ? m.args[3].v : 1
        F = m.args[1].args[1]
        for _ in 1:k
            F = CallNode("diff", Node[F])
        end
        return CallNode("of", Node[F, x])
    end
    return substitute(GiacExpr(g), GiacExpr(x) => GiacExpr(a))
end

# `f` applied to `expr` with its held forms kept from GIAC, which would evaluate them: each is a
# symbol while `f` works on the rest, and goes back with `f` applied to what it holds. With
# `values`, an evaluation whose value `f` changes is replaced by that changed value.
function _around_held(f, expr::GiacExpr; values::Bool = false)
    held = Dict{Node, Node}()
    protected = _replace_nodes(_node(expr)) do m
        _is_held(m) || return nothing
        s = SymNode("held__$(length(held) + 1)")
        h = _held_with(f, m)
        if values && _is_evaluation(h)
            v = _evaluation_value(h)
            w = f(v)
            isequal(_node(w), _node(v)) || (h = _node(w))
        end
        held[s] = h
        return s
    end
    return GiacExpr(_replace_nodes(m -> get(held, m, nothing), _node(f(GiacExpr(protected)))))
end

# commands that work inside a held form
const _INSIDE_HELD = Set([:factor, :simplify, :expand, :normal, :ratnormal, :regroup, :collect, :partfrac, :trigsimp, :tsimplify])
