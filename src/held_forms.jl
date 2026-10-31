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
_is_held(n::Node) = _is_integral(n) || _is_evaluation(n)

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

Integrate the held integral `I` (from [`integral`](@ref)) by parts, differentiating `u`: each term
`u v′` of the integrand becomes `u v` evaluated between the limits, less the integral of `u′ v`.
Terms without the factor `u` stay under the integral.

```julia
@giac_var t a b f(t) g(t)
by_parts(integral(f * diff(g, t), t, a, b), f)
# f g |ₐᵇ - ∫ f′ g dt
```
"""
function by_parts(I::GiacExpr, u::GiacExpr)
    n = _node(I)
    _is_integral(n) || throw(ArgumentError("by_parts works on a held integral; make one with integral(f, t, a, b)"))
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

_has_factor(term::Node, u::Node) = isequal(term, u) ||
    (term isa MulNode && any(((b, e),) -> isequal(b, u) && isequal(e, _ONE), term.factors))

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
    end
    return CallNode("preval", Node[g(m.args[1]), g(m.args[2]), g(m.args[3]), m.args[4]])
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
            v = invoke_cmd(:eval, GiacExpr(h))
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
