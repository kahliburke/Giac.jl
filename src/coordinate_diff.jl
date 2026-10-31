# Differentiating in a function of time or one of its derivatives, as mechanics does: in ∂L/∂q and
# ∂L/∂q̇ the coordinate q(t) and the velocity q̇ are independent variables. GIAC's own `diff` in
# q(t) differentiates q̇ as well, and has no answer for an unknown function of q(t).

"""
    diff(f, q)
    diff(f, q̇)

The partial derivative of `f` in `q`, a function of a variable declared as `@giac_var t q(t) q̇`,
or in one of its derivatives. Every function of one variable and each of its derivatives counts as
a variable of its own, as in a Lagrangian `L(q, q̇, t)`:

```julia
@giac_var t m g ℓ θ(t) θ̇ L(x, v, s)
diff(m * ℓ^2 * θ̇^2 / 2 + m * g * ℓ * cos(θ), θ)   # -m*g*ℓ*sin(θ(t))
diff(m * ℓ^2 * θ̇^2 / 2, θ̇)                        # m*ℓ^2*diff(θ(t),t)
diff(L(θ, θ̇, t), θ̇)                                # ∂L/∂θ̇, at (θ, θ̇, t)
```

In any other variable it is GIAC's `diff`, with the chain rule through functions of it:
`diff(cos(θ), t)` is `-θ̇ sin θ`. For the derivative with the coordinates held fixed, see
[`partial`](@ref).
"""
function Base.diff(f::GiacExpr, x::GiacExpr)
    xn = _node(x)
    _is_coordinate(xn) || return invoke_cmd(:diff, f, x)
    return _diff_holding(f, xn)
end

"""
    partial(f, x)

The partial derivative of `f` in `x` with every function of one variable and each of its
derivatives held fixed: in `t`, the explicit dependence on time, `∂L/∂t`. In a coordinate `q` or
velocity `q̇` it is [`diff`](@ref).

```julia
@giac_var t m k F Ω q(t) q̇
L = m * q̇^2 / 2 - k * q^2 / 2 + F * q * cos(Ω * t)
partial(L, t)        # -F*Ω*q(t)*sin(Ω*t)
diff(L, t)           # the total derivative, through q and q̇ as well
```
"""
partial(f::GiacExpr, x::GiacExpr) = _diff_holding(f, _node(x))

"""
    differential(f, x, y, …)
    differential(x)

The differential of a scalar `f` in the variables `x, y, …`: `∂f/∂x dx + ∂f/∂y dy + …`, each `dx`
a symbol named after its variable and written as a differential. A variable may be a function of
time `q(t)` or one of its derivatives; each derivative is a [`partial`](@ref), so the coefficient
of `dt` is the explicit dependence on time. `differential(x)` of a variable alone is its `dx`, so
a coefficient reads off as `diff(df, differential(x))`.

```julia
@giac_var q q̇ p t L(x, v, s)
dH = differential(p * q̇ - L(q, q̇, t), q, q̇, p, t)
# -∂L/∂q dq + (p - ∂L/∂q̇) dq̇ + q̇ dp - ∂L/∂t dt
diff(dH, differential(p))     # q̇
```
"""
function differential(f::GiacExpr, x::GiacExpr, xs::GiacExpr...)
    dxs = [differential(v) for v in (x, xs...)]
    # a differential form is written in the order its variables were given
    lock(_DIFFERENTIAL_LOCK) do
        foreach(((k, dx),) -> _DIFFERENTIAL_ORDER[string(dx)] = k, enumerate(dxs))
    end
    return sum(partial(f, v) * dx for (v, dx) in zip((x, xs...), dxs))
end

const _DIFFERENTIAL_ORDER = Dict{String, Int}()
const _DIFFERENTIAL_LOCK = ReentrantLock()
_differential_order(name::AbstractString) = lock(() -> get(_DIFFERENTIAL_ORDER, name, typemax(Int)), _DIFFERENTIAL_LOCK)

function differential(x::GiacExpr)
    name = _wrt_name(x)
    # a trace or determinant, or an expression, is differentiated through its matrices
    (name === nothing || is_identifier(x) && lock(() -> haskey(_ATOMS, string(x)), _ATOMS_LOCK)) &&
        return _differential(x, nothing)
    return giac_identifier("d" * name)
end

# an undefined function of variables, `q(t)` or `f(x, y)`
_is_function_of_variables(n::Node) =
    n isa CallNode && n.f == "of" && length(n.args) >= 2 && all(a -> a isa SymNode, n.args)

# a coordinate: an undefined function of one variable, `q(t)`, or a derivative of one, `diff(q(t), t, 2)`.
# A function of several, such as `L(q, q̇, t)`, is differentiated through.
_is_trajectory(n::Node) = _is_function_of_variables(n) && length(n.args) == 2
_is_coordinate(n::Node) = _is_trajectory(n) ||
    (n isa CallNode && n.f == "diff" && length(n.args) >= 2 && _is_trajectory(n.args[1]) &&
     all(a -> a isa SymNode || a isa NumNode, n.args[2:end]))

# The tree with `f(n)` in place of each subtree for which it is not `nothing`, outermost first
function _replace_nodes(f, n::Node)
    r = f(n)
    r === nothing || return r
    if n isa CallNode
        args = Node[_replace_nodes(f, a) for a in n.args]
        return all(i -> args[i] === n.args[i], eachindex(args)) ? n : CallNode(n.f, args)
    end
    n isa AddNode && return _add((_replace_nodes(f, _scale(m, c)) for (m, c) in n.terms)...)
    n isa MulNode && return _mul(NumNode(n.coef), (_pow(_replace_nodes(f, b), _replace_nodes(f, e)) for (b, e) in n.factors)...)
    return n
end

# every coordinate in a tree, outermost first: q̇ is one, not q inside it
function _coordinates!(found, n::Node)
    if _is_coordinate(n)
        n in found || push!(found, n)
    elseif n isa CallNode
        foreach(a -> _coordinates!(found, a), n.args)
    elseif n isa AddNode
        foreach(t -> _coordinates!(found, first(t)), n.terms)
    elseif n isa MulNode
        foreach(((b, e),) -> (_coordinates!(found, b); _coordinates!(found, e)), n.factors)
    end
    return found
end

# Each coordinate becomes a symbol of its own, GIAC differentiates in `x` (in its symbol, for a
# coordinate), and the coordinates go back in. GIAC writes the derivative of an unknown function at
# a symbol as `diff(M(s), s)`, which it cannot evaluate once `s` is `q(t)` again: that is written
# `M'(s)`, `of(diff(M), s)`, first.
function _diff_holding(f::GiacExpr, x::Node)
    coordinates = _coordinates!(_is_coordinate(x) ? Node[x] : Node[], _node(f))
    isempty(coordinates) && return invoke_cmd(:diff, f, GiacExpr(x))
    # (a name starting with `_` is a unit to GIAC)
    frozen = Dict{Node, Node}(c => SymNode("coordinate__$k") for (k, c) in enumerate(coordinates))
    back = Dict{Node, Node}(s => c for (c, s) in frozen)
    g = _replace_nodes(n -> get(frozen, n, nothing), _node(f))
    d = invoke_cmd(:diff, GiacExpr(g), GiacExpr(get(frozen, x, x)))
    thawed = _replace_nodes(_node(d)) do n
        if n isa CallNode && n.f == "diff" && length(n.args) in (2, 3) && haskey(back, n.args[2])
            inner = n.args[1]
            if inner isa CallNode && inner.f == "of" && length(inner.args) == 2 && isequal(inner.args[2], n.args[2])
                k = length(n.args) == 3 && n.args[3] isa NumNode ? n.args[3].v : 1
                F = inner.args[1]
                for _ in 1:k
                    F = CallNode("diff", Node[F])
                end
                return CallNode("of", Node[F, back[n.args[2]]])
            end
        end
        return get(back, n, nothing)
    end
    return GiacExpr(thawed)
end
