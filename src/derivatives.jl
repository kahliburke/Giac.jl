# Derivatives taken on the tree (core.jl). `diff` in a variable is answered here when every part of
# the expression that depends on the variable is a sum, product, power or elementary function;
# anything else (an unknown function of it, a held command, a GIAC object the tree does not model)
# leaves the whole derivative to GIAC.

_is_zero_node(n::Node) = n isa NumNode && iszero(n.v)

_ln(b::Node) = b isa SymNode && b.name == "e" ? _ONE : _call("ln", Node[b])

# f′(u) for the function f applied as `n = f(u)`
const _TREE_DERIVATIVES = Dict{String, Function}(
    "exp" => (n, u) -> n,
    "ln" => (n, u) -> _pow(u, _MINUS_ONE),
    "sin" => (n, u) -> _call("cos", Node[u]),
    "cos" => (n, u) -> _neg(_call("sin", Node[u])),
    "tan" => (n, u) -> _add(_ONE, _pow(n, NumNode(2))),
    "asin" => (n, u) -> _pow(_sub(_ONE, _pow(u, NumNode(2))), NumNode(-1 // 2)),
    "acos" => (n, u) -> _neg(_pow(_sub(_ONE, _pow(u, NumNode(2))), NumNode(-1 // 2))),
    "atan" => (n, u) -> _pow(_add(_ONE, _pow(u, NumNode(2))), _MINUS_ONE),
    "sinh" => (n, u) -> _call("cosh", Node[u]),
    "cosh" => (n, u) -> _call("sinh", Node[u]),
    "tanh" => (n, u) -> _sub(_ONE, _pow(n, NumNode(2))),
    "asinh" => (n, u) -> _pow(_add(_pow(u, NumNode(2)), _ONE), NumNode(-1 // 2)),
    "acosh" => (n, u) -> _pow(_sub(_pow(u, NumNode(2)), _ONE), NumNode(-1 // 2)),
    "atanh" => (n, u) -> _pow(_sub(_ONE, _pow(u, NumNode(2))), _MINUS_ONE))

# ∂n/∂x for the variable named `x`, or `nothing` where the tree does not differentiate it
_derivative(n::NumNode, x::String) = _ZERO
_derivative(n::SymNode, x::String) = n.name == x ? _ONE : _ZERO
_derivative(n::GenNode, x::String) = nothing

function _derivative(n::AddNode, x::String)
    terms = Node[]
    for (m, c) in n.terms
        d = _derivative(m, x)
        d === nothing && return nothing
        _is_zero_node(d) || push!(terms, _mul(NumNode(c), d))
    end
    return isempty(terms) ? _ZERO : _add(terms...)
end

# the product times the logarithmic derivative of each factor b^e: e b′/b + e′ ln b
function _derivative(n::MulNode, x::String)
    terms = Node[]
    for (b, e) in n.factors
        db = _derivative(b, x)
        db === nothing && return nothing
        de = _derivative(e, x)
        de === nothing && return nothing
        _is_zero_node(db) || push!(terms, _mul(n, e, db, _pow(b, _MINUS_ONE)))
        _is_zero_node(de) || push!(terms, _mul(n, de, _ln(b)))
    end
    return isempty(terms) ? _ZERO : _add(terms...)
end

function _derivative(n::CallNode, x::String)
    n.f == "of" && return _special_derivative(n, x)
    rule = length(n.args) == 1 ? get(_TREE_DERIVATIVES, n.f, nothing) : nothing
    rule === nothing && return x in _symbol_names(n) ? _giac_derivative(n, x) : _ZERO
    u = only(n.args)
    du = _derivative(u, x)
    (du === nothing || _is_zero_node(du)) && return du
    return _mul(du, rule(n, u))          # u′ f′(u), in GIAC's order
end

# GIAC's derivative of a call the tree has no rule for, where the expression around it has calls
# GIAC cannot differentiate (special functions, root sums), so that it is differentiated here
# whole; nothing otherwise, for GIAC to differentiate all of it
function _giac_derivative(n::CallNode, x::String)
    get(task_local_storage(), :giac_diff_calls, false) || return nothing
    (_has_special(n) || _has_rootsum(n)) && return nothing
    return _node(_invoke_cmd_direct(:diff, (GiacExpr(n), GiacExpr(SymNode(x)))))
end

# `diff(f, x)` or `diff(f, x, k)` on the tree, for a variable `x`; `nothing` for GIAC to answer
function _tree_diff(f, x, k = 1)
    (f isa GiacExpr && x isa GiacExpr && k isa Integer && k >= 0) || return nothing
    xn = _node(x)
    (xn isa SymNode && !(xn.name in _GIAC_NAMED_VALUES)) || return nothing
    n = _node(f)
    n isa GenNode && return nothing
    assist = _has_special(n) || _has_rootsum(n)
    for _ in 1:k
        n = assist ? task_local_storage(() -> _derivative(n, xn.name), :giac_diff_calls, true) : _derivative(n, xn.name)
        n === nothing && return nothing
    end
    return GiacExpr(n)
end
_tree_diff(args...) = nothing
