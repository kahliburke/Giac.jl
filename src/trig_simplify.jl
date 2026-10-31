# GIAC's simplify can rewrite trigonometric functions in angles the expression did not have:
# tangents of half angles, powers of cos(θ/3) for cos θ. When it does and the result is not
# clearly shorter, the expression's normal form is returned instead.

const _TRIG = Set(["sin", "cos", "tan", "sinh", "cosh", "tanh"])

function _trig_safe_simplify(e::GiacExpr)
    r = _invoke_cmd_direct(:simplify, (e,))
    n = _node(e)
    _has_trig(n) || return r
    m = _node(r)
    issubset(_trig_args!(Set{Node}(), m), _trig_args!(Set{Node}(), n)) && return r
    alt = _invoke_cmd_direct(:normal, (e,))
    return 3 * _size(m) < 2 * _size(_node(alt)) ? r : alt
end

_has_trig(n::Node) = false
_has_trig(n::CallNode) = n.f in _TRIG || any(_has_trig, n.args)
_has_trig(n::AddNode) = any(((m, c),) -> _has_trig(m), n.terms)
_has_trig(n::MulNode) = any(((b, e),) -> _has_trig(b) || _has_trig(e), n.factors)

# the arguments of the trigonometric and hyperbolic functions in a tree
_trig_args!(s, n::Node) = s
function _trig_args!(s, n::CallNode)
    n.f in _TRIG && length(n.args) == 1 && push!(s, n.args[1])
    foreach(a -> _trig_args!(s, a), n.args)
    return s
end
_trig_args!(s, n::AddNode) = (foreach(((m, c),) -> _trig_args!(s, m), n.terms); s)
_trig_args!(s, n::MulNode) = (foreach(((b, e),) -> (_trig_args!(s, b); _trig_args!(s, e)), n.factors); s)

# the number of nodes in a tree, as a measure of how long it is written
_size(n::Union{NumNode, SymNode}) = 1
_size(n::AddNode) = sum((_size(m) + 1 for (m, c) in n.terms); init = 0)
_size(n::MulNode) = 1 + sum((_size(b) + (isequal(e, _ONE) ? 0 : _size(e)) for (b, e) in n.factors); init = 0)
_size(n::CallNode) = 1 + sum((_size(a) for a in n.args); init = 0)
_size(n::GenNode) = 1 + length(n.key) ÷ 4
