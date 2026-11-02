# The sign of an expression from what is known of its variables, and simplification by it.
#
# A sign is the set of signs a value can have, among negative, zero and positive. It is first
# worked out through the expression's structure: sums, products and powers combine their parts'
# signs, and functions such as exp, abs and ln have known signs. When that leaves it open, the
# expression's polynomial skeleton (each function call and fractional power a new variable of
# known sign) goes to `positivity`, which reads coefficient signs and, with JuMP and Clarabel,
# looks for a sum of squares.

# (can be negative, can be zero, can be positive)
const _Signs = NTuple{3, Bool}
const _S_POS, _S_NONNEG, _S_ZERO, _S_NONPOS, _S_NEG, _S_ANY =
    (false, false, true), (false, true, true), (false, true, false), (true, true, false), (true, false, false), (true, true, true)

const _SIGN_NAMES = Dict{_Signs, Symbol}(_S_POS => :positive, _S_NONNEG => :nonnegative, _S_ZERO => :zero,
                                         _S_NONPOS => :nonpositive, _S_NEG => :negative)
_sign_name(s::_Signs) = get(_SIGN_NAMES, s, :unknown)

_flip(s::_Signs) = (s[3], s[2], s[1])
_values(s::_Signs) = [v for (v, ok) in zip((-1, 0, 1), s) if ok]
_from_values(vs) = (-1 in vs, 0 in vs, 1 in vs)
_number_sign(v::Real) = v > 0 ? _S_POS : v < 0 ? _S_NEG : _S_ZERO

function _sum_sign(a::_Signs, b::_Signs)
    neg = a[1] || b[1]
    pos = a[3] || b[3]
    zero = (a[2] && b[2]) || (a[1] && b[3]) || (a[3] && b[1])
    return (neg, zero, pos)
end
_product_sign(a::_Signs, b::_Signs) = _from_values([x * y for x in _values(a) for y in _values(b)])

function _power_sign(b::_Signs, e::Node)
    e isa NumNode && e.v isa Real || return b == _S_POS ? _S_POS : _S_ANY
    if e.v isa Integer || (e.v isa Rational && isone(denominator(e.v)))
        k = Integer(e.v)
        s = iseven(k) ? (false, b[2], b[1] || b[3]) : b
        return k < 0 ? (s[1], false, s[3]) : s            # b^-k: b is not zero
    end
    # a fractional power is real and nonnegative only for a nonnegative base
    b[1] && return _S_ANY
    return e.v < 0 ? (false, false, true) : b
end

const _SAME_SIGN_FUNCTIONS = Set(["sinh", "tanh", "atan", "asinh", "atanh", "erf", "sign", "cbrt"])

# the signs a variable can have, from hints, its declaration, or GIAC's assumptions
function _variable_sign(name::String, hints)
    haskey(hints, name) && return hints[name]
    name in ("pi", "e", "euler_gamma") && return _S_POS
    name in _GIAC_NAMED_VALUES && return _S_ANY
    tag = _declared_tag(name)
    s = tag === :positive ? _S_POS : tag === :negative ? _S_NEG : tag === :nonnegative ? _S_NONNEG :
        tag === :nonpositive ? _S_NONPOS :
        tag isa Tuple && length(tag) == 2 && tag[1] isa Real && tag[1] >= 0 ? _S_POS :
        tag isa Tuple && length(tag) == 2 && tag[2] isa Real && tag[2] <= 0 ? _S_NEG : nothing
    if s === nothing
        # an assumption made with `assuming` is GIAC state: ask GIAC
        g = try
            _node(Commands.invoke_cmd(:sign, giac_identifier(name)))
        catch
            nothing
        end
        s = g isa NumNode && g.v isa Real && !iszero(g.v) ? _number_sign(g.v) : _S_ANY
    end
    hints[name] = s
    return s
end

# the signs from structure alone
_signs(n::NumNode, hints) = n.v isa Real ? _number_sign(n.v) : _S_ANY
_signs(n::SymNode, hints) = _variable_sign(n.name, hints)
_signs(n::GenNode, hints) = _S_ANY
function _signs(n::AddNode, hints)
    s = _S_ZERO
    for (m, c) in n.terms
        s = _sum_sign(s, _product_sign(_signs(m, hints), c isa Real ? _number_sign(c) : _S_ANY))
        s == _S_ANY && return s
    end
    return s
end
function _signs(n::MulNode, hints)
    s = n.coef isa Real ? _number_sign(n.coef) : _S_ANY
    for (b, e) in n.factors
        s = _product_sign(s, _power_sign(_signs(b, hints), e))
    end
    return s
end
function _signs(n::CallNode, hints)
    length(n.args) == 1 || return _S_ANY
    a = n.args[1]
    n.f in ("exp", "cosh") && return _S_POS
    if n.f == "abs"
        s = _signs(a, hints)
        return s[2] ? _S_NONNEG : _S_POS
    end
    n.f in ("ln", "log") && return _signs(_add(a, NumNode(-1)), hints)
    n.f == "sqrt" && (s = _signs(a, hints); return s[1] ? _S_ANY : s)
    n.f in _SAME_SIGN_FUNCTIONS && return _signs(a, hints)
    return _S_ANY
end

# The polynomial skeleton: each function call, fractional power and GIAC object replaced by a new
# variable, recorded in `atoms`
function _skeleton(n::Node, atoms)
    (n isa NumNode || n isa SymNode) && return n
    n isa AddNode && return _add((_mul(NumNode(c), _skeleton(m, atoms)) for (m, c) in n.terms)...)
    if n isa MulNode
        parts = Node[NumNode(n.coef)]
        for (b, e) in n.factors
            whole = e isa NumNode && (e.v isa Integer || (e.v isa Rational && isone(denominator(e.v))))
            push!(parts, whole ? _pow(_skeleton(b, atoms), e) : _atom(_pow(b, e), atoms))
        end
        return _mul(parts...)
    end
    return _atom(n, atoms)
end
_atom(n::Node, atoms) = get!(() -> SymNode("sgnatom" * string(length(atoms) + 1)), atoms, n)

# The sign by `positivity` on the skeleton, each variable of known sign passed as positive (a
# negative one written as minus a positive one); nothing when it decides nothing more
function _skeleton_signs(n::Node, hints, sos)
    isempty(methods(positivity)) && return nothing
    atoms = Dict{Node, SymNode}()
    sk = _skeleton(n, atoms)
    signs = Dict{String, _Signs}(a.name => _signs(m, hints) for (m, a) in atoms)
    for name in _symbol_names(sk)
        haskey(signs, name) || (signs[name] = _variable_sign(name, hints))
    end
    positive = GiacExpr[]
    flips = Pair{GiacExpr, GiacExpr}[]
    weak = false
    for (name, s) in signs
        v = giac_identifier(name)
        if !s[1] && s[3]
            push!(positive, v)
            weak |= s[2]
        elseif s[1] && !s[3]
            w = giac_identifier(name * "′")
            push!(positive, w)
            push!(flips, v => -w)
            weak |= s[2]
        elseif s == _S_ZERO
            push!(flips, v => convert(GiacExpr, 0))
        end
    end
    e = GiacExpr(sk)
    isempty(flips) || (e = substitute(e, flips...))
    r = try
        positivity(e; positive = Tuple(positive), sos)
    catch err
        err isa Union{ArgumentError, ErrorException, MethodError} || rethrow()
        return nothing
    end
    r.sign == 0 && return nothing
    strict = r.strict && !weak
    return r.sign > 0 ? (strict ? _S_POS : _S_NONNEG) : (strict ? _S_NEG : _S_NONPOS)
end

function _decided_signs(n::Node, hints, sos)
    s = _signs(n, hints)
    count(s) == 1 && return s
    (n isa AddNode || n isa MulNode) || return s
    t = _skeleton_signs(n, hints, sos)
    t === nothing && return s
    # both hold: keep what they agree may happen
    both = (s[1] && t[1], s[2] && t[2], s[3] && t[3])
    return any(both) ? both : t
end

function _sign_hints(positive, negative)
    hints = Dict{String, _Signs}(string(v) => _S_POS for v in positive)
    foreach(v -> hints[string(v)] = _S_NEG, negative)
    return hints
end

"""
    sign_of(e; positive = (), negative = (), sos = true)

The sign of `e`, as far as it can be decided: `:positive`, `:negative`, `:zero`,
`:nonnegative`, `:nonpositive`, or `:unknown`.

What is known of a variable comes from `positive` and `negative`, from its declaration
(`@giac_var k::positive`, `k::(0, 1)`), and from assumptions in force (`assuming(k > 0)`). The
sign is worked out through sums, products and powers, and functions with known signs (`exp`,
`cosh`, `abs`, `sqrt`, `ln`, odd functions such as `sinh` and `atan`). When that leaves it open,
the polynomial skeleton (each function call a variable of the sign found for it) goes to
[`positivity`](@ref), with Nemo loaded: coefficient signs, then with JuMP and Clarabel a sum of
squares (`sos = false` to skip it).

```julia
@giac_var x y k::positive
sign_of(exp(x) + k)                            # :positive
sign_of(x^2 - 2x*y + y^2)                      # :nonnegative (a sum of squares)
sign_of(log(1 + k))                            # :positive
```
"""
sign_of(e; positive = (), negative = (), sos = true) =
    _sign_name(_decided_signs(_node(convert(GiacExpr, e)), _sign_hints(positive, negative), sos))

# ---------------------------------------------------------------------------------------------
# Simplifying by signs
# ---------------------------------------------------------------------------------------------

_resolve(n::Node, hints, sos) = n
_resolve(n::AddNode, hints, sos) = _add((_mul(NumNode(c), _resolve(m, hints, sos)) for (m, c) in n.terms)...)
_resolve(n::MulNode, hints, sos) =
    _mul(NumNode(n.coef), (_pow(_resolve(b, hints, sos), _resolve(e, hints, sos)) for (b, e) in n.factors)...)
function _resolve(n::CallNode, hints, sos)
    args = Node[_resolve(a, hints, sos) for a in n.args]
    if n.f == "abs" && length(args) == 1
        s = _decided_signs(args[1], hints, sos)
        !s[1] && return args[1]
        !s[3] && return _negated(args[1])
    elseif n.f == "sign" && length(args) == 1
        s = _decided_signs(args[1], hints, sos)
        count(s) == 1 && return NumNode(s[3] ? 1 : s[1] ? -1 : 0)
    elseif n.f in ("max", "min") && length(args) == 2
        s = _decided_signs(_add(args[1], _neg(args[2])), hints, sos)
        larger = !s[1] ? 1 : !s[3] ? 2 : 0
        larger == 0 || return args[n.f == "max" ? larger : 3 - larger]
    end
    return args == n.args ? n : _call(n.f, args)
end

"""
    simplify_signs(e; positive = (), negative = (), sos = true)

`e` with `abs`, `sign`, `max` and `min` resolved wherever [`sign_of`](@ref) decides the sign of
their arguments: `abs(u)` becomes `u` or `-u`, `sign(u)` a number, `max(a, b)` and `min(a, b)` one
of their arguments.

```julia
@giac_var x a::positive b::positive
simplify_signs(sqrt((a - b)^2 + 1) * abs(a*b + 1))     # sqrt((a-b)^2+1)*(a*b+1)
simplify_signs(max(a^2 + b^2, 2a*b))                    # a^2+b^2
```
"""
simplify_signs(e; positive = (), negative = (), sos = true) =
    GiacExpr(_resolve(_node(convert(GiacExpr, e)), _sign_hints(positive, negative), sos))
