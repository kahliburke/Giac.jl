# The symbolic core: expressions as Julia trees, with arithmetic (+ - * / ^) done in Julia. GIAC
# does everything else. A `GiacExpr` holds a tree, a GIAC object, or both, and makes the other
# when it is needed (see types.jl).
#
# The forms are normal as they are built: a sum is a list of distinct monomials with numeric
# coefficients, a product a numeric coefficient and a list of distinct bases with exponents.
# Terms and factors keep the order they first appeared in; like terms merge where the first one
# stood. Hashes and equality do not depend on the order.

# ---------------------------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------------------------

struct NumNode <: Node
    v::Number
    h::UInt
end

struct SymNode <: Node
    name::String
    h::UInt
end

# a function call: `sin(x)`, `of(q, t)` (an undefined function applied), `=` (an equation);
# `gen` is the GIAC object it was read from, if any, reused when it goes back to GIAC
struct CallNode <: Node
    f::String
    args::Vector{Node}
    h::UInt
    gen::Any
end

# monomial => coefficient; the constant term is under `_UNIT`
struct AddNode <: Node
    terms::Vector{Pair{Node, Number}}
    h::UInt
end

# a numeric coefficient and base => exponent
struct MulNode <: Node
    coef::Number
    factors::Vector{Pair{Node, Node}}
    h::UInt
end

# a GIAC object the core does not model (a vector, a string, ...), kept as GIAC has it
struct GenNode <: Node
    expr::GiacExpr
    key::String
    h::UInt
end

Base.hash(n::Node, h::UInt) = hash(n.h, h)
Base.isequal(a::Node, b::Node) = a.h == b.h && _same(a, b)
Base.:(==)(a::Node, b::Node) = isequal(a, b)

_same(a::Node, b::Node) = false
_same(a::NumNode, b::NumNode) = isequal(a.v, b.v) && (a.v isa AbstractFloat) == (b.v isa AbstractFloat)
_same(a::SymNode, b::SymNode) = a.name == b.name
_same(a::CallNode, b::CallNode) = a.f == b.f && length(a.args) == length(b.args) && all(isequal.(a.args, b.args))
_same(a::GenNode, b::GenNode) = a.key == b.key
function _same(a::AddNode, b::AddNode)
    length(a.terms) == length(b.terms) || return false
    return all(t -> any(u -> isequal(first(t), first(u)) && _same_number(last(t), last(u)), b.terms), a.terms)
end
function _same(a::MulNode, b::MulNode)
    (_same_number(a.coef, b.coef) && length(a.factors) == length(b.factors)) || return false
    return all(f -> any(g -> isequal(first(f), first(g)) && isequal(last(f), last(g)), b.factors), a.factors)
end
_same_number(a, b) = isequal(a, b) && (a isa AbstractFloat) == (b isa AbstractFloat)

# order-independent hashes for sums and products
_mix(a::UInt, b::UInt) = hash(a, b)
NumNode(v::Number) = (c = _canonical(v); NumNode(c, hash(c, c isa AbstractFloat ? 0x666c6f6174 % UInt : 0x6e756d % UInt)))
SymNode(name::AbstractString) = SymNode(String(name), hash(name, 0x73796d % UInt))
CallNode(f::AbstractString, args::Vector{Node}, gen = nothing) =
    CallNode(String(f), args, foldl((h, a) -> hash(a, h), args; init = hash(f, 0x63616c6c % UInt)), gen)
AddNode(terms::Vector{Pair{Node, Number}}) =
    AddNode(terms, foldl((h, t) -> h + _mix(first(t).h, hash(last(t))), terms; init = 0x616464 % UInt))
MulNode(coef::Number, factors::Vector{Pair{Node, Node}}) =
    MulNode(coef, factors, foldl((h, f) -> h + _mix(first(f).h, last(f).h), factors; init = hash(coef, 0x6d756c % UInt)))
GenNode(e::GiacExpr, key::String) = GenNode(e, key, hash(key, 0x67656e % UInt))

# ---------------------------------------------------------------------------------------------
# Numbers: exact where they can be, never overflowing
# ---------------------------------------------------------------------------------------------

_canonical(v::Bool) = Int(v)
_canonical(v::Integer) = typemin(Int) <= v <= typemax(Int) ? Int(v) : BigInt(v)
_canonical(v::Rational) = isone(denominator(v)) ? _canonical(numerator(v)) :
                          (typemin(Int) < numerator(v) <= typemax(Int) && denominator(v) <= typemax(Int) ?
                           Rational{Int}(v) : Rational{BigInt}(v))
_canonical(v::AbstractFloat) = Float64(v)
_canonical(v::Complex) = (iszero(imag(v)) && !(imag(v) isa AbstractFloat)) ? _canonical(real(v)) :
                         Complex(_canonical(real(v)), _canonical(imag(v)))
_canonical(v::Number) = v

_exact(v) = v isa Union{Integer, Rational} || (v isa Complex && real(v) isa Union{Integer, Rational})

# arithmetic on numbers, redone with big integers on overflow
_widen(v::Integer) = big(v)
_widen(v::Rational) = Rational{BigInt}(v)
_widen(v::Complex) = Complex(_widen(real(v)), _widen(imag(v)))
_widen(v) = v
function _checked(op, a, b)
    if a isa Int && b isa Int
        r, overflow = op === (+) ? Base.Checked.add_with_overflow(a, b) :
                      op === (*) ? Base.Checked.mul_with_overflow(a, b) : (op(a, b), false)
        return _canonical(overflow ? op(big(a), big(b)) : r)
    end
    return _canonical(try
        op(a, b)
    catch err
        err isa OverflowError || rethrow()
        op(_widen(a), _widen(b))
    end)
end
_nadd(a, b) = _checked(+, a, b)
_nmul(a, b) = _checked(*, a, b)
function _npow(a, k::Integer)
    if _exact(a)
        k < 0 && iszero(a) && return nothing
        r = k >= 0 ? _widen(a)^k : (1 // _widen(a))^(-k)
        return _canonical(r)
    end
    return _canonical(a^k)
end

_isone_exact(c) = c isa Integer && isone(c)
_negative(c) = c isa Real && c < 0

const _ZERO = NumNode(0)
const _ONE = NumNode(1)
const _MINUS_ONE = NumNode(-1)
const _HALF = NumNode(1 // 2)
const _UNIT = _ONE                      # the monomial of a constant term

# ---------------------------------------------------------------------------------------------
# Building normal forms
# ---------------------------------------------------------------------------------------------

# a term as monomial and coefficient
function _split_coef(n::Node)
    n isa NumNode && return (_UNIT, n.v)
    if n isa MulNode && !_isone_exact(n.coef)
        rest = length(n.factors) == 1 && isequal(last(n.factors[1]), _ONE) ? first(n.factors[1]) : MulNode(1, n.factors)
        return (rest, n.coef)
    end
    return (n, 1)
end

# a monomial times a coefficient
function _scale(m::Node, c::Number)
    iszero(c) && return NumNode(c isa AbstractFloat ? 0.0 : 0)
    (m === _UNIT || isequal(m, _UNIT)) && return NumNode(c)
    isone(c) && return m
    # an integer's root takes in the integer's powers in the new coefficient (see `_mul`)
    _has_numeric_root(m) && return _mul(NumNode(c), m)
    return m isa MulNode ? MulNode(c, m.factors) : MulNode(c, Pair{Node, Node}[m => _ONE])
end

_has_numeric_root(m::Node) = false
_has_numeric_root(m::MulNode) = any(((b, e),) -> b isa NumNode && b.v isa Integer && e isa NumNode && e.v isa Rational, m.factors)

_add(a::NumNode, b::NumNode) = NumNode(_nadd(a.v, b.v))

function _add(xs::Node...)
    terms = Pair{Node, Number}[]
    # like terms found by a scan while the sum is small, through an index beyond that
    n = sum(x -> x isa AddNode ? length(x.terms) : 1, xs)
    index = n > 12 ? Dict{Node, Int}() : nothing
    function merge!(m, c)
        if m isa AddNode
            # a number times a sum is distributed, so its terms merge with the others
            foreach(((m2, c2),) -> merge!(m2, _nmul(c, c2)), m.terms)
            return
        end
        i = index === nothing ? something(findfirst(t -> isequal(first(t), m), terms), 0) : get(index, m, 0)
        if i == 0
            push!(terms, m => c)
            index === nothing || (index[m] = length(terms))
        else
            terms[i] = m => _nadd(last(terms[i]), c)
        end
    end
    for x in xs
        for (m, c) in (x isa AddNode ? x.terms : (_split_coef(x),))
            merge!(m, c)
        end
    end
    kept = filter(t -> !iszero(last(t)), terms)
    isempty(kept) && return any(t -> last(t) isa AbstractFloat, terms) ? NumNode(0.0) : _ZERO
    length(kept) == 1 && return _scale(first(kept[1]), last(kept[1]))
    return AddNode(kept)
end

function _mul(xs::Node...)
    coef = 1
    factors = Pair{Node, Node}[]
    for x in xs
        if x isa NumNode
            coef = _nmul(coef, x.v)
        elseif x isa MulNode
            coef = _nmul(coef, x.coef)
            foreach(f -> _push_factor!(factors, first(f), last(f)), x.factors)
        else
            _push_factor!(factors, x, _ONE)
        end
    end
    _merge_exponentials!(factors)
    # numeric bases whose exponents became integers fold into the coefficient
    kept = Pair{Node, Node}[]
    for (b, e) in factors
        isequal(e, _ZERO) && continue
        x = _exp_power(b, e)
        if x !== nothing
            push!(kept, x => _ONE)
        elseif b isa NumNode && e isa NumNode && e.v isa Integer
            p = _npow(b.v, e.v)
            p === nothing ? push!(kept, b => e) : (coef = _nmul(coef, p))
        elseif b isa NumNode && b.v isa Integer && b.v > 1 && e isa NumNode && e.v isa Rational && coef isa Union{Integer, Rational}
            # the powers of the base in the coefficient go into its exponent, and the whole part of
            # that comes out, leaving one between -1 and 1: 3/√3 = √3, 3^(3/2) = 3√3, √3/3 = 1/√3
            k = _multiplicity(b.v, coef)
            t = e.v + k
            w = trunc(Integer, t)
            coef = _nmul(coef, _npow(b.v, w - k))
            push!(kept, b => NumNode(_canonical(t - w)))
        else
            push!(kept, b => e)
        end
    end
    iszero(coef) && return NumNode(coef)
    isempty(kept) && return NumNode(coef)
    _isone_exact(coef) && length(kept) == 1 && isequal(last(kept[1]), _ONE) && return first(kept[1])
    return MulNode(coef, kept)
end

# the power of the integer b > 1 in the rational c: 2 for 18 and b = 3, -1 for 1/6 and b = 2
function _multiplicity(b::Integer, c)
    k = 0
    num, den = numerator(c), denominator(c)
    while !iszero(num) && iszero(rem(num, b))
        num = div(num, b)
        k += 1
    end
    while iszero(rem(den, b))
        den = div(den, b)
        k -= 1
    end
    return k
end

# e^a e^b as e^{a+b}, for exponentials to whole powers
function _merge_exponentials!(factors)
    is_exp((b, e)) = b isa CallNode && b.f == "exp" && length(b.args) == 1 && e isa NumNode && e.v isa Integer
    idx = findall(is_exp, factors)
    (length(idx) > 1 || length(idx) == 1 && !isequal(last(factors[idx[1]]), _ONE)) || return factors
    a = _add((_mul(first(factors[i]).args[1], last(factors[i])) for i in idx)...)
    deleteat!(factors, idx)
    _REWRITTEN[] = true
    m = _exp(a)
    isequal(m, _ONE) || push!(factors, m => _ONE)
    return factors
end

function _push_factor!(factors, b::Node, e::Node)
    for (i, (b2, e2)) in pairs(factors)
        if isequal(b2, b)
            factors[i] = b => (e2 isa NumNode && e isa NumNode ? NumNode(_nadd(e2.v, e.v)) : _add(e2, e))
            return
        end
    end
    push!(factors, b => e)
end

function _pow(b::Node, e::Node)
    isequal(e, _ZERO) && return _ONE
    isequal(e, _ONE) && return b
    x = _exp_power(b, e)
    x === nothing || return x
    if b isa NumNode && e isa NumNode
        if e.v isa Integer
            p = _npow(b.v, e.v)
            p === nothing && return _giac_node("($(_str(b)))^($(_str(e)))")
            return NumNode(p)
        end
        r = _numeric_root(b.v, e.v)
        r === nothing || return r
        return _mul(MulNode(1, Pair{Node, Node}[b => e]))
    end
    if b isa MulNode && e isa NumNode && e.v isa Integer
        # an integer power distributes over a product
        c = _npow(b.coef, e.v)
        c === nothing && return _giac_node("($(_str(b)))^($(_str(e)))")
        return _mul(NumNode(c), (_pow(f, _mul(x, e)) for (f, x) in b.factors)...)
    end
    if b isa MulNode && e isa NumNode && e.v isa Real
        r = _nonnegative_power(b, e)
        r === nothing || return r
    end
    return MulNode(1, Pair{Node, Node}[b => e])
end

# A fractional power of a product, distributed over the factors known not to be negative (and a
# positive coefficient): √(4m²ω²) = 2mω for m, ω ≥ 0. `nothing` if none is.
# The roots stay apart, √2 √m, so that they cancel and combine; the typesetter gathers them.
function _nonnegative_power(b::MulNode, e::NumNode)
    c = b.coef
    known = [f for f in b.factors if _known_nonnegative(first(f)) && last(f) isa NumNode && last(f).v isa Real]
    split_coef = c isa Real && c > 0 && !isone(c)
    (isempty(known) && !split_coef) && return nothing
    rest = Pair{Node, Node}[f for f in b.factors if !any(k -> k === f, known)]
    parts = Node[split_coef ? _pow(NumNode(c), e) : _ONE]
    foreach(((f, x),) -> push!(parts, _pow(f, _mul(x, e))), known)
    rc = split_coef ? 1 : c
    if !isempty(rest) || !isone(rc)
        r = isempty(rest) ? NumNode(rc) : length(rest) == 1 && isone(rc) && isequal(last(rest[1]), _ONE) ? first(rest[1]) : MulNode(rc, rest)
        push!(parts, MulNode(1, Pair{Node, Node}[r => e]))
    end
    return _mul(parts...)
end

# b = s^n k with the largest nth power s^n that divides b (searched for b ≤ 10^8)
function _root_parts(b::Integer, n::Integer)
    b <= 1 && return (b, 1)
    r = round(BigInt, big(b)^(1 / big(n)))
    for s in (r + 1, r, r - 1)
        s >= 1 && big(s)^n == b && return (_canonical(s), 1)
    end
    if b <= 10^8
        for s in r:-1:2
            k, rem = divrem(b, big(s)^n)
            rem == 0 && return (_canonical(s), _canonical(k))
        end
    end
    return (1, b)
end

# whether a tree has a floating-point number in it
_has_float(n::Node) = false
_has_float(n::NumNode) = n.v isa AbstractFloat || (n.v isa Complex && real(n.v) isa AbstractFloat)
_has_float(n::AddNode) = any(((m, c),) -> c isa AbstractFloat || _has_float(m), n.terms)
_has_float(n::MulNode) = n.coef isa AbstractFloat || any(((b, e),) -> _has_float(b) || _has_float(e), n.factors)
_has_float(n::CallNode) = any(_has_float, n.args)
# a multiprecision float (at Digits above 15), or a complex number with one, is kept as GIAC has it
function _has_float(n::GenNode)
    t = giac_type(n.expr)
    return t in (GenTypes.REAL, GenTypes.FLOAT) || (t == GenTypes.CPLX && occursin('.', n.key))
end

# Whether a node is known not to be negative: a number ≥ 0, a name declared positive or
# nonnegative (or in an interval above 0), pi, and sums, products and powers of such
_known_nonnegative(n::Node) = false
_known_nonnegative(n::NumNode) = n.v isa Real && n.v >= 0
function _known_nonnegative(n::SymNode)
    n.name == "pi" && return true
    tag = _declared_tag(n.name)
    return tag === :positive || tag === :nonnegative || (tag isa Tuple && length(tag) == 2 && tag[1] isa Real && tag[1] >= 0)
end
_known_nonnegative(n::MulNode) = n.coef isa Real && n.coef >= 0 &&
                                 all(((b, e),) -> _known_nonnegative(b) && e isa NumNode && e.v isa Real, n.factors)
_known_nonnegative(n::AddNode) = all(((m, c),) -> c isa Real && c >= 0 && (m isa NumNode || _known_nonnegative(m)), n.terms)
_known_nonnegative(n::CallNode) = n.f == "exp" && length(n.args) == 1 && _known_real(n.args[1])
_known_real(n::Node) = _known_nonnegative(n) || (n isa NumNode && n.v isa Real) ||
                       (n isa SymNode && _declared_tag(n.name) in (:real, :integer, :negative, :nonpositive))

# b^e for numbers with a fractional exponent: computed for floats, exact for perfect powers
function _numeric_root(b, e)
    (b isa Real && e isa Real) || return nothing
    if b isa AbstractFloat || e isa AbstractFloat
        return b >= 0 ? NumNode(Float64(b)^Float64(e)) : nothing
    end
    # a fraction: the root of its numerator over the root of its denominator, √(1/2) = 1/√2
    if b isa Rational && b > 0 && e isa Rational
        part(k, x) = isone(k) ? _ONE : something(_numeric_root(k, x), MulNode(1, Pair{Node, Node}[NumNode(k) => NumNode(x)]))
        return _mul(part(numerator(b), e), part(denominator(b), -e))
    end
    (b isa Integer && b >= 0 && e isa Rational) || return nothing
    iszero(b) && return e > 0 ? _ZERO : nothing
    # the largest qth power that divides b comes out: 8^(1/2) = 2·2^(1/2)
    s, k = _root_parts(b, denominator(e))
    out = NumNode(_npow(_canonical(s), numerator(e)))
    isone(k) && return out
    isone(s) && return nothing
    return _mul(out, MulNode(1, Pair{Node, Node}[NumNode(k) => NumNode(e)]))
end

# exp(c)^k as exp(c*k) for a real number c: e^2 is exp(2), e^x is exp(x)
# exp(a)^k as exp(k a), for a real number a or a whole power k: e^2 is exp(2), (e^x)^2 is e^{2x}
function _exp_power(b::Node, e::Node)
    (b isa CallNode && b.f == "exp" && length(b.args) == 1 && !isequal(e, _ONE)) || return nothing
    (b.args[1] isa NumNode && b.args[1].v isa Real || e isa NumNode && e.v isa Integer) || return nothing
    return _exp(_mul(b.args[1], e))
end

_exp(a::Node) = isequal(a, _ZERO) ? _ONE : CallNode("exp", Node[a])

# A function applied, written with its argument's sign taken out where the function is even or
# odd: cos(b - a) is cos(a - b) whichever way GIAC wrote it, sin(-x) is -sin(x)
# Set when reading a GIAC object rewrites its form (under the GIAC lock; see `_node`)
const _REWRITTEN = Ref(false)

const _EVEN_FUNCTIONS = Set(["cos", "cosh", "sec", "sech"])
const _ODD_FUNCTIONS = Set(["sin", "tan", "sinh", "tanh", "asin", "atan", "asinh", "atanh", "csc", "cot", "csch", "coth"])
function _call(h::String, args::Vector{Node}, gen = nothing)
    if (h == "cos" || h == "sin") && length(args) == 1
        x = _pi_shifted(h, args[1])
        x === nothing || return x
    end
    even, odd = h in _EVEN_FUNCTIONS, h in _ODD_FUNCTIONS
    ((even || odd) && length(args) == 1 && _negated_first(args[1], even)) || return CallNode(h, args, gen)
    _REWRITTEN[] = true
    f = CallNode(h, Node[_negate(args[1])])
    return even ? f : _neg(f)
end

# cos(u + jπ/2) and sin(u + jπ/2) with the multiple of π/2 taken out: cos(u + π) = -cos(u),
# sin(u + π/2) = cos(u). `nothing` if the argument has no such term.
function _pi_shifted(h::String, a::Node)
    x = a isa MulNode && any(f -> first(f) isa AddNode, a.factors) ? _expand(a) : a
    x isa AddNode || return nothing
    i = findfirst(((m, c),) -> m isa SymNode && m.name == "pi" && c isa Union{Integer, Rational} && isinteger(2c), x.terms)
    i === nothing && return nothing
    u = _add((_scale(m, c) for (k, (m, c)) in enumerate(x.terms) if k != i)...)
    isequal(u, _ZERO) && return nothing
    j = mod(Int(2 * x.terms[i][2]), 4)
    _REWRITTEN[] = true
    f = iseven(j) ? h : (h == "cos" ? "sin" : "cos")
    negative = h == "cos" ? j in (1, 2) : j in (2, 3)
    r = _call(f, Node[u])
    return negative ? _neg(r) : r
end

# Whether -u is the argument to write: more negative terms than positive ones, and a tie decided
# by the hash, so that u and -u settle on the same one
function _negated_first(u::Node, even::Bool)
    if u isa MulNode
        return u.coef isa Real && u.coef < 0
    elseif u isa AddNode
        cs = [c for (m, c) in u.terms if !(m isa NumNode)]
        all(c -> c isa Real, cs) || return false
        neg, pos = count(<(0), cs), count(>(0), cs)
        neg == pos || return neg > pos
        # as many of each: only an even function, which takes no sign, settles on one of u and -u
        return even && u.h > _negate(u).h
    end
    return u isa NumNode && u.v isa Real && u.v < 0
end

_neg(a::Node) = _mul(_MINUS_ONE, a)
# -u with a sum's terms negated, not a sum times -1
_negate(u::Node) = u isa AddNode ? AddNode(Pair{Node, Number}[m => -c for (m, c) in u.terms]) : _neg(u)
_sub(a::Node, b::Node) = _add(a, _neg(b))
function _div(a::Node, b::Node)
    b isa NumNode && iszero(b.v) && return _giac_node("($(_str(a)))/($(_str(b)))")
    return _mul(a, _pow(b, _MINUS_ONE))
end

# an edge case decided by GIAC (a division by zero, 0^-1)
_giac_node(src::String) = _node(giac_eval(src))

# ---------------------------------------------------------------------------------------------
# Printing, in GIAC's syntax
# ---------------------------------------------------------------------------------------------

_str(n::Node) = sprint(_print, n)

const _INFIX = Dict("=" => " = ", "==" => "==", "!=" => "!=", "<" => "<", ">" => ">", "<=" => "<=", ">=" => ">=")

function _print_number(io::IO, v)
    if v isa Rational
        print(io, numerator(v), "/", denominator(v))
    elseif v isa AbstractFloat
        isinf(v) ? print(io, v > 0 ? "inf" : "-inf") : isnan(v) ? print(io, "undef") : print(io, repr(v))
    elseif v isa Complex
        re, im = real(v), imag(v)
        if iszero(re)
            isone(im) ? print(io, "i") : isone(-im) ? print(io, "-i") : (_print_number(io, im); print(io, "*i"))
        else
            _print_number(io, re)
            _negative(im) ? print(io, "-") : print(io, "+")
            a = abs(im)
            isone(a) ? print(io, "i") : (_print_number(io, a); print(io, "*i"))
        end
    else
        print(io, v)
    end
end

_print(io::IO, n::NumNode) = _print_number(io, n.v)
_print(io::IO, n::SymNode) = print(io, n.name)
_print(io::IO, n::GenNode) = print(io, n.key)

function _print(io::IO, n::CallNode)
    if haskey(_INFIX, n.f) && length(n.args) == 2
        _print(io, n.args[1]); print(io, n.f == "=" ? "=" : _INFIX[n.f]); _print(io, n.args[2])
    elseif n.f == "of"
        _print_atom(io, n.args[1]); print(io, "("); _print_args(io, n.args, 2); print(io, ")")
    else
        print(io, n.f, "("); _print_args(io, n.args, 1); print(io, ")")
    end
end

function _print_args(io::IO, args, from)
    for i in from:length(args)
        i > from && print(io, ",")
        _print(io, args[i])
    end
end

# each term written directly: its sign, then its coefficient and factors
function _print(io::IO, n::AddNode)
    for (i, (m, c)) in pairs(n.terms)
        neg = _negative(c)
        neg ? print(io, "-") : i > 1 && print(io, "+")
        a = neg ? -c : c
        m isa NumNode ? _print_number(io, a) : _print_product(io, a, m isa MulNode ? m.factors : (m => _ONE,))
    end
end

function _print(io::IO, n::MulNode)
    neg = _negative(n.coef)
    neg && print(io, "-")
    _print_product(io, neg ? -n.coef : n.coef, n.factors)
end

# a product with a coefficient that is not negative: the coefficient's numerator and the factors
# with nonnegative exponents above the line, its denominator and the others below
function _print_product(io::IO, c, factors)
    p, q = c isa Rational ? (numerator(c), denominator(c)) : (c, 1)
    k = 0
    if !isone(p)
        p isa Complex ? (print(io, "("); _print_number(io, p); print(io, ")")) : _print_number(io, p)
        k += 1
    end
    nden = isone(q) ? 0 : 1
    for (b, e) in factors
        if e isa NumNode && _negative(e.v)
            nden += 1
        else
            k > 0 && print(io, "*")
            _print_factor(io, b, e)
            k += 1
        end
    end
    k == 0 && print(io, "1")
    nden == 0 && return
    print(io, "/")
    nden > 1 && print(io, "(")
    j = 0
    if !isone(q)
        print(io, q)
        j += 1
    end
    for (b, e) in factors
        if e isa NumNode && _negative(e.v)
            j > 0 && print(io, "*")
            _print_factor(io, b, NumNode(-e.v))
            j += 1
        end
    end
    nden > 1 && print(io, ")")
end

function _print_factor(io::IO, b::Node, e::Node)
    isequal(e, _ONE) && return _print_atom(io, b)
    isequal(e, _HALF) && return (print(io, "sqrt("); _print(io, b); print(io, ")"))
    _print_atom(io, b)
    print(io, "^")
    simple = e isa NumNode && e.v isa Integer && e.v >= 0 || e isa SymNode
    simple ? _print(io, e) : (print(io, "("); _print(io, e); print(io, ")"))
end

# a factor or a base: parenthesized unless it is a name, a call, or a nonnegative integer
function _print_atom(io::IO, n::Node)
    atomic = n isa SymNode || (n isa CallNode && !haskey(_INFIX, n.f)) ||
             (n isa NumNode && n.v isa Integer && n.v >= 0) || (n isa NumNode && n.v isa AbstractFloat && n.v >= 0) ||
             (n isa GenNode && occursin(r"^[\w.\[\]]+$|^\w+\(.*\)$", n.key))
    atomic ? _print(io, n) : (print(io, "("); _print(io, n); print(io, ")"))
end

# ---------------------------------------------------------------------------------------------
# GIAC's shape of a node: the head and arguments GIAC itself would show
# ---------------------------------------------------------------------------------------------

# The head alone, as `_view` would give it, without building the arguments
_view_head(n::Node) = nothing
_view_head(n::CallNode) = n.f
_view_head(n::AddNode) = "+"
function _view_head(n::MulNode)
    c = n.coef
    (isone(-c) || _negative(c)) && return "-"
    if isone(c) && length(n.factors) == 1
        e = last(n.factors[1])
        return e isa NumNode && _negative(e.v) ? "inv" : "^"
    end
    return "*"
end

# (head, arguments) for a compound node, `nothing` for an atom
_view(n::Node) = nothing
_view(n::CallNode) = (n.f, n.args)
_view(n::AddNode) = ("+", Node[_scale(m, c) for (m, c) in n.terms])
function _view(n::MulNode)
    c = n.coef
    rest = length(n.factors) == 1 && isequal(last(n.factors[1]), _ONE) ? first(n.factors[1]) : MulNode(1, n.factors)
    isone(-c) && return ("-", Node[rest])
    _negative(c) && return ("-", Node[_scale(rest, -c)])       # -2*x as -(2*x)
    if isone(c) && length(n.factors) == 1
        b, e = n.factors[1]
        e isa NumNode && _negative(e.v) && return ("inv", Node[_pow(b, NumNode(-e.v))])
        return ("^", Node[b, e])
    end
    args = Node[]
    isone(c) || push!(args, NumNode(c))
    for (b, e) in n.factors
        push!(args, _pow(b, e))
    end
    return ("*", args)
end

# GIAC's type tag for a node
function _node_type(n::Node)
    n isa SymNode && return IDNT
    n isa GenNode && return giac_type(n.expr)
    if n isa NumNode
        v = n.v
        return v isa Int ? INT : v isa Integer ? ZINT : v isa Rational ? FRAC : v isa AbstractFloat ? DOUBLE : CPLX
    end
    return SYMB
end

# ---------------------------------------------------------------------------------------------
# From GIAC
# ---------------------------------------------------------------------------------------------

const _CALL_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"

# The tree of a GIAC object (with the GIAC lock held). A wrapper with `gen_export` writes the
# whole tree in one call; otherwise it is read node by node.
_node_from_gen(gen) = _has_export() ? _node_from_export(gen) : _node_from_gen_by_node(gen)

_has_export() = isdefined(GiacCxxBindings, :gen_export)

const _EXPORT_HEADS = join(keys(_INFIX), "\n")

function _node_from_export(gen)
    B = GiacCxxBindings
    buf = Vector{UInt8}(codeunits(String(B.gen_export(gen, _EXPORT_HEADS))))
    pos = Ref(1)
    node = GC.@preserve buf _read_node(buf, pos)
    return node === nothing ? _node_from_gen_by_node(gen) : node
end

function _read_value(::Type{T}, buf::Vector{UInt8}, pos::Ref{Int}) where {T}
    v = ltoh(unsafe_load(Ptr{T}(pointer(buf, pos[]))))
    pos[] += sizeof(T)
    return v
end

function _read_string(buf::Vector{UInt8}, pos::Ref{Int})
    n = Int(_read_value(UInt32, buf, pos))
    s = String(buf[pos[]:(pos[] + n - 1)])
    pos[] += n
    return s
end

# a node of an exported tree (see the wrapper's gen_export); `nothing` for a shape this side does
# not expect, which makes the caller read the object node by node
function _read_node(buf::Vector{UInt8}, pos::Ref{Int})
    tag = buf[pos[]]
    pos[] += 1
    if tag == 0x01
        return NumNode(Int(_read_value(Int64, buf, pos)))
    elseif tag == 0x02
        sign = _read_value(Int32, buf, pos)
        n = Int(_read_value(UInt32, buf, pos))
        magnitude = n == 0 ? big(0) : parse(BigInt, bytes2hex(view(buf, pos[]:(pos[] + n - 1))); base = 16)
        pos[] += n
        return NumNode(sign < 0 ? -magnitude : magnitude)
    elseif tag == 0x03
        return NumNode(_from_giac_double(_read_value(Float64, buf, pos)))
    elseif tag == 0x04
        return SymNode(_read_string(buf, pos))
    elseif tag == 0x05
        p, q = _read_node(buf, pos), _read_node(buf, pos)
        (p isa NumNode && q isa NumNode) || return nothing
        return NumNode(p.v // q.v)
    elseif tag == 0x06
        re, im = _read_node(buf, pos), _read_node(buf, pos)
        (re isa NumNode && im isa NumNode && re.v isa Real && im.v isa Real) || return nothing
        return NumNode(Complex(re.v, im.v))
    elseif tag == 0x07
        h = _read_string(buf, pos)
        k = Int(_read_value(UInt32, buf, pos))
        args = Vector{Node}(undef, k)
        for i in 1:k
            a = _read_node(buf, pos)
            a === nothing && return nothing
            args[i] = a
        end
        s = _signed_infinity(h, args)
        s === nothing || return s
        # GIAC leaves a product or power of integers unevaluated only as a factorization (ifactor's
        # 2^3*3^2*5): read node by node, which keeps it as GIAC has it
        h in ("*", "^") && k >= 2 && all(a -> a isa NumNode && a.v isa Integer, args) && return nothing
        h == "+" && return _add(args...)
        h == "*" && return _mul(args...)
        h == "-" && return k == 1 ? _neg(args[1]) : k == 2 ? _sub(args[1], args[2]) : nothing
        h == "inv" && return k == 1 ? _pow(args[1], _MINUS_ONE) : nothing
        h == "^" && return k == 2 ? _pow(args[1], args[2]) : nothing
        h == "/" && return k == 2 ? _div(args[1], args[2]) : nothing
        return _call(h, args)
    elseif tag == 0x09
        g = GiacCxxBindings.gen_export_opaque(Int32(_read_value(UInt32, buf, pos)))
        return GenNode(GiacExpr(_make_gen_ptr(g)), String(GiacCxxBindings.to_string(g)))
    end
    return nothing
end

# GIAC's +infinity and -infinity are + and - applied to its unsigned infinity. The tree has +∞ as
# `inf`, which GIAC reads as +infinity, so that -∞ and other arithmetic on it keep the sign.
const _PLUS_INFINITY = SymNode("inf")
_is_plus_infinity(n::Node) = n isa SymNode && n.name == "inf"
function _signed_infinity(h::String, args)
    (length(args) == 1 && args[1] isa SymNode && args[1].name == "infinity") || return nothing
    h == "+" && return _PLUS_INFINITY
    h == "-" && return _neg(_PLUS_INFINITY)
    return nothing
end

function _node_from_gen_by_node(gen)
    B = GiacCxxBindings
    t = T(B.type(gen))
    t == INT && return NumNode(Int(B.to_int64(gen)))
    t == ZINT && return NumNode(parse(BigInt, String(B.zint_to_string(gen))))
    t == DOUBLE && return NumNode(_from_giac_double(Float64(B.to_double(gen))))
    t == IDNT && return SymNode(String(B.idnt_name(gen)))
    if t == FRAC
        p, q = _node_from_gen_by_node(B.frac_num(gen)), _node_from_gen_by_node(B.frac_den(gen))
        (p isa NumNode && q isa NumNode) && return NumNode(p.v // q.v)
    elseif t == CPLX
        re, im = _node_from_gen_by_node(B.cplx_re(gen)), _node_from_gen_by_node(B.cplx_im(gen))
        (re isa NumNode && im isa NumNode && re.v isa Real && im.v isa Real) && return NumNode(Complex(re.v, im.v))
    elseif t == SYMB
        h = String(B.symb_sommet_name(gen))
        if (h in ("+", "*", "-", "inv", "^", "/") || haskey(_INFIX, h) || occursin(_CALL_NAME, h)) && !_is_factorization(gen)
            leaf = B.symb_feuille(gen)
            raw = Any[]
            if B.is_vector(leaf) && B.subtype(leaf) in (B.VECTSUBTYPE_SEQ, _SORTED_VECT)
                for i in 0:(B.vect_size(leaf) - 1)
                    _push_gen_args!(raw, B.vect_at(leaf, i))
                end
            else
                push!(raw, leaf)
            end
            args = Node[_node_from_gen_by_node(a) for a in raw]
            s = _signed_infinity(h, args)
            s === nothing || return s
            h == "+" && return _add(args...)
            h == "*" && return _mul(args...)
            h == "-" && return length(args) == 1 ? _neg(args[1]) : _sub(args[1], args[2])
            h == "inv" && length(args) == 1 && return _pow(args[1], _MINUS_ONE)
            h == "^" && length(args) == 2 && return _pow(args[1], args[2])
            h == "/" && length(args) == 2 && return _div(args[1], args[2])
            h in ("+", "*", "-", "inv", "^", "/") || return _call(h, args, gen)
        end
    end
    e = GiacExpr(_make_gen_ptr(gen))
    return GenNode(e, String(B.to_string(gen)))
end

# A product of powers of integers, as ifactor writes 2^3*3^2*5: a form, kept as GIAC has it rather
# than multiplied out
function _is_factorization(gen)
    B = GiacCxxBindings
    integer(g) = (t = T(B.type(g)); t == INT || t == ZINT)
    function power(g)
        (T(B.type(g)) == SYMB && String(B.symb_sommet_name(g)) == "^") || return false
        leaf = B.symb_feuille(g)
        return B.is_vector(leaf) && B.vect_size(leaf) == 2 && integer(B.vect_at(leaf, 0)) && integer(B.vect_at(leaf, 1))
    end
    power(gen) && return true
    (T(B.type(gen)) == SYMB && String(B.symb_sommet_name(gen)) == "*") || return false
    leaf = B.symb_feuille(gen)
    B.is_vector(leaf) || return false
    gs = [B.vect_at(leaf, i) for i in 0:(B.vect_size(leaf) - 1)]
    # GIAC multiplies integers out, so a product of them left standing is a factorization, with
    # or without powers (2^128+1 is 59649589127497217*5704689200685129054721)
    return length(gs) >= 2 && all(g -> integer(g) || power(g), gs)
end

function _push_gen_args!(out, gen)
    B = GiacCxxBindings
    if B.is_vector(gen) && B.subtype(gen) == B.VECTSUBTYPE_SEQ
        for i in 0:(B.vect_size(gen) - 1)
            _push_gen_args!(out, B.vect_at(gen, i))
        end
    else
        push!(out, gen)
    end
end

# ---------------------------------------------------------------------------------------------
# To GIAC: the object built with GIAC's own operators, without printing and parsing
# ---------------------------------------------------------------------------------------------

# The GIAC object for a tree (with the GIAC lock held)
# A wrapper with `gen_import` builds the GIAC object from a program written here, in one call. The
# program mirrors `_gen_from_node`, operation for operation, so the object is the same.
_has_import() = isdefined(GiacCxxBindings, :gen_import)

function _gen_import(n::Node)
    B = GiacCxxBindings
    io = IOBuffer()
    opaque = B.CxxWrap.StdVector{B.Gen}()
    _program!(io, n, opaque)
    return B.gen_import(String(take!(io)), opaque)
end

_put!(io, tag::UInt8) = write(io, tag)
_put_str!(io, s::AbstractString) = (write(io, htol(UInt32(ncodeunits(s)))); write(io, s))
function _put_opaque!(io, gen, opaque)
    push!(opaque, gen)
    write(io, 0x09, htol(UInt32(length(opaque) - 1)))
end

function _program!(io, n::NumNode, opaque)
    v = n.v
    if v isa Integer
        if typemin(Int64) <= v <= typemax(Int64) && typemin(Int32) <= v <= typemax(Int32)
            write(io, 0x01, htol(Int64(v)))
        else
            m = abs(big(v))
            bytes = UInt8[]
            while m > 0
                pushfirst!(bytes, UInt8(m & 0xff))
                m >>= 8
            end
            write(io, 0x02, htol(Int32(sign(v))), htol(UInt32(length(bytes))), bytes)
        end
    elseif v isa Rational
        _put!(io, 0x05)
        _program!(io, NumNode(numerator(v)), opaque)
        _program!(io, NumNode(denominator(v)), opaque)
    elseif v isa AbstractFloat
        write(io, 0x03, htol(Float64(v)))
    else
        _put!(io, 0x06)
        _program!(io, NumNode(real(v)), opaque)
        _program!(io, NumNode(imag(v)), opaque)
    end
    return io
end

# a variable goes in as the session's identifier for its name (see `_identifier_gen`)
function _program!(io, n::SymNode, opaque)
    n.name in _GIAC_NAMED_VALUES || return (_put_opaque!(io, _identifier_gen(n.name), opaque); io)
    _put!(io, 0x0a); _put_str!(io, n.name)
    return io
end
_program!(io, n::GenNode, opaque) = (_put_opaque!(io, _get_gen(n.expr.ptr), opaque); io)

function _program!(io, n::AddNode, opaque)
    write(io, 0x0b, htol(UInt32(length(n.terms))))
    foreach(((m, c),) -> _program!(io, _scale(m, c), opaque), n.terms)
    return io
end

# a product as numerator / denominator, as `_gen_from_node` builds it: factors multiplied left to
# right, the coefficient in front, the denominator divided last
function _program!(io, n::MulNode, opaque)
    num, den = Vector{UInt8}[], Vector{UInt8}[]
    for (b, e) in n.factors
        below = e isa NumNode && _negative(e.v)
        x = below ? NumNode(-e.v) : e
        f = IOBuffer()
        if isequal(x, _ONE)
            _program!(f, b, opaque)
        else
            _put!(f, 0x0d)
            _program!(f, b, opaque)
            _program!(f, x, opaque)
        end
        push!(below ? den : num, take!(f))
    end
    product(parts) = (b = IOBuffer(); foreach(_ -> _put!(b, 0x0e), 2:length(parts)); foreach(p -> write(b, p), parts); take!(b))
    numerator_program = isempty(num) ? nothing : product(num)
    c = n.coef
    if !_isone_exact(c)
        coef = take!(_program!(IOBuffer(), NumNode(c), opaque))
        numerator_program = isone(-c) && numerator_program !== nothing ? vcat(0x10, numerator_program) :
                            numerator_program === nothing ? coef : vcat(0x0e, coef, numerator_program)
    end
    if isempty(den)
        write(io, numerator_program)
    else
        _put!(io, 0x0f)
        write(io, numerator_program === nothing ? take!(_program!(IOBuffer(), _ONE, opaque)) : numerator_program)
        write(io, product(den))
    end
    return io
end

function _program!(io, n::CallNode, opaque)
    n.gen === nothing || return (_put_opaque!(io, n.gen, opaque); io)
    _put!(io, 0x07)
    _put_str!(io, n.f == "=" ? "equal" : n.f)
    if n.f == "of" && length(n.args) > 2
        # f(a, b) is of(f, seq(a, b))
        write(io, htol(UInt32(2)))
        _program!(io, n.args[1], opaque)
        write(io, 0x11, htol(UInt32(length(n.args) - 1)))
        foreach(a -> _program!(io, a, opaque), n.args[2:end])
    else
        write(io, htol(UInt32(length(n.args))))
        foreach(a -> _program!(io, a, opaque), n.args)
    end
    return io
end

function _gen_from_node(n::NumNode)
    B = GiacCxxBindings
    v = n.v
    v isa Integer && return _int_gen(v)
    v isa Rational && return B.make_fraction(_int_gen(numerator(v)), _int_gen(denominator(v)))
    v isa AbstractFloat && return B.Gen(Float64(v))
    return B.make_complex(_gen_from_node(NumNode(real(v))), _gen_from_node(NumNode(imag(v))))
end

function _int_gen(v::Integer)
    B = GiacCxxBindings
    typemin(Int32) <= v <= typemax(Int32) && return B.Gen(Int32(v))
    m, bytes = abs(big(v)), UInt8[]
    while m > 0
        pushfirst!(bytes, UInt8(m & 0xff))
        m >>= 8
    end
    return B.make_zint_from_bytes(GiacCxxBindings.CxxWrap.StdVector(bytes), sign(v))
end

_gen_from_node(n::SymNode) = n.name in _GIAC_NAMED_VALUES ? GiacCxxBindings.giac_eval(n.name) :
                             _identifier_gen(n.name)

# One GIAC identifier per name, for the session; each use gets a copy, sharing its name. A
# context's table of assumptions and values keys an entry by the name of the identifier it was
# made with, so that identifier must never be freed.
const _IDENTIFIERS = Dict{String, Any}()
_identifier_gen(name::AbstractString) =
    with_giac_lock(() -> copy(get!(() -> GiacCxxBindings.make_identifier(name), _IDENTIFIERS, name)))
_gen_from_node(n::GenNode) = _get_gen(n.expr.ptr)

_gen_from_node(n::AddNode) = _gen_sum([_gen_from_node(_scale(m, c)) for (m, c) in n.terms])

# GIAC copies a sum's terms into each new sum, so the terms are added in a balanced tree rather
# than one at a time; the sums are flattened, giving the same terms in the same order
function _gen_sum(ts, lo = 1, hi = length(ts))
    lo >= hi && return lo == hi ? ts[lo] : nothing
    mid = (lo + hi) >>> 1
    return _gen_sum(ts, lo, mid) + _gen_sum(ts, mid + 1, hi)
end

# a product as numerator / denominator, which GIAC keeps as a product with `inv`
function _gen_from_node(n::MulNode)
    B = GiacCxxBindings
    num = den = nothing
    for (b, e) in n.factors
        below = e isa NumNode && _negative(e.v)
        x = below ? NumNode(-e.v) : e
        f = isequal(x, _ONE) ? _gen_from_node(b) : B.giac_pow(_gen_from_node(b), _gen_from_node(x))
        below ? (den = den === nothing ? f : den * f) : (num = num === nothing ? f : num * f)
    end
    c = n.coef
    if !_isone_exact(c)
        num = isone(-c) && num !== nothing ? -num :
              num === nothing ? _gen_from_node(NumNode(c)) : _gen_from_node(NumNode(c)) * num
    end
    den === nothing && return num
    return (num === nothing ? B.Gen(Int32(1)) : num) / den
end

function _gen_from_node(n::CallNode)
    n.gen === nothing || return n.gen
    B = GiacCxxBindings
    v = B.CxxWrap.StdVector{B.Gen}()
    if n.f == "of" && length(n.args) > 2
        # GIAC applies a function to one argument, a sequence when there are several: f(a, b) is
        # of(f, seq(a, b)), read with the sequence flattened
        seq = B.CxxWrap.StdVector{B.Gen}()
        foreach(a -> push!(seq, _gen_from_node(a)), n.args[2:end])
        push!(v, _gen_from_node(n.args[1]))
        push!(v, B.make_vect(seq, Int32(B.VECTSUBTYPE_SEQ)))
    else
        foreach(a -> push!(v, _gen_from_node(a)), n.args)
    end
    return B.make_symbolic_unevaluated(n.f == "=" ? "equal" : n.f, v)
end

# ---------------------------------------------------------------------------------------------
# Expansion: products distributed over sums, positive integer powers of sums multiplied out
# ---------------------------------------------------------------------------------------------

_expand(n::Node) = n
function _expand(n::AddNode)
    expanded = [_expand(_scale(m, c)) for (m, c) in n.terms]
    return any(x -> x isa AddNode, expanded) || any(i -> !isequal(expanded[i], _scale(n.terms[i]...)), eachindex(expanded)) ?
           _add(expanded...) : n
end
function _expand(n::MulNode)
    any(((b, e),) -> b isa AddNode || b isa MulNode || b isa CallNode, n.factors) || return n
    acc = NumNode(n.coef)
    for (b, e) in n.factors
        f = _expand(b)
        if f isa AddNode && e isa NumNode && e.v isa Integer && e.v > 0
            for _ in 1:e.v
                acc = _times(acc, f)              # merged after each step, so the sums stay small
            end
        else
            acc = _times(acc, isequal(e, _ONE) ? f : _pow(f, e))
        end
    end
    return acc
end

# a product of expanded factors, distributed
_terms(n::Node) = n isa AddNode ? (_scale(m, c) for (m, c) in n.terms) : (n,)
_times(a::Node, b::Node) = _add((_mul(t, u) for t in _terms(a) for u in _terms(b))...)

# whether a node has a sum in a denominator (a rational function, not a polynomial)
_has_denominator(n::Node) = false
_has_denominator(n::AddNode) = any(t -> _has_denominator(first(t)), n.terms)
_has_denominator(n::MulNode) = any(((b, e),) -> (b isa AddNode && !(e isa NumNode && e.v isa Integer && e.v > 0)) || _has_denominator(b), n.factors)

# The names of the symbols in a tree; named constants (pi, i) are not symbols
function _symbol_names(n::Node)
    out = Set{String}()
    _symbol_names!(out, n)
    return out
end
_symbol_names!(out, n::NumNode) = out
_symbol_names!(out, n::SymNode) = (n.name in _GIAC_NAMED_VALUES || push!(out, n.name); out)
_symbol_names!(out, n::AddNode) = (foreach(t -> _symbol_names!(out, first(t)), n.terms); out)
_symbol_names!(out, n::MulNode) = (foreach(((b, e),) -> (_symbol_names!(out, b); _symbol_names!(out, e)), n.factors); out)
# a Bessel or Airy function's name is not a symbol (special_functions.jl)
_symbol_names!(out, n::CallNode) =
    (foreach(a -> _symbol_names!(out, a), _special_name(n) === nothing ? n.args : @view(n.args[2:end])); out)
_symbol_names!(out, n::GenNode) = (foreach(s -> push!(out, string(s)), free_symbols(n.expr)); out)

# the names of an expression's free variables, read from its tree
_free_names(e::GiacExpr) = _symbol_names(_node(e))
