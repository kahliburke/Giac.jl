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
    return m isa MulNode ? MulNode(c, m.factors) : MulNode(c, Pair{Node, Node}[m => _ONE])
end

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
        else
            push!(kept, b => e)
        end
    end
    iszero(coef) && return NumNode(coef)
    isempty(kept) && return NumNode(coef)
    _isone_exact(coef) && length(kept) == 1 && isequal(last(kept[1]), _ONE) && return first(kept[1])
    return MulNode(coef, kept)
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
        return MulNode(1, Pair{Node, Node}[b => e])
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

# whether a tree has a floating-point number in it
_has_float(n::Node) = false
_has_float(n::NumNode) = n.v isa AbstractFloat || (n.v isa Complex && real(n.v) isa AbstractFloat)
_has_float(n::AddNode) = any(((m, c),) -> c isa AbstractFloat || _has_float(m), n.terms)
_has_float(n::MulNode) = n.coef isa AbstractFloat || any(((b, e),) -> _has_float(b) || _has_float(e), n.factors)
_has_float(n::CallNode) = any(_has_float, n.args)

# Whether a node is known not to be negative: a number ≥ 0, a name declared positive or
# nonnegative (or in an interval above 0), pi, and sums, products and powers of such
_known_nonnegative(n::Node) = false
_known_nonnegative(n::NumNode) = n.v isa Real && n.v >= 0
function _known_nonnegative(n::SymNode)
    n.name == "pi" && return true
    tag = lock(() -> get(_DECLARED, n.name, nothing), _DECLARED_LOCK)
    return tag === :positive || tag === :nonnegative || (tag isa Tuple && length(tag) == 2 && tag[1] isa Real && tag[1] >= 0)
end
_known_nonnegative(n::MulNode) = n.coef isa Real && n.coef >= 0 &&
                                 all(((b, e),) -> _known_nonnegative(b) && e isa NumNode && e.v isa Real, n.factors)
_known_nonnegative(n::AddNode) = all(((m, c),) -> c isa Real && c >= 0 && (m isa NumNode || _known_nonnegative(m)), n.terms)
_known_nonnegative(n::CallNode) = n.f == "exp" && length(n.args) == 1 && _known_real(n.args[1])
_known_real(n::Node) = _known_nonnegative(n) || (n isa NumNode && n.v isa Real) ||
                       (n isa SymNode && lock(() -> get(_DECLARED, n.name, nothing), _DECLARED_LOCK) in (:real, :integer, :negative, :nonpositive))

# b^e for numbers with a fractional exponent: computed for floats, exact for perfect powers
function _numeric_root(b, e)
    (b isa Real && e isa Real) || return nothing
    if b isa AbstractFloat || e isa AbstractFloat
        return b >= 0 ? NumNode(Float64(b)^Float64(e)) : nothing
    end
    (b isa Integer && b >= 0 && e isa Rational) || return nothing
    iszero(b) && return e > 0 ? _ZERO : nothing
    q = denominator(e)
    r = round(BigInt, big(b)^(1 / big(q)))
    for c in (r - 1, r, r + 1)
        c >= 0 && c^q == b && return NumNode(_npow(_canonical(c), numerator(e)))
    end
    # the largest qth power that divides b comes out: 8^(1/2) = 2·2^(1/2)
    if b <= 10^8
        for s in r:-1:2
            k, rem = divrem(b, big(s)^q)
            rem == 0 && return _mul(NumNode(_npow(_canonical(s), numerator(e))), MulNode(1, Pair{Node, Node}[NumNode(k) => NumNode(e)]))
        end
    end
    return nothing
end

# exp(c)^k as exp(c*k) for a real number c: e^2 is exp(2), e^x is exp(x)
function _exp_power(b::Node, e::Node)
    (b isa CallNode && b.f == "exp" && length(b.args) == 1 && b.args[1] isa NumNode && b.args[1].v isa Real && !isequal(e, _ONE)) || return nothing
    return CallNode("exp", Node[_mul(b.args[1], e)])
end

_neg(a::Node) = _mul(_MINUS_ONE, a)
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

# The tree of a GIAC object (with the GIAC lock held)
function _node_from_gen(gen)
    B = GiacCxxBindings
    t = T(B.type(gen))
    t == INT && return NumNode(Int(B.to_int64(gen)))
    t == ZINT && return NumNode(parse(BigInt, String(B.zint_to_string(gen))))
    t == DOUBLE && return NumNode(Float64(B.to_double(gen)))
    t == IDNT && return SymNode(String(B.idnt_name(gen)))
    if t == FRAC
        p, q = _node_from_gen(B.frac_num(gen)), _node_from_gen(B.frac_den(gen))
        (p isa NumNode && q isa NumNode) && return NumNode(p.v // q.v)
    elseif t == CPLX
        re, im = _node_from_gen(B.cplx_re(gen)), _node_from_gen(B.cplx_im(gen))
        (re isa NumNode && im isa NumNode && re.v isa Real && im.v isa Real) && return NumNode(Complex(re.v, im.v))
    elseif t == SYMB
        h = String(B.symb_sommet_name(gen))
        if h in ("+", "*", "-", "inv", "^", "/") || haskey(_INFIX, h) || occursin(_CALL_NAME, h)
            leaf = B.symb_feuille(gen)
            raw = Any[]
            if B.is_vector(leaf) && B.subtype(leaf) in (B.VECTSUBTYPE_SEQ, _SORTED_VECT)
                for i in 0:(B.vect_size(leaf) - 1)
                    _push_gen_args!(raw, B.vect_at(leaf, i))
                end
            else
                push!(raw, leaf)
            end
            args = Node[_node_from_gen(a) for a in raw]
            h == "+" && return _add(args...)
            h == "*" && return _mul(args...)
            h == "-" && return length(args) == 1 ? _neg(args[1]) : _sub(args[1], args[2])
            h == "inv" && length(args) == 1 && return _pow(args[1], _MINUS_ONE)
            h == "^" && length(args) == 2 && return _pow(args[1], args[2])
            h == "/" && length(args) == 2 && return _div(args[1], args[2])
            h in ("+", "*", "-", "inv", "^", "/") || return CallNode(h, args, gen)
        end
    end
    e = GiacExpr(_make_gen_ptr(gen))
    return GenNode(e, String(B.to_string(gen)))
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
                             GiacCxxBindings.make_identifier(n.name)
_gen_from_node(n::GenNode) = _get_gen(n.expr.ptr)

function _gen_from_node(n::AddNode)
    g = nothing
    for (m, c) in n.terms
        t = _gen_from_node(_scale(m, c))
        g = g === nothing ? t : g + t
    end
    return g
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
_symbol_names!(out, n::CallNode) = (foreach(a -> _symbol_names!(out, a), n.args); out)
_symbol_names!(out, n::GenNode) = (foreach(s -> push!(out, string(s)), free_symbols(n.expr)); out)
