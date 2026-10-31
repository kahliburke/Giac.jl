# Patterns: the left side of a rule, read back from Giac's evaluation of it with each slot
# replaced by a placeholder variable.

abstract type Pattern end

"`~x`: any subexpression; `~x::pred` one that passes `pred`; `(~x = d)` is `d` where absent."
struct Slot <: Pattern
    name::Symbol
    predicate::Any          # a function of one GiacExpr, or `nothing`
    default::Any            # a GiacExpr, or `nothing`
end

"`~~xs`: any number of terms, factors or arguments; `~~xs::pred` those that pass `pred`."
struct Segment <: Pattern
    name::Symbol
    predicate::Any
end

"An expression that matches only itself; `number` is its value if it is a number."
struct Leaf <: Pattern
    key::String
    expr::GiacExpr
    number::Any
end
Leaf(e::GiacExpr) = Leaf(_key(e), e, _is_number(e) ? Giac.to_julia(e) : nothing)

struct Term <: Pattern
    head::String
    args::Vector{Pattern}
end

# Sums and products are matched as multisets; the value is the identity element.
const _AC = Dict("+" => 0, "*" => 1)

_key(e) = string(e)
_head(e::GiacExpr) = iscall(e) ? symb_funcname(e) : ""

"""
The placeholder variable standing for a slot while Giac evaluates a pattern: `~name` (`kind`
`:slot`), `~~name` (`:segment`), or `+(~~name)` / `*(~~name)` (`:sum` / `:product`).
"""
_slot_var(name::Symbol, kind::Symbol = :slot) = giac_identifier("rule_" * string(kind) * "_" * string(name))

"The pattern a placeholder stands for."
function _slot_pattern(name::Symbol, kind::Symbol, pred, default)
    kind === :slot && return Slot(name, pred, default === nothing ? nothing : convert(GiacExpr, default))
    kind === :segment && return Segment(name, pred)
    return Term(kind === :sum ? "+" : "*", Pattern[Segment(name, pred)])
end

# matching order within a sum or product: constraining terms, slots, segments with a test, the
# segment without one
_rank(p) = p isa Segment ? (p.predicate === nothing ? 3 : 2) : p isa Slot ? 1 : 0

"Read Giac's tree of an evaluated pattern back as a `Pattern`, turning placeholders into slots."
function _pattern(e::GiacExpr, slots::Dict{String, Pattern})
    key = _key(e)
    haskey(slots, key) && return slots[key]
    iscall(e) || return Leaf(e)
    head = symb_funcname(e)
    args = Pattern[_pattern(a, slots) for a in arguments(e)]
    if haskey(_AC, head)
        # `+(~~xs)` among the terms of a sum is the segment itself
        args = Pattern[a isa Term && a.head == head && length(a.args) == 1 && a.args[1] isa Segment ? a.args[1] : a
                       for a in args]
        args = sort(args; by = _rank)
        count(a -> a isa Segment && a.predicate === nothing, args) <= 1 ||
            throw(ArgumentError("a sum or product in a rule can have one segment `~~xs` without a test, not several"))
    end
    return Term(head, args)
end
_pattern(x::Number, slots) = _pattern(convert(GiacExpr, x), slots)

# ---------------------------------------------------------------------------------------------
# Slot syntax in written code
# ---------------------------------------------------------------------------------------------

# `~x` or `~x::pred` as `(name, pred)`, or `nothing`
function _slot(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 2 || return nothing
    s = ex.args[2]
    s isa Symbol && return (s, nothing)
    s isa Expr && s.head === :(::) && length(s.args) == 2 && s.args[1] isa Symbol && return (s.args[1], s.args[2])
    return nothing
end

# `~~x` or `~~x::pred` as `(name, pred)`, or `nothing`
_segment(ex) = ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 2 ? _slot(ex.args[2]) : nothing

# `(~n = 1)` parses as a method definition of `~` whose body is a block
_unblock(ex) = ex isa Expr && ex.head === :block ? last(filter(a -> !(a isa LineNumberNode), ex.args)) : ex

"""
Replace each slot in `ex` by `f(name, pred, kind, default)`: `~x`, `~x::pred` and `(~x = d)` are
of kind `:slot`, `~~x` and `~~x::pred` of kind `:segment`, and `+(~~x)` and `*(~~x)` of kinds
`:sum` and `:product`. `pred` and `default` are code, or `nothing`.
"""
function _replace_slots(f, ex)
    if ex isa Expr && ex.head === :call && ex.args[1] in (:+, :*) && length(ex.args) == 2
        seg = _segment(ex.args[2])
        seg === nothing || return f(seg[1], seg[2], ex.args[1] === :+ ? :sum : :product, nothing)
    end
    seg = _segment(ex)
    seg === nothing || return f(seg[1], seg[2], :segment, nothing)
    # `(~x = d)`, and `f(~x = d)` (where it parses as a keyword argument)
    if ex isa Expr && ex.head in (:(=), :kw) && length(ex.args) == 2
        s = _slot(ex.args[1])
        s === nothing || return f(s[1], s[2], :slot, _unblock(ex.args[2]))
    end
    s = _slot(ex)
    s === nothing || return f(s[1], s[2], :slot, nothing)
    # `^` binds tighter than unary `~`: read `~x^n` as `(~x)^n`
    if ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 2
        a = ex.args[2]
        a isa Expr && a.head === :call && a.args[1] === :^ && length(a.args) == 3 &&
            return _replace_slots(f, Expr(:call, :^, Expr(:call, :~, a.args[2]), a.args[3]))
    end
    ex isa Expr || return ex
    return Expr(ex.head, (_replace_slots(f, a) for a in ex.args)...)
end

"""
Code that evaluates the left side `lhs` of a rule or pattern with placeholders for its slots, and
the list of `(name, kind, predicate, default)` code for them.
"""
function _lhs_code(lhs)
    slots = Dict{Tuple{Symbol, Symbol}, Tuple{Any, Any}}()      # (name, kind) => (predicate, default)
    code = _replace_slots(lhs) do name, pred, kind, default
        any(((n, k),) -> n === name && (k === :slot) != (kind === :slot), keys(slots)) &&
            throw(ArgumentError("`$name` is used both as a slot `~$name` and a segment `~~$name`"))
        old = get(slots, (name, kind), (nothing, nothing))
        slots[(name, kind)] = (something(pred, old[1], Some(nothing)), something(default, old[2], Some(nothing)))
        :($_slot_var($(QuoteNode(name)), $(QuoteNode(kind))))
    end
    list = [:(($(QuoteNode(n)), $(QuoteNode(k)), $p, $d)) for ((n, k), (p, d)) in slots]
    return code, :(Any[$(list...)])
end

"The pattern for an evaluated left side, given the `(name, kind, predicate, default)` slot list."
function _make_pattern(lhs_value, slots::Vector)
    table = Dict{String, Pattern}(_key(_slot_var(n, k)) => _slot_pattern(n, k, p, d) for (n, k, p, d) in slots)
    return _pattern(lhs_value, table)
end
