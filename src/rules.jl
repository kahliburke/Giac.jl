"""
    Giac.Rules

Pattern rewrite rules on Giac expressions.

```julia
using Giac, Giac.Rules
@giac_var x y z
r = @rule ~u * ~v + ~u * ~w => ~u * (~v + ~w)
rewrite(x*y + x*z + 1, r)          # x*(y+z)+1
```

`~x` is a slot that matches any subexpression; a slot used twice must match equal subexpressions.
`~x::pred` matches only where `pred(subexpr)` is `true`. The right side is ordinary Julia code
evaluated with the slots' values; if it returns `nothing`, the rule does not apply to that match.

The left side is evaluated by Giac with each slot replaced by a placeholder variable, so a
pattern has the same normal form as the expressions it meets (`~a - ~b` is `a + -b`, `~a / ~b`
is `a * inv(b)`). Sums and products match in any order: each slot takes one term, the last slot
in a nested sum or product takes all the remaining terms, and a slot with nothing left to match
stands for `0` in a sum and `1` in a product. At the top of a rule, terms the pattern does not
use are kept: `sin(~x)^2 + cos(~x)^2 => 1` turns `sin(a)^2 + cos(a)^2 + b` into `b + 1`.
"""
module Rules

using ..Giac: Giac, GiacExpr, giac_identifier, iscall, arguments, operation, symb_funcname

export @rule, @rules, rewrite

# ---------------------------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------------------------

abstract type Pattern end

struct Slot <: Pattern
    name::Symbol
    predicate::Any          # a function of one GiacExpr, or `nothing`
end

struct Leaf <: Pattern
    key::String
end

struct Term <: Pattern
    head::String
    args::Vector{Pattern}
end

# Sums and products are matched as multisets; the value is the identity element.
const _AC = Dict("+" => 0, "*" => 1)

_key(e) = string(e)
_head(e::GiacExpr) = iscall(e) ? symb_funcname(e) : ""

"The placeholder variable standing for slot `~name` while Giac evaluates a pattern."
_slot_var(name::Symbol) = giac_identifier("rule_slot_" * string(name))

"Read Giac's tree of an evaluated pattern back as a `Pattern`, turning placeholders into slots."
function _pattern(e::GiacExpr, slots::Dict{String, Slot})
    key = _key(e)
    haskey(slots, key) && return slots[key]
    iscall(e) || return Leaf(key)
    head = symb_funcname(e)
    args = Pattern[_pattern(a, slots) for a in arguments(e)]
    # in a sum or product, match the more constraining arguments first and the slots last
    haskey(_AC, head) && (args = vcat(filter(a -> !(a isa Slot), args), filter(a -> a isa Slot, args)))
    return Term(head, args)
end
_pattern(x::Number, slots) = _pattern(convert(GiacExpr, x), slots)

# ---------------------------------------------------------------------------------------------
# Matching, by backtracking: each `_match` calls the continuation `k` with the extended bindings
# and returns its result, or returns `nothing` so the caller tries the next alternative.
# ---------------------------------------------------------------------------------------------

const Bindings = Base.ImmutableDict{Symbol, GiacExpr}

_match(p::Leaf, s, b, k) = p.key == _key(s) ? k(b) : nothing

function _match(p::Slot, s, b, k)
    bound = get(b, p.name, nothing)
    bound === nothing || return _key(bound) == _key(s) ? k(b) : nothing
    p.predicate === nothing || p.predicate(s) === true || return nothing
    return k(Bindings(b, p.name, s))
end

function _match(p::Term, s, b, k)
    haskey(_AC, p.head) &&
        return _match_ac(p.head, p.args, 1, _ac_args(p.head, s), b, (b′, rest) -> isempty(rest) ? k(b′) : nothing, true)
    _head(s) == p.head || return nothing
    sargs = arguments(s)
    length(sargs) == length(p.args) || return nothing
    return _match_seq(p.args, sargs, 1, b, k)
end

function _match_seq(ps, ss, i, b, k)
    i > length(ps) && return k(b)
    return _match(ps[i], ss[i], b, b′ -> _match_seq(ps, ss, i + 1, b′, k))
end

"The terms of `s` as arguments of `head`: a negated product is a product with a factor `-1`."
function _ac_args(head, s::GiacExpr)
    h = _head(s)
    h == head && return arguments(s)
    head == "*" && h == "-" && return vcat(convert(GiacExpr, -1), _ac_args("*", only(arguments(s))))
    return [s]
end

"Combine terms with `head` (a sum or a product), Giac normalising the result."
function _combine(head, args)
    isempty(args) && return convert(GiacExpr, _AC[head])
    return reduce(head == "+" ? (+) : (*), args)
end

# Match the pattern arguments `ps[i:end]` against the multiset `rest` of subject arguments; `k`
# gets the bindings and the subject arguments left over. With `absorb`, the last free slot takes
# all remaining arguments.
function _match_ac(head, ps, i, rest, b, k, absorb)
    i > length(ps) && return k(b, rest)
    p = ps[i]
    if absorb && p isa Slot && i == length(ps) && !haskey(b, p.name)
        return _match(p, _combine(head, rest), b, b′ -> k(b′, GiacExpr[]))
    end
    for j in eachindex(rest)
        r = _match(p, rest[j], b, b′ -> _match_ac(head, ps, i + 1, deleteat!(copy(rest), j), b′, k, absorb))
        r === nothing || return r
    end
    # a slot with no term left for it (or already bound to the identity) stands for the identity
    if p isa Slot && (isempty(rest) || haskey(b, p.name))
        return _match(p, convert(GiacExpr, _AC[head]), b, b′ -> _match_ac(head, ps, i + 1, rest, b′, k, absorb))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------------------------

"""
    Rule

A rewrite rule made by [`@rule`](@ref). Calling it on an expression, `r(e)`, rewrites `e` at the
top level and returns the result, or `nothing` if the rule does not match.
"""
struct Rule
    lhs::Pattern
    rhs::Function           # bindings -> replacement (or `nothing` to decline the match)
    source::Expr
end

function Rule(lhs_value, slots::Vector{Slot}, rhs::Function, source::Expr)
    table = Dict(_key(_slot_var(s.name)) => s for s in slots)
    return Rule(_pattern(lhs_value, table), rhs, source)
end

Base.show(io::IO, r::Rule) = print(io, "Rule(", r.source, ")")

function (r::Rule)(e::GiacExpr)
    p = r.lhs
    if p isa Term && haskey(_AC, p.head)
        # at the top of a rule, terms the pattern does not use are kept beside the replacement
        _head(e) == p.head || return nothing
        return _match_ac(p.head, p.args, 1, arguments(e), Bindings(), (b, rest) -> _replace(r, b, rest), false)
    end
    return _match(p, e, Bindings(), b -> _replace(r, b, GiacExpr[]))
end

function _replace(r::Rule, b, rest)
    v = r.rhs(b)
    v === nothing && return nothing
    v = convert(GiacExpr, v)
    return isempty(rest) ? v : _combine(r.lhs.head, vcat(v, rest))
end

"""
    @rule lhs => rhs

A rewrite rule. Slots `~x` in `lhs` match subexpressions, `~x::pred` only those satisfying
`pred`; `rhs` is Julia code using the slots' values, and declines the match by returning
`nothing`. Other names on either side are ordinary Julia variables.

```julia
@rule sin(~x)^2 + cos(~x)^2 => 1
@rule ~u * ~v + ~u * ~w => ~u * (~v + ~w)
@rule a11 + a22 => τ
@rule exp(~x) * exp(~y) => exp(~x + ~y)
```

See [`Giac.Rules`](@ref) for how sums and products match, and [`rewrite`](@ref) to apply rules
throughout an expression.
"""
macro rule(ex)
    return esc(_rule_expr(ex))
end

"""
    @rules begin
        lhs₁ => rhs₁
        lhs₂ => rhs₂
    end

A vector of rules, one per line, for [`rewrite`](@ref).
"""
macro rules(block)
    lines = filter(l -> !(l isa LineNumberNode), block isa Expr && block.head === :block ? block.args : [block])
    return esc(Expr(:vect, map(_rule_expr, lines)...))
end

function _rule_expr(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) && length(ex.args) == 3 ||
        throw(ArgumentError("a rule is written `lhs => rhs`, got `$ex`"))
    lhs, rhs = ex.args[2], ex.args[3]
    slots = Dict{Symbol, Any}()                         # name => predicate expression
    lhs_code = _replace_slots(lhs) do name, pred
        pred === nothing || (slots[name] = pred)
        haskey(slots, name) || (slots[name] = nothing)
        :($_slot_var($(QuoteNode(name))))
    end
    b = gensym(:bindings)
    rhs_code = _replace_slots(rhs) do name, _
        :($b[$(QuoteNode(name))])
    end
    slot_list = [:($Slot($(QuoteNode(n)), $p)) for (n, p) in slots]
    return :($Rule($lhs_code, $Slot[$(slot_list...)], $b -> $rhs_code, $(QuoteNode(ex))))
end

"Replace each slot `~x` or `~x::pred` in `ex` by `f(name, pred)`."
function _replace_slots(f, ex)
    if ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 2
        s = ex.args[2]
        s isa Symbol && return f(s, nothing)
        s isa Expr && s.head === :(::) && s.args[1] isa Symbol && return f(s.args[1], s.args[2])
        # `^` binds tighter than unary `~`: read `~x^n` as `(~x)^n`
        s isa Expr && s.head === :call && s.args[1] === :^ && length(s.args) == 3 &&
            return _replace_slots(f, Expr(:call, :^, Expr(:call, :~, s.args[2]), s.args[3]))
    end
    ex isa Expr || return ex
    return Expr(ex.head, (_replace_slots(f, a) for a in ex.args)...)
end

# ---------------------------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------------------------

"""
    rewrite(e, rules...; maxpasses = 100)

Apply rules throughout `e`, bottom-up, until nothing changes. Each argument is a `Rule` or a
vector of them (from [`@rules`](@ref)); at each subexpression the first rule that matches is
used. Applies entry by entry to arrays. Throws if the rules still change the expression after
`maxpasses` passes.
"""
function rewrite(e::GiacExpr, rules...; maxpasses::Integer = 100)
    rs = collect(Rule, Iterators.flatten(r isa Rule ? (r,) : r for r in rules))
    for _ in 1:maxpasses
        e′ = _pass(e, rs)
        _key(e′) == _key(e) && return e′
        e = e′
    end
    throw(ErrorException("the rules still change the expression after $maxpasses passes"))
end

rewrite(x::Number, rules...; kwargs...) = rewrite(convert(GiacExpr, x), rules...; kwargs...)
rewrite(A::AbstractArray, rules...; kwargs...) = map(e -> rewrite(e, rules...; kwargs...), A)

function _pass(e::GiacExpr, rules)
    if iscall(e)
        args = arguments(e)
        new = [_pass(a, rules) for a in args]
        any(i -> _key(new[i]) != _key(args[i]), eachindex(args)) && (e = _rebuild(e, new))
    end
    for r in rules
        v = r(e)
        v === nothing || return v
    end
    return e
end

# A call of an undefined function `f(a, b)` is Giac's `of(f, a, b)`; the name itself is callable.
_rebuild(e::GiacExpr, args) = symb_funcname(e) == "of" ? args[1](args[2:end]...) : operation(e)(args...)

end # module Rules
