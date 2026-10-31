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
`~x::pred` matches only where `pred(subexpr)` is `true`. `~~xs` is a segment: it matches any
number of terms of a sum or product, or of arguments of a call, and its value is a vector of
them; `+(~~xs)` and `*(~~xs)` match a whole sum or product. The right side is ordinary Julia code
evaluated with the slots' values, after an optional `where` condition; if it returns `nothing`,
the rule does not apply to that match.

The left side is evaluated by Giac with each slot replaced by a placeholder variable, so a
pattern has the same normal form as the expressions it meets (`~a - ~b` is `a + -b`, `~a / ~b`
is `a * inv(b)`). Sums and products match in any order: each slot takes one term, the last slot
in a nested sum or product takes all the remaining terms, and a slot with nothing left to match
stands for `0` in a sum and `1` in a product. At the top of a rule, terms the pattern does not
use are kept: `sin(~x)^2 + cos(~x)^2 => 1` turns `sin(a)^2 + cos(a)^2 + b` into `b + 1`.
"""
module Rules

using ..Giac: Giac, GiacExpr, giac_identifier, iscall, arguments, operation, symb_funcname

export @rule, @rules, rewrite, is_free

# ---------------------------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------------------------

abstract type Pattern end

struct Slot <: Pattern
    name::Symbol
    predicate::Any          # a function of one GiacExpr, or `nothing`
end

struct Segment <: Pattern
    name::Symbol
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

"""
The placeholder variable standing for a slot while Giac evaluates a pattern: `~name` (`kind`
`:slot`), `~~name` (`:segment`), or `+(~~name)` / `*(~~name)` (`:sum` / `:product`).
"""
_slot_var(name::Symbol, kind::Symbol = :slot) = giac_identifier("rule_" * string(kind) * "_" * string(name))

"The pattern a placeholder stands for."
function _slot_pattern(name::Symbol, kind::Symbol, pred)
    kind === :slot && return Slot(name, pred)
    kind === :segment && return Segment(name)
    return Term(kind === :sum ? "+" : "*", Pattern[Segment(name)])
end

_rank(p) = p isa Segment ? 2 : p isa Slot ? 1 : 0

"Read Giac's tree of an evaluated pattern back as a `Pattern`, turning placeholders into slots."
function _pattern(e::GiacExpr, slots::Dict{String, Pattern})
    key = _key(e)
    haskey(slots, key) && return slots[key]
    iscall(e) || return Leaf(key)
    head = symb_funcname(e)
    args = Pattern[_pattern(a, slots) for a in arguments(e)]
    if haskey(_AC, head)
        # match the more constraining arguments first, then slots, then the segment
        args = sort(args; by = _rank)
        count(a -> a isa Segment, args) <= 1 ||
            throw(ArgumentError("a sum or product in a rule can have one segment `~~xs`, not several"))
    end
    return Term(head, args)
end
_pattern(x::Number, slots) = _pattern(convert(GiacExpr, x), slots)

# ---------------------------------------------------------------------------------------------
# Matching, by backtracking: each `_match` calls the continuation `k` with the extended bindings
# and returns its result, or returns `nothing` so the caller tries the next alternative.
# ---------------------------------------------------------------------------------------------

const Bindings = Base.ImmutableDict{Symbol, Any}     # a slot's expression, a segment's vector

_match(p::Leaf, s, b, k) = p.key == _key(s) ? k(b) : nothing

function _match(p::Slot, s, b, k)
    bound = get(b, p.name, nothing)
    bound === nothing || return _key(bound) == _key(s) ? k(b) : nothing
    p.predicate === nothing || p.predicate(s) === true || return nothing
    return k(Bindings(b, p.name, s))
end

# a segment on its own matches a single expression
_match(p::Segment, s, b, k) = _bind_segment(p, GiacExpr[s], b, k)

function _bind_segment(p::Segment, terms, b, k)
    bound = get(b, p.name, nothing)
    bound === nothing || return sort(_key.(bound)) == sort(_key.(terms)) ? k(b) : nothing
    return k(Bindings(b, p.name, terms))
end

function _match(p::Term, s, b, k)
    if haskey(_AC, p.head)
        # `+(~~xs)` matches a sum, not a single term
        length(p.args) == 1 && p.args[1] isa Segment && _head(s) != p.head && return nothing
        return _match_ac(p.head, p.args, 1, _ac_args(p.head, s), b, (b′, rest) -> isempty(rest) ? k(b′) : nothing, true)
    end
    _head(s) == p.head || return nothing
    return _match_seq(p.args, arguments(s), 1, 1, b, k)
end

# Match the pattern arguments `ps[i:end]` against the subject arguments `ss[j:end]` in order; a
# segment takes a run of them, as many as it can.
function _match_seq(ps, ss, i, j, b, k)
    i > length(ps) && return j > length(ss) ? k(b) : nothing
    p = ps[i]
    if p isa Segment
        for n in (length(ss) - j + 1):-1:0
            r = _bind_segment(p, ss[j:(j + n - 1)], b, b′ -> _match_seq(ps, ss, i + 1, j + n, b′, k))
            r === nothing || return r
        end
        return nothing
    end
    j > length(ss) && return nothing
    return _match(p, ss[j], b, b′ -> _match_seq(ps, ss, i + 1, j + 1, b′, k))
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
    # the segment comes last and takes every term left
    p isa Segment && return _bind_segment(p, rest, b, b′ -> k(b′, GiacExpr[]))
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

# `slots` holds a `(name, kind, predicate)` for each placeholder in the evaluated left side
function Rule(lhs_value, slots::Vector, rhs::Function, source::Expr)
    table = Dict{String, Pattern}(_key(_slot_var(n, k)) => _slot_pattern(n, k, p) for (n, k, p) in slots)
    return Rule(_pattern(lhs_value, table), rhs, source)
end

# the right side and the `where` condition (or `nothing`) as written
_rule_sides(source::Expr) = _split_where(source.args[3])

# `where` binds tighter than most operators: `a + b where c && d` parses as
# `(a + (b where c)) && d`. Find it at the end of the right side and take the condition apart.
function _split_where(ex)
    ex isa Expr || return (ex, nothing)
    ex.head === :where && length(ex.args) == 2 && return (ex.args[1], ex.args[2])
    if ex.head in (:&&, :||)
        body, cond = _split_where(ex.args[1])
        cond === nothing || return (body, Expr(ex.head, cond, ex.args[2:end]...))
    elseif ex.head === :call && length(ex.args) > 1
        body, cond = _split_where(ex.args[end])
        cond === nothing || return (Expr(:call, ex.args[1:(end - 1)]..., body), cond)
    end
    return (ex, nothing)
end

Base.show(io::IO, r::Rule) = print(io, "Rule(", r.source, ")")

# typeset with slots as plain names: `sin(~t)^2 + cos(~t)^2 => 1` as sin²t + cos²t → 1, a
# segment `~~xs` as `xs`, and a `where` condition after the rule
function _rule_latex(r::Rule)
    rhs, cond = _rule_sides(r.source)
    tex = _side_latex(r.source.args[2]) * " \\;\\to\\; " * _side_latex(rhs)
    return cond === nothing ? tex : tex * " \\quad \\text{if } " * _condition_latex(cond)
end

_plain(ex) = _replace_slots((name, _, kind) -> kind === :slot || kind === :segment ? name :
                                               Expr(:call, kind === :sum ? :sum : :product, name), ex)
_side_latex(ex) = _is_math(_plain(ex)) ? Giac.written_latex(_plain(ex)) : _code_latex(_plain(ex))
_code_latex(ex) = "\\texttt{" * replace(_code_string(ex), r"([_{}&%#$])" => s"\\\1") * "}"

const _RELATIONS = Dict(:< => "<", :> => ">", :<= => "\\le", :>= => "\\ge", :(==) => "=", :!= => "\\ne")

# a condition: comparisons of math typeset, `&&` as "and", `is_free(a, x)` as "a free of x"
function _condition_latex(c)
    c isa Expr && c.head === :&& && return join(map(_condition_latex, c.args), " \\text{ and } ")
    if c isa Expr && c.head === :call && length(c.args) == 3
        op, a, b = c.args
        haskey(_RELATIONS, op) && _is_math(_plain(a)) && _is_math(_plain(b)) &&
            return _side_latex(a) * " " * _RELATIONS[op] * " " * _side_latex(b)
        op === :is_free && return _side_latex(a) * " \\text{ free of } " * _side_latex(b)
    end
    return _code_latex(_plain(c))
end

# Written code Giac can typeset: names, numbers, and calls of named functions on such terms.
# Anything else (a conditional, a comparison, a string) is Julia logic and shown as code.
_is_math(::Union{Symbol, Number}) = true
_is_math(ex::Expr) = ex.head === :call && ex.args[1] isa Symbol &&
                     ex.args[1] ∉ (:(==), :(!=), :(===), :isequal, :string, :<, :>, :(<=), :(>=)) &&
                     all(_is_math, ex.args[2:end])
_is_math(_) = false

# one line of code; a conditional prints as `c ? a : b` (Julia prints its `Expr` as an `if` block)
_code_string(ex) = ex isa Expr && ex.head === :if && length(ex.args) == 3 ?
                   join(_code_string.(ex.args), " ? ", " : ") : string(ex)

Base.show(io::IO, ::MIME"text/latex", r::Rule) = print(io, "\$\$", _rule_latex(r), "\$\$")
Base.show(io::IO, ::MIME"text/latex", rs::AbstractVector{Rule}) =
    print(io, "\$\$\\begin{aligned}", join(("&" * _rule_latex(r) for r in rs), " \\\\ "), "\\end{aligned}\$\$")

function (r::Rule)(e::GiacExpr)
    p = r.lhs
    p isa Term && haskey(_AC, p.head) || return _match(p, e, Bindings(), b -> _replace(r, b, GiacExpr[]))
    # at the top of a rule, terms the pattern does not use are kept beside the replacement; with
    # a segment to take the other terms, a single term is a sum of one
    args = _head(e) == p.head ? arguments(e) : any(a -> a isa Segment, p.args) ? GiacExpr[e] : return nothing
    v = _match_ac(p.head, p.args, 1, args, Bindings(), (b, rest) -> _replace(r, b, rest), false)
    v === nothing && p.head == "+" || return v
    # a sum matches up to sign: if the rule rewrites -e to v, e is -v
    negated = -e
    _head(negated) == "+" || return nothing
    v = _match_ac(p.head, p.args, 1, arguments(negated), Bindings(), (b, rest) -> _replace(r, b, rest), false)
    return v === nothing ? nothing : -v
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

A segment `~~xs` matches the terms of a sum or product not matched otherwise (any number, even
none), or a run of a call's arguments; its value is a vector. `+(~~xs)` matches a sum of two or
more terms and `*(~~xs)` a product; on the right side they are the sum and product again.

`where cond` after the right side makes the rule apply only where `cond`, Julia code using the
slots, is `true`. It is shown with the rule.

```julia
@rule sin(~x)^2 + cos(~x)^2 => 1
@rule ~u * ~v + ~u * ~w => ~u * (~v + ~w)
@rule a11 + a22 => τ
@rule exp(~x) * exp(~y) => exp(~x + ~y)
@rule ~u * +(~~ts) => sum(~u * t for t in ~~ts)                  # distribute over any sum
@rule ~x^~n => 0 where ~n > 3                                     # drop powers above the third
@rule ~a * ~x^2 + ~b * ~x + ~~rest => ~a * (~x + ~b / (2 * ~a))^2 - ~b^2 / (4 * ~a) + +(~~rest) where is_free(~a, ~x) && is_free(~b, ~x)
```

See [`Giac.Rules`](@ref) for how sums and products match, and [`rewrite`](@ref) to apply rules
throughout an expression.
"""
macro rule(ex)
    return esc(_rule_expr(ex))
end

"""
    is_free(e, x)

Whether `e` does not involve the variable `x` (or any of the variables in an array `x`). For
conditions in rules: `@rule ~a * ~x => … where is_free(~a, ~x)`.
"""
function is_free(e, x)
    names = Set(string(v) for v in (x isa AbstractArray ? vec(x) : (x,)) for v in _symbols(v))
    return !any(v -> string(v) in names, _symbols(e))
end
_symbols(e::GiacExpr) = Giac.free_symbols(e)
_symbols(e::AbstractArray) = (s for x in e for s in _symbols(x))
_symbols(::Number) = ()

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
    lhs = ex.args[2]
    rhs, cond = _rule_sides(ex)
    slots = Dict{Tuple{Symbol, Symbol}, Any}()          # (name, kind) => predicate expression
    lhs_code = _replace_slots(lhs) do name, pred, kind
        any(((n, k),) -> n === name && (k === :slot) != (kind === :slot), keys(slots)) &&
            throw(ArgumentError("`$name` is used both as a slot `~$name` and a segment `~~$name`"))
        (pred !== nothing || !haskey(slots, (name, kind))) && (slots[(name, kind)] = pred)
        :($_slot_var($(QuoteNode(name)), $(QuoteNode(kind))))
    end
    b = gensym(:bindings)
    value(name, _, kind) = kind === :sum ? :(sum($b[$(QuoteNode(name))]; init = 0)) :
                           kind === :product ? :(prod($b[$(QuoteNode(name))]; init = 1)) :
                           :($b[$(QuoteNode(name))])
    rhs_code = _replace_slots(value, rhs)
    cond === nothing || (rhs_code = :($_holds($(_replace_slots(value, cond))) ? $rhs_code : nothing))
    slot_list = [:(($(QuoteNode(n)), $(QuoteNode(k)), $p)) for ((n, k), p) in slots]
    return :($Rule($lhs_code, Any[$(slot_list...)], $b -> $rhs_code, $(QuoteNode(ex))))
end

# A condition holds if it is `true`, or a comparison Giac evaluates to `true` (`~n > 1` for a
# number `n`); one Giac cannot decide (`x > 0` for a variable `x`) does not hold.
_holds(c::Bool) = c
_holds(c::GiacExpr) = _key(c) == "true"
_holds(c) = throw(ArgumentError("a rule condition must give true or false, got $c"))

# the name of a segment `~~name`, or `nothing`
_segment_name(ex) = ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 2 &&
                    (s = ex.args[2]; s isa Expr && s.head === :call && s.args[1] === :~ && length(s.args) == 2 &&
                     s.args[2] isa Symbol) ? ex.args[2].args[2] : nothing

"""
Replace each slot in `ex` by `f(name, pred, kind)`: `~x` and `~x::pred` are of kind `:slot`, `~~x`
of kind `:segment`, and `+(~~x)` and `*(~~x)` of kinds `:sum` and `:product`.
"""
function _replace_slots(f, ex)
    if ex isa Expr && ex.head === :call && ex.args[1] in (:+, :*) && length(ex.args) == 2
        name = _segment_name(ex.args[2])
        name === nothing || return f(name, nothing, ex.args[1] === :+ ? :sum : :product)
    end
    name = _segment_name(ex)
    name === nothing || return f(name, nothing, :segment)
    if ex isa Expr && ex.head === :call && ex.args[1] === :~ && length(ex.args) == 2
        s = ex.args[2]
        s isa Symbol && return f(s, nothing, :slot)
        s isa Expr && s.head === :(::) && s.args[1] isa Symbol && return f(s.args[1], s.args[2], :slot)
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
    rewrite(e, rules...; order = :bottomup, once = false, trace = false, maxpasses = 100)
    rewrite(rules...; kwargs...)

Apply rules throughout `e` until nothing changes. Each argument is a `Rule` or a vector of them
(from [`@rules`](@ref)); at each subexpression the first rule that matches is used. Applies
entry by entry to arrays.

- `order = :bottomup` rewrites the arguments of an expression before the expression itself;
  `:topdown` rewrites an expression first and then the arguments of the result.
- `once = true` makes a single pass instead of repeating until nothing changes.
- `trace = true` returns the steps as a [`@derivation`](@ref)-style display instead of the result:
  one rule applied in one place per step, each step labelled with its rule. The result is the
  value of the last step (`last(d)`).

Throws if the rules still change the expression after `maxpasses` passes (or steps, tracing).
Without an expression, `rewrite(rules...)` is a function of one, to compose rule sets in order:
`e |> rewrite(expand_rules) |> rewrite(trig_rules)`.
"""
function rewrite(e::GiacExpr, rules...; order::Symbol = :bottomup, once::Bool = false,
                 trace::Bool = false, maxpasses::Integer = 100)
    order in (:bottomup, :topdown) || throw(ArgumentError("order is :bottomup or :topdown, got :$order"))
    rs = collect(Rule, Iterators.flatten(r isa Rule ? (r,) : r for r in rules))
    trace && return _trace(e, rs, order, once ? 1 : maxpasses)
    for _ in 1:(once ? 1 : maxpasses)
        e′ = order === :bottomup ? _pass(e, rs) : _pass_topdown(e, rs)
        (once || _key(e′) == _key(e)) && return e′
        e = e′
    end
    throw(ErrorException("the rules still change the expression after $maxpasses passes"))
end

rewrite(x::Number, rules...; kwargs...) = rewrite(convert(GiacExpr, x), rules...; kwargs...)
rewrite(A::AbstractArray{<:Union{GiacExpr, Number}}, rules...; kwargs...) = map(e -> rewrite(e, rules...; kwargs...), A)
rewrite(rules::Union{Rule, AbstractVector{Rule}}...; kwargs...) = e -> rewrite(e, rules...; kwargs...)

# The first rule that rewrites `e` to something else: `(result, rule)`, or `nothing`.
function _apply(e::GiacExpr, rules)
    for r in rules
        v = r(e)
        v === nothing || _key(v) == _key(e) || return (v, r)
    end
    return nothing
end

function _pass(e::GiacExpr, rules)
    e = _rebuild_args(a -> _pass(a, rules), e)
    a = _apply(e, rules)
    return a === nothing ? e : first(a)
end

function _pass_topdown(e::GiacExpr, rules)
    a = _apply(e, rules)
    a === nothing || (e = first(a))
    return _rebuild_args(x -> _pass_topdown(x, rules), e)
end

# `e` with `f` applied to each argument (`e` itself if none changes)
function _rebuild_args(f, e::GiacExpr)
    iscall(e) || return e
    args = arguments(e)
    new = map(f, args)
    return any(i -> _key(new[i]) != _key(args[i]), eachindex(args)) ? _rebuild(e, new) : e
end

# One rewrite at the first place a rule applies, in the given order: `(result, rule)` or `nothing`.
function _step(e::GiacExpr, rules, order)
    if order === :topdown
        a = _apply(e, rules)
        a === nothing || return a
    end
    if iscall(e)
        args = arguments(e)
        for i in eachindex(args)
            s = _step(args[i], rules, order)
            s === nothing && continue
            new = copy(args)
            new[i] = first(s)
            return (_rebuild(e, new), last(s))
        end
    end
    return order === :bottomup ? _apply(e, rules) : nothing
end

function _trace(e::GiacExpr, rules, order, maxsteps)
    rows = Tuple{String, String}[]
    lhs = Giac.giac_latex(e)
    for _ in 1:maxsteps
        s = _step(e, rules, order)
        s === nothing && break
        e, r = s
        push!(rows, (lhs, Giac.giac_latex(e) * " \\qquad \\text{by } " * _rule_latex(r)))
        lhs = "{}"
    end
    maxsteps > 1 && _step(e, rules, order) !== nothing &&
        throw(ErrorException("the rules still change the expression after $maxsteps steps"))
    isempty(rows) && push!(rows, (lhs, Giac.giac_latex(e) * " \\qquad \\text{(no rule applies)}"))
    return RewriteTrace(Giac.Derivation(rows), e)
end

"""
The steps of a traced [`rewrite`](@ref): displays as aligned equations, one per rule applied.
`last(t)` is the result.
"""
struct RewriteTrace
    steps::Giac.Derivation
    result::GiacExpr
end
Base.last(t::RewriteTrace) = t.result
Base.show(io::IO, m::MIME"text/latex", t::RewriteTrace) = show(io, m, t.steps)
Base.show(io::IO, m::MIME"text/markdown", t::RewriteTrace) = show(io, m, t.steps)
Base.show(io::IO, t::RewriteTrace) = print(io, "RewriteTrace with $(length(t.steps.rows)) step(s) to ", t.result)

# A call of an undefined function `f(a, b)` is Giac's `of(f, a, b)`; the name itself is callable.
_rebuild(e::GiacExpr, args) = symb_funcname(e) == "of" ? args[1](args[2:end]...) : operation(e)(args...)

end # module Rules
