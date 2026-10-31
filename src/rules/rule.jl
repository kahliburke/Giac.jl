# Rules: a pattern, and Julia code computing the replacement from what the slots matched.

"""
    Rule

A rewrite rule made by [`@rule`](@ref). Calling it on an expression, `r(e)`, rewrites `e` at the
top level and returns the result, or `nothing` if the rule does not match.
"""
struct Rule
    lhs::Pattern
    rhs::Function           # bindings -> replacement (or `nothing` to decline the match)
    cond::Any               # bindings -> Bool, from `where`; or `nothing`
    source::Expr
    coefficient::Any        # the numeric coefficient written in the left side, if a product
    heads::Any              # the heads of the expressions it can match at the top, or `nothing` for any
end

Rule(lhs::Pattern, rhs::Function, cond, source::Expr) =
    Rule(lhs, rhs, cond, source, _coefficient(lhs), _top_heads(lhs))
Rule(lhs_value, slots::Vector, rhs::Function, cond, source::Expr) =
    Rule(_make_pattern(lhs_value, slots), rhs, cond, source)

# The heads an expression must have for the pattern to match it at the top, or `nothing` if it can
# match more: a slot or segment matches anything, a power also `inv`, a sum or product with a
# segment or a defaulted slot also a single term, and a product with a numeric coefficient
# anything it can scale.
function _top_heads(p::Pattern)
    p isa Term || return p isa Leaf ? ("",) : nothing
    p.head == "^" && return ("^", "inv")
    if haskey(_AC, p.head)
        any(a -> a isa Segment || a isa Slot && a.default !== nothing, p.args) && return nothing
        p.head == "*" && _coefficient(p) != 1 && return nothing
        return (p.head, "-")          # a negated sum or product matches up to sign
    end
    return (p.head,)
end

# the replacement for bindings `b`, or `nothing` if the condition does not hold
_rhs(r::Rule, b) = r.cond === nothing || r.cond(b) ? r.rhs(b) : nothing

function (r::Rule)(e::GiacExpr)
    r.heads === nothing || _head(e) in r.heads || return nothing
    p = r.lhs
    p isa Term && haskey(_AC, p.head) || return _match(p, e, Bindings(), b -> _replace(r, b, GiacExpr[]))
    v = _apply_ac(r, p, e)
    v === nothing && p.head == "*" || return v
    # a product matches up to its numeric coefficient: `sin(t)*cos(t)` is half of `2*sin(t)*cos(t)`
    pc, (c, fs) = r.coefficient, _split_term(e)
    pc == c && return nothing
    v = _apply_ac(r, p, _scaled(pc, fs))
    return v === nothing ? nothing : _scaled(_div(c, pc), [v])
end

function _apply_ac(r::Rule, p::Term, e::GiacExpr)
    # At the top of a rule, terms the pattern does not use are kept beside the replacement. A
    # single term is a sum (or product) of one when a segment or a slot with a default can stand
    # for the rest.
    args = _head(e) == p.head ? arguments(e) :
           any(a -> a isa Segment || a isa Slot && a.default !== nothing, p.args) ? GiacExpr[e] : return nothing
    v = _match_ac(p.head, p.args, 1, args, Bindings(), (b, rest) -> _replace(r, b, rest), false)
    v === nothing && p.head == "+" || return v
    # a sum matches up to sign: if the rule rewrites -e to v, e is -v
    negated = -e
    if _head(negated) == "+"
        v = _match_ac(p.head, p.args, 1, arguments(negated), Bindings(), (b, rest) -> _replace(r, b, rest), false)
        v === nothing || return -v
    end
    # and up to a common factor
    return _match_scaled(p.args, args, (b, k, others) -> begin
        v = _rhs(r, b)
        v === nothing ? nothing : _combine("+", vcat(k * convert(GiacExpr, v), others))
    end)
end

function _replace(r::Rule, b, rest)
    v = _rhs(r, b)
    v === nothing && return nothing
    v = convert(GiacExpr, v)
    return isempty(rest) ? v : _combine(r.lhs.head, vcat(v, rest))
end

"""
    @rule lhs => rhs
    @rule lhs => rhs where condition

A rewrite rule. Slots `~x` in `lhs` match subexpressions, `~x::pred` only those satisfying
`pred`; `rhs` is Julia code using the slots' values, and declines the match by returning
`nothing`. Other names on either side are ordinary Julia variables.

A segment `~~xs` matches the terms of a sum or product not matched otherwise (any number, even
none), or a run of a call's arguments; its value is a vector. `~~xs::pred` takes the terms that
pass `pred`, so segments with tests split a sum: `+(~~lin::of_degree(1, x)) + ~~rest`. `+(~~xs)`
matches a sum of two or more terms and `*(~~xs)` a product; on the right side they are the sum
and product again.

`(~x = d)` is a slot that stands for `d` where the subject has nothing for it: `~x^(~n = 1)`
matches `y` with `n = 1`, and `(~a = 1) * sin(~t)` matches `sin(y)`.

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
    @rules begin
        lhs₁ => rhs₁
        lhs₂ => rhs₂
    end

A vector of rules, one per line, for [`rewrite`](@ref).
"""
macro rules(block)
    lines = filter(l -> !(l isa LineNumberNode), block isa Expr && block.head === :block ? block.args : [block])
    return esc(:($Rule[$(map(_rule_expr, lines)...)]))
end

function _rule_expr(ex)
    ex isa Expr && ex.head === :call && ex.args[1] === :(=>) && length(ex.args) == 3 ||
        throw(ArgumentError("a rule is written `lhs => rhs`, got `$ex`"))
    lhs_code, slot_list = _lhs_code(ex.args[2])
    rhs, cond = _rule_sides(ex)
    b = gensym(:bindings)
    cond_code = cond === nothing ? nothing : :($b -> $_holds($(_slot_values(cond, b))))
    return :($Rule($lhs_code, $slot_list, $b -> $(_slot_values(rhs, b)), $cond_code, $(QuoteNode(ex))))
end

# code with each slot replaced by its value in the bindings `b`
function _slot_values(ex, b)
    value(name, _, kind, _) = kind === :sum ? :(sum($b[$(QuoteNode(name))]; init = 0)) :
                              kind === :product ? :(prod($b[$(QuoteNode(name))]; init = 1)) :
                              :($b[$(QuoteNode(name))])
    return _replace_slots(value, ex)
end

# A condition holds if it is `true`, or a comparison Giac evaluates to `true` (`~n > 1` for a
# number `n`); one Giac cannot decide (`x > 0` for a variable `x`) does not hold.
_holds(c::Bool) = c
_holds(c::GiacExpr) = _key(c) == "true"
_holds(c) = throw(ArgumentError("a rule condition must give true or false, got $c"))

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

# ---------------------------------------------------------------------------------------------
# Typesetting
# ---------------------------------------------------------------------------------------------

Base.show(io::IO, r::Rule) = print(io, "Rule(", r.source, ")")

# typeset with slots as plain names: `sin(~t)^2 + cos(~t)^2 => 1` as sin²t + cos²t → 1, a
# segment `~~xs` as `xs`, and a `where` condition after the rule
function _rule_latex(r::Rule)
    rhs, cond = _rule_sides(r.source)
    tex = _side_latex(r.source.args[2]) * " \\;\\to\\; " * _side_latex(rhs)
    return cond === nothing ? tex : tex * " \\quad \\text{if } " * _condition_latex(cond)
end

_plain(ex) = _replace_slots((name, _, kind, _) -> kind === :slot || kind === :segment ? name :
                                                  Expr(:call, kind === :sum ? :sum : :product, name), ex)

function _side_latex(ex)
    p = _plain(ex)
    _is_math(p) && return Giac.written_latex(p)
    g = _generator_latex(p)
    return g === nothing ? _code_latex(p) : g
end

# `sum(u * t for t in ts)` as a sum over `t ∈ ts` (and `prod` as a product), or `nothing`
function _generator_latex(ex)
    ex isa Expr && ex.head === :call && ex.args[1] in (:sum, :prod) && length(ex.args) == 2 || return nothing
    g = ex.args[2]
    g isa Expr && g.head === :generator && length(g.args) == 2 || return nothing
    body, it = g.args
    it isa Expr && it.head === :(=) && it.args[1] isa Symbol && _is_math(body) && _is_math(it.args[2]) || return nothing
    op = ex.args[1] === :sum ? "\\sum" : "\\prod"
    return op * "_{" * Giac.written_latex(it.args[1]) * " \\in " * Giac.written_latex(it.args[2]) * "} " *
           Giac.written_latex(body)
end
_code_latex(ex) = "\\texttt{" * replace(_code_string(ex), r"([_{}&%#$])" => s"\\\1") * "}"

const _RELATIONS = Dict(:< => "<", :> => ">", :<= => "\\le", :>= => "\\ge", :(==) => "=", :!= => "\\ne")

# The tests in `predicates.jl`, typeset as phrases: `is_positive(a)` as "a > 0".
const _PREDICATE_LATEX = Dict(
    :is_positive => a -> a * " > 0", :is_negative => a -> a * " < 0",
    :is_nonnegative => a -> a * " \\ge 0", :is_real => a -> a * " \\in \\mathbb{R}",
    :is_integer => a -> a * " \\in \\mathbb{Z}", :is_integer_valued => a -> a * " \\in \\mathbb{Z}")

# a condition: comparisons of math typeset, `&&` as "and", `is_free(a, x)` as "a free of x"
function _condition_latex(c)
    c isa Expr && c.head === :&& && return join(map(_condition_latex, c.args), " \\text{ and } ")
    c isa Expr && c.head === :|| && return join(map(_condition_latex, c.args), " \\text{ or } ")
    if c isa Expr && c.head === :call && length(c.args) == 3
        op, a, b = c.args
        haskey(_RELATIONS, op) && _is_math(_plain(a)) && _is_math(_plain(b)) &&
            return _side_latex(a) * " " * _RELATIONS[op] * " " * _side_latex(b)
        op === :is_free && return _side_latex(a) * " \\text{ free of } " * _side_latex(b)
    end
    if c isa Expr && c.head === :call && length(c.args) == 4 && c.args[1] === :in_interval
        bound(v) = v isa GiacExpr ? Giac.giac_latex(v) : _side_latex(v)
        return bound(c.args[3]) * " \\le " * _side_latex(c.args[2]) * " \\le " * bound(c.args[4])
    end
    if c isa Expr && c.head === :call && length(c.args) == 2 && haskey(_PREDICATE_LATEX, c.args[1]) && _is_math(_plain(c.args[2]))
        return _PREDICATE_LATEX[c.args[1]](_side_latex(c.args[2]))
    end
    return _code_latex(_plain(c))
end

# Written code Giac can typeset: names, numbers, and calls of named functions on such terms.
# Anything else (a conditional, a comparison, a string) is Julia logic and shown as code.
_is_math(::Union{Symbol, Number}) = true
_is_math(ex::Expr) = ex.head === :call && ex.args[1] isa Symbol &&
                     ex.args[1] ∉ (:(==), :(!=), :(===), :isequal, :string, :<, :>, :(<=), :(>=), :length) &&
                     all(_is_math, ex.args[2:end])
_is_math(_) = false

# one line of code; a conditional prints as `c ? a : b` (Julia prints its `Expr` as an `if` block)
_code_string(ex) = ex isa Expr && ex.head === :if && length(ex.args) == 3 ?
                   join(_code_string.(ex.args), " ? ", " : ") : string(ex)

Base.show(io::IO, ::MIME"text/latex", r::Rule) = print(io, "\$\$", _rule_latex(r), "\$\$")
Base.show(io::IO, ::MIME"text/latex", rs::AbstractVector{Rule}) =
    print(io, "\$\$\\begin{aligned}", join(("&" * _rule_latex(r) for r in rs), " \\\\ "), "\\end{aligned}\$\$")
