# Patterns as queries: test an expression's shape, take it apart, find where a pattern occurs.

"""
    SymbolicPattern

A pattern made by [`@pattern`](@ref), for `match`, `occursin` and `findall`, and for
`rewrite(…; within = p)`.
"""
struct SymbolicPattern
    pattern::Pattern
    source::Any
end

"""
    @pattern lhs

A pattern, written as the left side of a rule: slots `~x`, segments `~~xs`, tests and defaults
as in [`@rule`](@ref). A pattern matches a whole expression; slots and segments take what is
left, so `~a * x^2 + ~~rest` matches any sum with an `x^2` term. A term of a sum written with a
factor that defaults to 0, `(~b = 0) * x`, may be absent.

```julia
p = @pattern ~a * x^2 + (~b = 0) * x + ~c
m = match(p, 3x^2 + 2x + 1)
m[:a], m[:b], m[:c]                  # 3, 2, 1
match(p, 3x^2 + 1)[:b]               # 0
occursin(@pattern(sin(~t)), e)       # does sin appear anywhere in e?
findall(@pattern(sin(~t)), e)        # every match, outermost first
```
"""
macro pattern(ex)
    code, slots = _lhs_code(ex)
    return esc(:($SymbolicPattern($_make_pattern($code, $slots), $(QuoteNode(ex)))))
end

Base.show(io::IO, p::SymbolicPattern) = print(io, "@pattern ", p.source)
Base.show(io::IO, ::MIME"text/latex", p::SymbolicPattern) = print(io, "\$\$", _side_latex(p.source), "\$\$")

"""
    PatternMatch

What a [`@pattern`](@ref) matched: `m[:a]` is the value of slot `~a` (an expression) or segment
`~~a` (a vector), `m.expr` the whole expression matched.
"""
struct PatternMatch
    expr::GiacExpr
    captures::Dict{Symbol, Any}
end
Base.getindex(m::PatternMatch, name::Symbol) = m.captures[name]
Base.haskey(m::PatternMatch, name::Symbol) = haskey(m.captures, name)
Base.keys(m::PatternMatch) = keys(m.captures)
function Base.show(io::IO, m::PatternMatch)
    print(io, "PatternMatch(", m.expr)
    for name in sort!(collect(keys(m.captures)))
        print(io, ", ", name, " = ", m.captures[name])
    end
    print(io, ")")
end

_bindings(p::SymbolicPattern, e::GiacExpr) = _match(p.pattern, e, Bindings(), b -> b)
_matches(p::SymbolicPattern, e::GiacExpr) = _bindings(p, e) !== nothing

"""
    match(p::SymbolicPattern, e) -> Union{PatternMatch, Nothing}

Match the whole of `e` against the pattern `p`.
"""
function Base.match(p::SymbolicPattern, e::GiacExpr)
    b = _bindings(p, e)
    return b === nothing ? nothing : PatternMatch(e, Dict{Symbol, Any}(b))
end
Base.match(p::SymbolicPattern, x::Number) = match(p, convert(GiacExpr, x))

"""
    occursin(p::SymbolicPattern, e)

Whether `p` matches `e` or any subexpression of it.
"""
Base.occursin(p::SymbolicPattern, e::GiacExpr) = any(s -> _matches(p, s), _subexpressions(e))

"""
    findall(p::SymbolicPattern, e) -> Vector{PatternMatch}

Every subexpression of `e` (including `e`) that `p` matches, outermost first.
"""
Base.findall(p::SymbolicPattern, e::GiacExpr) =
    PatternMatch[m for s in _subexpressions(e) for m in (match(p, s),) if m !== nothing]

# `e` and its subexpressions, outermost first
function _subexpressions(e::GiacExpr, out = GiacExpr[])
    push!(out, e)
    iscall(e) && foreach(a -> _subexpressions(a, out), arguments(e))
    return out
end

"""
    @match e begin
        pattern₁ => value₁
        pattern₂ => value₂ where condition
        _ => otherwise
    end

The value for the first pattern that matches the whole of `e` (and whose condition holds), with
the slots available as `~x` in it, as in a rule's right side. `_` matches anything; without it,
an expression no pattern matches is an error.

```julia
degree_two(e, x) = @match e begin
    ~a * x^2 + (~b = 0) * x + ~c => (~a, ~b, ~c) where is_free(~a, x) && is_free(~b, x) && is_free(~c, x)
    _ => nothing
end
```
"""
macro match(subject, block)
    lines = filter(l -> !(l isa LineNumberNode), block isa Expr && block.head === :block ? block.args : [block])
    branches, fallback = Any[], nothing
    for l in lines
        l isa Expr && l.head === :call && l.args[1] === :(=>) && length(l.args) == 3 ||
            throw(ArgumentError("a @match branch is written `pattern => value`, got `$l`"))
        lhs, rhs = l.args[2], l.args[3]
        if lhs === :_
            fallback = rhs
        else
            body, cond = _split_where(rhs)
            wrapped = cond === nothing ? :(Some($body)) : Expr(:where, :(Some($body)), cond)
            push!(branches, _rule_expr(Expr(:call, :(=>), lhs, wrapped)))
        end
    end
    s, r = gensym(:subject), gensym(:result)
    otherwise = fallback === nothing ? :(throw(ArgumentError("no pattern matches " * string($s)))) : fallback
    return esc(quote
        let $s = convert($GiacExpr, $subject), $r = $_first_branch($s, $Rule[$(branches...)])
            $r === nothing ? $otherwise : something($r)
        end
    end)
end

function _first_branch(e::GiacExpr, rules)
    for r in rules
        v = _match(r.lhs, e, Bindings(), r.rhs)
        v === nothing || return v
    end
    return nothing
end
