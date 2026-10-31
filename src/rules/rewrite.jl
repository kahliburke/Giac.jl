# Applying rules throughout an expression.

# what a rewrite applies, in which order, and where (`within`: a pattern, or `nothing` for
# everywhere)
struct _Ctx
    rules::Vector{Rule}
    order::Symbol
    within::Any
end

_rule_list(rules) = collect(Rule, Iterators.flatten(r isa Rule ? (r,) : r for r in rules))

"""
    rewrite(e, rules...; order = :bottomup, once = false, trace = false, within = nothing,
            at = nothing, maxpasses = 100)
    rewrite(rules...; kwargs...)

Apply rules throughout `e` until nothing changes. Each argument is a `Rule` or a vector of them
(from [`@rules`](@ref)); at each subexpression the first rule that matches is used. Applies
entry by entry to arrays.

- `order = :bottomup` rewrites the arguments of an expression before the expression itself;
  `:topdown` rewrites an expression first and then the arguments of the result.
- `once = true` makes a single pass instead of repeating until nothing changes.
- `within = @pattern(…)` rewrites only inside the subexpressions that pattern matches.
- `at = n` makes one rewrite, at the `n`th place a rule applies (in the order above).
- `trace = true` returns the steps instead of the result, displayed as aligned equations: one
  rule in one place per step, each labelled with its rule, repeats of a rule shown as one step.
  `trace = :steps` shows every step. The result is `last(t)`.

Rules that undo each other are reported by name. Without an expression, `rewrite(rules...)` is
a function of one, to compose rule sets in order: `e |> rewrite(expand_rules) |> rewrite(trig_rules)`.
"""
function rewrite(e::GiacExpr, rules...; order::Symbol = :bottomup, once::Bool = false,
                 trace::Union{Bool, Symbol} = false, within = nothing, at::Union{Nothing, Integer} = nothing,
                 maxpasses::Integer = 100)
    order in (:bottomup, :topdown) || throw(ArgumentError("order is :bottomup or :topdown, got :$order"))
    trace in (false, true, :steps) || throw(ArgumentError("trace is true, false or :steps, got $trace"))
    ctx = _Ctx(_rule_list(rules), order, within)
    if at !== nothing
        s = _step(e, ctx, false, Ref(Int(at)))
        trace === false && return s === nothing ? e : first(s)
        return _trace_display(e, s === nothing ? [] : [s]; collapse = false)
    end
    trace === false || return _trace(e, ctx, once ? 1 : maxpasses, trace === true)
    seen = Dict(_key(e) => 0)
    fired = Vector{Rule}[]
    for pass in 1:(once ? 1 : maxpasses)
        push!(fired, Rule[])
        e′ = _pass(e, ctx, false, fired[end])
        (once || _key(e′) == _key(e)) && return e′
        k = _key(e′)
        haskey(seen, k) && _cycle_error(reduce(vcat, fired[(seen[k] + 1):pass]), e′)
        seen[k] = pass
        e = e′
    end
    throw(ErrorException("the rules still change the expression after $maxpasses passes"))
end

rewrite(x::Number, rules...; kwargs...) = rewrite(convert(GiacExpr, x), rules...; kwargs...)
rewrite(A::AbstractArray{<:Union{GiacExpr, Number}}, rules...; kwargs...) = map(e -> rewrite(e, rules...; kwargs...), A)
rewrite(rules::Union{Rule, AbstractVector{Rule}}...; kwargs...) = e -> rewrite(e, rules...; kwargs...)

function _cycle_error(rules, e)
    names = unique(r -> objectid(r), rules)
    throw(ErrorException("these rules undo each other, returning to $e:\n" *
                         join(("  " * string(r.source) for r in names), "\n")))
end

# Whether rules apply at `e`: everywhere, or inside a match of `within`.
_inside(ctx::_Ctx, e, inside) = inside || ctx.within === nothing || _matches(ctx.within, e)

# The first rule that rewrites `e` to something else: `(result, rule)`, or `nothing`.
function _apply(e::GiacExpr, ctx::_Ctx, inside)
    inside || return nothing
    for r in ctx.rules
        v = r(e)
        v === nothing || _key(v) == _key(e) || return (v, r)
    end
    return nothing
end

# one pass over `e`, recording the rules used in `fired`
function _pass(e::GiacExpr, ctx::_Ctx, inside, fired)
    inside = _inside(ctx, e, inside)
    if ctx.order === :topdown
        a = _apply(e, ctx, inside)
        a === nothing || (e = first(a); push!(fired, last(a)))
        return _rebuild_args(x -> _pass(x, ctx, inside, fired), e)
    end
    e = _rebuild_args(x -> _pass(x, ctx, inside, fired), e)
    a = _apply(e, ctx, inside)
    a === nothing && return e
    push!(fired, last(a))
    return first(a)
end

# `e` with `f` applied to each argument (`e` itself if none changes)
function _rebuild_args(f, e::GiacExpr)
    iscall(e) || return e
    args = arguments(e)
    new = map(f, args)
    return any(i -> _key(new[i]) != _key(args[i]), eachindex(args)) ? _rebuild(e, new) : e
end

# A call of an undefined function `f(a, b)` is Giac's `of(f, a, b)`; the name itself is callable.
_rebuild(e::GiacExpr, args) = symb_funcname(e) == "of" ? args[1](args[2:end]...) : operation(e)(args...)

# One rewrite at the `count[]`th place a rule applies, in the given order: `(result, rule)`, or
# `nothing` if there are fewer places.
function _step(e::GiacExpr, ctx::_Ctx, inside, count::Ref{Int})
    inside = _inside(ctx, e, inside)
    if ctx.order === :topdown
        a = _apply(e, ctx, inside)
        a === nothing || (count[] -= 1) == 0 && return a
    end
    if iscall(e)
        args = arguments(e)
        for i in eachindex(args)
            s = _step(args[i], ctx, inside, count)
            s === nothing && continue
            new = copy(args)
            new[i] = first(s)
            return (_rebuild(e, new), last(s))
        end
    end
    if ctx.order === :bottomup
        a = _apply(e, ctx, inside)
        a === nothing || (count[] -= 1) == 0 && return a
    end
    return nothing
end

function _trace(e0::GiacExpr, ctx::_Ctx, maxsteps, collapse)
    steps = Tuple{GiacExpr, Any}[]
    seen = Dict(_key(e0) => 0)
    e = e0
    for i in 1:maxsteps
        s = _step(e, ctx, false, Ref(1))
        s === nothing && break
        push!(steps, s)
        e = first(s)
        k = _key(e)
        haskey(seen, k) && _cycle_error([last(t) for t in steps[(seen[k] + 1):i]], e)
        seen[k] = i
    end
    maxsteps > 1 && _step(e, ctx, false, Ref(1)) !== nothing &&
        throw(ErrorException("the rules still change the expression after $maxsteps steps"))
    return _trace_display(e0, steps; collapse)
end

"""
The steps of a traced [`rewrite`](@ref) or of [`simplest`](@ref): displays as aligned
equations, one per step, each labelled with what made it. `last(t)` is the result.
"""
struct RewriteTrace
    steps::Giac.Derivation
    result::GiacExpr
end
Base.last(t::RewriteTrace) = t.result
Base.show(io::IO, m::MIME"text/latex", t::RewriteTrace) = show(io, m, t.steps)
Base.show(io::IO, m::MIME"text/markdown", t::RewriteTrace) = show(io, m, t.steps)
Base.show(io::IO, t::RewriteTrace) = print(io, "RewriteTrace with $(length(t.steps.rows)) step(s) to ", t.result)

# `steps` are `(expression after, what made it)`; with `collapse`, a run of steps by the same
# rule shows as one, with its count
function _trace_display(e0, steps; collapse::Bool)
    runs = Tuple{GiacExpr, Any, Int}[]
    for (e, label) in steps
        if collapse && !isempty(runs) && runs[end][2] === label
            runs[end] = (e, label, runs[end][3] + 1)
        else
            push!(runs, (e, label, 1))
        end
    end
    rows = Tuple{String, String}[]
    lhs = Giac.giac_latex(e0)
    for (e, label, n) in runs
        times = n > 1 ? " \\;(\\times $n)" : ""
        push!(rows, (lhs, Giac.giac_latex(e) * " \\qquad \\text{by } " * _label_latex(label) * times))
        lhs = "{}"
    end
    isempty(rows) && push!(rows, (lhs, Giac.giac_latex(e0) * " \\qquad \\text{(no rule applies)}"))
    return RewriteTrace(Giac.Derivation(rows), isempty(steps) ? e0 : first(steps[end]))
end

# In a `@derivation`, a trace continues the step it is the value of: `lhs = start`, then one row
# per step.
function Giac._derivation_rows(left, written, t::RewriteTrace, simplifier)
    steps = t.steps.rows
    rows = [(left, join(vcat(written, first(steps[1])), " = "))]
    append!(rows, (("{}", last(r)) for r in steps))
    return rows
end

_label_latex(r::Rule) = _rule_latex(r)
_label_latex(f) = "\\text{" * string(nameof(f)) * "}"
