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
- `trace = true` returns the path instead of the result: a [`RewriteTrace`](@ref), the vector of
  expressions along the way (`last(t)` is the result), displayed as aligned equations, one rule
  in one place per step, each labelled with its rule, repeats of a rule shown as one step.
  `trace = :steps` shows every step.

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
    seen = Dict{GiacExpr, Int}(e => 0)
    fired = Vector{Rule}[]
    for pass in 1:(once ? 1 : maxpasses)
        push!(fired, Rule[])
        e′ = _pass(e, ctx, false, fired[end])
        (once || isequal(e′, e)) && return e′
        haskey(seen, e′) && _cycle_error(reduce(vcat, fired[(seen[e′] + 1):pass]), e′)
        seen[e′] = pass
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
        v === nothing || isequal(v, e) || return (v, r)
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
    return any(i -> !isequal(new[i], args[i]), eachindex(args)) ? _rebuild(e, new) : e
end

# A call of an undefined function `f(a, b)` is Giac's `of(f, a, b)`; the name itself is callable.
# A rebuilt sum or product has its like terms collected (`regroup`: `f(t) + f(t)` is `2f(t)`,
# `x*x` is `x^2`), which Giac's arithmetic leaves as written; nothing is expanded.
function _rebuild(e::GiacExpr, args)
    h = symb_funcname(e)
    h == "of" && return args[1](args[2:end]...)
    v = operation(e)(args...)
    return h in ("+", "*") ? Giac.invoke_cmd(:regroup, v) : v
end

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
    RewriteTrace

The path a traced [`rewrite`](@ref) or [`simplest`](@ref) took: a vector of the expressions along
it, so `first(t)` is the start, `last(t)` the result, `t[i]` the expression after step `i - 1`,
and `length(t)` one more than the number of steps. `t.by[i]` is the rule or transform that
turned `t[i]` into `t[i + 1]`. Displays as aligned equations, one per step, each labelled with
what made it; repeats of a rule show as one step unless traced with `trace = :steps`.
"""
struct RewriteTrace <: AbstractVector{GiacExpr}
    exprs::Vector{GiacExpr}
    by::Vector{Any}
    collapse::Bool
end
Base.size(t::RewriteTrace) = size(t.exprs)
Base.getindex(t::RewriteTrace, i::Int) = t.exprs[i]
Base.IndexStyle(::Type{RewriteTrace}) = IndexLinear()

# `steps` are `(expression after, what made it)`
_trace_display(e0, steps; collapse::Bool) =
    RewriteTrace(GiacExpr[e0; GiacExpr[first(s) for s in steps]], Any[last(s) for s in steps], collapse)

# The rows of the display: one per step (or per run of a rule, collapsed), the first starting from
# the expression the trace starts with.
function _trace_rows(t::RewriteTrace)
    runs = Tuple{GiacExpr, Any, Int}[]
    for (e, label) in zip(t.exprs[2:end], t.by)
        if t.collapse && !isempty(runs) && runs[end][2] === label
            runs[end] = (e, label, runs[end][3] + 1)
        else
            push!(runs, (e, label, 1))
        end
    end
    lhs = Giac.giac_latex(first(t))
    isempty(runs) && return [(lhs, Giac.giac_latex(first(t)) * " \\qquad \\text{(no rule applies)}")]
    rows = Tuple{String, String}[]
    for (e, label, n) in runs
        times = n > 1 ? " \\;(\\times $n)" : ""
        push!(rows, (lhs, Giac.giac_latex(e) * " \\qquad \\text{by } " * _label_latex(label) * times))
        lhs = "{}"
    end
    return rows
end

Base.show(io::IO, m::MIME"text/latex", t::RewriteTrace) = show(io, m, Giac.Derivation(_trace_rows(t)))
Base.show(io::IO, m::MIME"text/markdown", t::RewriteTrace) = show(io, m, Giac.Derivation(_trace_rows(t)))
Base.show(io::IO, t::RewriteTrace) = print(io, "RewriteTrace(", first(t), " → ", last(t), ", ", length(t.by), " step(s))")
function Base.show(io::IO, ::MIME"text/plain", t::RewriteTrace)
    print(io, "RewriteTrace with ", length(t.by), " step(s):\n  ", first(t))
    for (e, label) in zip(t.exprs[2:end], t.by)
        print(io, "\n  = ", e, "    by ", label isa Rule ? label.source : label isa Transform ? label.name : nameof(label))
    end
end

# In a `@derivation`, a trace continues the step it is the value of: `lhs = start`, then one row
# per step.
function Giac._derivation_rows(left, written, t::RewriteTrace, simplifier)
    steps = _trace_rows(t)
    rows = [(left, join(vcat(written, first(steps[1])), " = "))]
    append!(rows, (("{}", last(r)) for r in steps))
    return rows
end

_label_latex(r::Rule) = _rule_latex(r)
_label_latex(f) = "\\text{" * string(nameof(f)) * "}"
