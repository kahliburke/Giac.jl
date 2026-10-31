# Searching rewrites for the simplest form, where applying rules greedily would stop at a local
# minimum (a form that must get longer before it gets shorter).

"""
    expression_size(e)

The number of nodes in the tree of `e`: the default cost for [`simplest`](@ref).
"""
expression_size(e::GiacExpr) = iscall(e) ? 1 + sum(expression_size, arguments(e); init = 0) : 1

"""
    simplest(e, transforms...; cost = expression_size, depth = 8, width = 16)

Search rewrites of `e` for the one of least `cost`, and return the path to it as a trace (see
[`rewrite`](@ref); `last(t)` is the result). Each transform is a rule, a vector of rules, or a
function of a whole expression such as `expand` or `factor`. Each step applies one transform in
one place; the search keeps the `width` cheapest expressions at each of up to `depth` steps, so
it can pass through a longer form on the way to a shorter one. Ties in cost go to the shorter
written form.

```julia
simplest(e, trig_simplify(), Giac.Commands.expand, Giac.Commands.factor)
```
"""
function simplest(e::GiacExpr, transforms...; cost = expression_size, depth::Integer = 8, width::Integer = 16)
    rules, fns = Rule[], Any[]
    for t in transforms
        t isa Rule ? push!(rules, t) :
        t isa AbstractVector{<:Rule} ? append!(rules, t) :
        applicable(t, e) ? push!(fns, t) :
        throw(ArgumentError("a transform is a rule, a vector of rules, or a function of an expression; got $t"))
    end
    score(x) = (cost(x), length(_key(x)))
    Path = Vector{Tuple{GiacExpr, Any}}
    best, best_score = (e, Path()), score(e)
    seen = Set([_key(e)])
    frontier = [(e, Path())]
    for _ in 1:depth
        next = Tuple{GiacExpr, Path}[]
        for (x, path) in frontier, (y, label) in _successors(x, rules, fns)
            k = _key(y)
            k in seen && continue
            push!(seen, k)
            push!(next, (y, vcat(path, [(y, label)])))
        end
        isempty(next) && break
        scores = map(s -> score(first(s)), next)
        order = sortperm(scores)
        frontier = next[order[1:min(width, end)]]
        scores[order[1]] < best_score && ((best, best_score) = (next[order[1]], scores[order[1]]))
    end
    return _trace_display(e, last(best); collapse = false)
end

# every expression one step from `e`: a rule at any place in it, or a function of all of it
function _successors(e::GiacExpr, rules, fns)
    out = _rule_successors(e, rules)
    for f in fns
        v = f(e)
        v isa GiacExpr && _key(v) != _key(e) && push!(out, (v, f))
    end
    return out
end

function _rule_successors(e::GiacExpr, rules)
    out = Tuple{GiacExpr, Any}[]
    for r in rules
        v = r(e)
        v === nothing || _key(v) == _key(e) || push!(out, (v, r))
    end
    iscall(e) || return out
    args = arguments(e)
    for i in eachindex(args), (y, label) in _rule_successors(args[i], rules)
        new = copy(args)
        new[i] = y
        push!(out, (_rebuild(e, new), label))
    end
    return out
end
