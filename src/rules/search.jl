# Searching rewrites for the simplest form, where applying rules greedily would stop at a local
# minimum (a form that must get longer before it gets shorter).

"""
    expression_size(e)

The number of nodes in the tree of `e`: the default cost for [`simplest`](@ref).
"""
expression_size(e::GiacExpr) = iscall(e) ? 1 + sum(expression_size, arguments(e); init = 0) : 1

"""
    simplest(e; goal = :compact, verify = false, depth = 6, width = 10)
    simplest(e, transforms...; cost, goal, verify, depth, width)

Search rewrites of `e` for the form that best meets `goal`, and return the path to it as a trace
(see [`rewrite`](@ref); `last(t)` is the result). Without transforms it uses
[`standard_transforms`](@ref): Giac's normal form, factoring, trigonometric forms and the rule
sets for identities with conditions. Each transform is a rule, a vector of rules, a
[`Transform`](@ref), or a function of a whole expression such as `expand`.

Each step applies one transform in one place; the search keeps the `width` best expressions at
each of up to `depth` steps, so it can pass through a longer form on the way to a shorter one.

- `goal`: `:compact`, `:expanded`, `:factored` or `:trig_linear` (see [`goal_cost`](@ref)); or
  give `cost`, a function of an expression returning anything comparable.
- `verify = true` takes only steps that [`equivalent`](@ref) confirms, so a transformation that
  is wrong for the values in question (under the declared assumptions) is never used.

```julia
simplest(sin(t)^4 - cos(t)^4)                    # -cos(2t), through factor, sin²+cos² = 1, expand
simplest(sin(t)^3; goal = :trig_linear)          # 3sin(t)/4 - sin(3t)/4
```
"""
function simplest(e::GiacExpr, transforms...; goal::Symbol = :compact, cost = nothing, verify::Bool = false,
                  depth::Integer = 6, width::Integer = 10)
    rules, fns = Rule[], Any[]
    for t in (isempty(transforms) ? standard_transforms() : transforms)
        t isa Rule ? push!(rules, t) :
        t isa AbstractVector{<:Rule} ? append!(rules, t) :
        (t isa Transform || applicable(t, e)) ? push!(fns, t) :
        throw(ArgumentError("a transform is a rule, a vector of rules, or a function of an expression; got $t"))
    end
    score = cost === nothing ? (x -> (goal_cost(goal, x)..., length(_key(x)))) : (x -> (cost(x), length(_key(x))))
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
            verify && !equivalent(x, y) && continue
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

simplest(A::AbstractArray{GiacExpr}, transforms...; kwargs...) = map(e -> last(simplest(e, transforms...; kwargs...)), A)

# every expression one step from `e`: a rule at any place in it, or a function of all of it
function _successors(e::GiacExpr, rules, fns)
    out = _rule_successors(e, rules)
    for f in fns
        v = try
            f(e)
        catch
            continue
        end
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
