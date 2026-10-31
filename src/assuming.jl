# Assumptions on variables: declared with the variable (`@giac_var k::positive`), or scoped to a
# block (`assuming`).

# The assumption each name was declared with. Giac's assumptions are global per name; this
# records the declared ones so `assuming` can restore them when its block ends.
const _DECLARED = Dict{String, Any}()
const _DECLARED_LOCK = ReentrantLock()

const _ASSUMPTION_TAGS = (:positive, :negative, :nonnegative, :nonpositive, :real, :integer)

"""
    _declare!(v, tag)

Give the variable `v` (or every entry of an array) exactly the assumption `tag`, dropping any it
had: `:positive`, `:negative`, `:nonnegative`, `:nonpositive`, `:real`, `:integer`, an open
interval `(a, b)`, or `nothing` for none. Used by `@giac_var`.
"""
function _declare!(v::GiacExpr, tag)
    _forget!(v)
    name = string(v)
    lock(() -> (tag === nothing ? delete!(_DECLARED, name) : (_DECLARED[name] = tag)), _DECLARED_LOCK)
    tag === nothing || _assume_tag!(v, tag)
    return v
end
_declare!(A::AbstractArray, tag) = (foreach(v -> _declare!(v, tag), A); A)

function _assume_tag!(v::GiacExpr, tag)
    if tag isa Tuple
        length(tag) == 2 || throw(ArgumentError("an interval assumption is a pair (a, b), got $tag"))
        Commands.assume(v > tag[1]); Commands.additionally(v < tag[2])
    elseif tag === :positive;    Commands.assume(v > 0)
    elseif tag === :negative;    Commands.assume(v < 0)
    elseif tag === :nonnegative; Commands.assume(v >= 0)
    elseif tag === :nonpositive; Commands.assume(v <= 0)
    elseif tag === :real || tag === :integer
        Commands.invoke_cmd(:assume, v, giac_eval(string(tag)))
    else
        throw(ArgumentError("unknown assumption `$tag`; use one of $(join(_ASSUMPTION_TAGS, ", ")) or an interval (a, b)"))
    end
    return nothing
end

# Drop whatever Giac assumes about `v` (purging a name it knows nothing about is harmless).
_forget!(v::GiacExpr) = (try Commands.purge(v) catch; end; nothing)

_declared(v::GiacExpr) = lock(() -> get(_DECLARED, string(v), nothing), _DECLARED_LOCK)

"""
    assuming(f, conditions...)
    assuming(conditions...) do
        …
    end

Run `f()` with Giac assumptions such as `k > 0` in force, then remove them. Several conditions
on one variable combine (`k > 0, k < 1`), and they add to what the variable was declared with
(`@giac_var k::real`), which is restored afterwards.

```julia
@giac_var k
assuming(k > 0) do
    sqrt(k^2)            # k
end
sqrt(k^2)                # abs(k)
```

Assumptions are Giac state: inside the block they apply to everything Giac evaluates.
"""
function assuming(f, conditions::GiacExpr...)
    vars = GiacExpr[]
    for c in conditions
        cvars = collect(GiacExpr, free_symbols(c))
        # a variable already assumed something (declared, or by an earlier condition) gets more
        known = any(v -> any(isequal(v), vars) || _declared(v) !== nothing, cvars)
        known ? Commands.additionally(c) : Commands.assume(c)
        append!(vars, filter(v -> !any(isequal(v), vars), cvars))
    end
    try
        return f()
    finally
        foreach(v -> _declare!(v, _declared(v)), vars)     # back to the declared assumption, or none
    end
end
