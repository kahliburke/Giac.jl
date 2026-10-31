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

# ---------------------------------------------------------------------------------------------
# Properties of functions: `@giac_var φ(x)::odd`, `ψ(t)::(even, periodic(2π))`
# ---------------------------------------------------------------------------------------------

"""
    periodic(T)

A property for a function declared with `@giac_var`: `@giac_var ψ(t)::periodic(2π)` makes
`ψ(t + 2π)`, `ψ(t - 4π)` and the like simplify to `ψ(t)`.
"""
periodic(T) = Periodic(convert(GiacExpr, T))
struct Periodic
    period::GiacExpr
end

"""
    inverse_of(g)

A property for a function declared with `@giac_var`: `@giac_var f(x)::inverse_of(g)` makes
`f(g(x))` and `g(f(x))` simplify to `x`. `g` is another function variable (`@giac_var g(x)`).
"""
inverse_of(g) = InverseOf(_function_name(g))
struct InverseOf
    name::String
end

const _FUNCTION_PROPERTIES = (:odd, :even, :involution, :additive, :linear)
const _FUNCTION_FACTS = Dict{String, Vector{Any}}()

# the name of a function variable `f(x)` (or of `f` itself)
function _function_name(f::GiacExpr)
    is_identifier(f) && return string(f)
    iscall(f) && symb_funcname(f) == "of" && return string(first(arguments(f)))
    throw(ArgumentError("$f is not a function variable; declare one with `@giac_var f(x)`"))
end

"""
Record exactly the properties `tags` for the function variable `f` (dropping any it had):
`:odd`, `:even`, `:involution`, `:additive`, `:linear`, `periodic(T)`, `inverse_of(g)`, a tuple
of these, or `nothing`. Used by `@giac_var`; the rules they give are `Rules.function_identities()`.
"""
function _declare_function!(f::GiacExpr, tags)
    list = tags === nothing ? Any[] : tags isa Tuple ? collect(Any, tags) : Any[tags]
    for t in list
        t in _FUNCTION_PROPERTIES || t isa Periodic || t isa InverseOf ||
            throw(ArgumentError("unknown property `$t` for a function; use one of $(join(_FUNCTION_PROPERTIES, ", ")), periodic(T) or inverse_of(g)"))
    end
    name = _function_name(f)
    lock(_DECLARED_LOCK) do
        isempty(list) ? delete!(_FUNCTION_FACTS, name) : (_FUNCTION_FACTS[name] = list)
    end
    return f
end

_function_facts() = lock(() -> copy(_FUNCTION_FACTS), _DECLARED_LOCK)
