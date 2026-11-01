# Assumptions on variables: declared with the variable (`@giac_var k::positive`), or scoped to a
# block (`assuming`).

# The assumption each name was declared with. Giac's assumptions are global per name; this
# records the declared ones so `assuming` can restore them when its block ends.
const _DECLARED = Dict{String, Any}()
const _DECLARED_LOCK = ReentrantLock()

const _ASSUMPTION_TAGS = (:positive, :negative, :nonnegative, :nonpositive, :real, :integer, :complex)

"""
    _declare!(v, tag)

Give the variable `v` (or every entry of an array) exactly the assumption `tag`, dropping any it
had: `:positive`, `:negative`, `:nonnegative`, `:nonpositive`, `:real`, `:integer`, `:complex`, an open
interval `(a, b)`, or `nothing` for none. Used by `@giac_var`.
"""
# When each name was last declared: products are written with parameters in this order.
const _DECLARATION_ORDER = Dict{String, Int}()
_note_declared(name::AbstractString) = lock(() -> (_DECLARATION_ORDER[name] = length(_DECLARATION_ORDER) + 1 +
                                                   maximum(values(_DECLARATION_ORDER); init = 0)), _DECLARED_LOCK)

# `scope` is `:global`, `:local` (a declaration in a function or `let`, which belongs to the task
# that ran it; see "Declarations in a local scope" below), or `:auto`: local if this task already
# holds the name locally, as when `assuming` or a derivation's condition adds to a local
# declaration. All in one GIAC job, so no other task sees the name half declared.
function _declare!(v::GiacExpr, tag; scope::Symbol = :auto)
    name = string(v)
    f = _current_frame()
    local_ = scope === :local || (scope === :auto && f !== nothing && haskey(f.tags, name))
    local_ && (f = _new_frame!())                     # in the calling task, before handing off
    with_giac_lock() do
        if local_
            if get(_OWNER, name, nothing) !== f      # the task's first local declaration of the name
                _GLOBAL_STATE[name] = _giac_state(name)
                _OWNER[name] = f
            end
            f.tags[name] = tag
        else
            # a global declaration replaces a local one this task held
            if f !== nothing && get(_OWNER, name, nothing) === f
                delete!(_OWNER, name); delete!(_GLOBAL_STATE, name); delete!(f.tags, name)
            end
            lock(() -> (tag === nothing ? delete!(_DECLARED, name) : (_DECLARED[name] = tag)), _DECLARED_LOCK)
        end
        _forget!(v)
        _note_declared(name)
        tag === nothing || _assume_tag!(v, tag)
    end
    return v
end
# every entry that is a variable (not the zeros or negated entries of a structured matrix)
_declare!(A::AbstractArray, tag; scope::Symbol = :auto) =
    (foreach(v -> is_identifier(v) && _declare!(v, tag; scope), A); A)

# Where the macro's binding `s` lives, just after the declaration assigned it `v`: the module's
# global holds `v` itself only when the declaration ran at top level
_binding_scope(mod::Module, s::Symbol, v) = isdefined(mod, s) && getglobal(mod, s) === v ? :global : :local

function _assume_tag!(v::GiacExpr, tag)
    if tag isa Tuple
        length(tag) == 2 || throw(ArgumentError("an interval assumption is a pair (a, b), got $tag"))
        Commands.assume(v > tag[1]); Commands.additionally(v < tag[2])
    elseif tag === :positive;    Commands.assume(v > 0)
    elseif tag === :negative;    Commands.assume(v < 0)
    elseif tag === :nonnegative; Commands.assume(v >= 0)
    elseif tag === :nonpositive; Commands.assume(v <= 0)
    elseif tag === :real || tag === :integer || tag === :complex
        Commands.invoke_cmd(:assume, v, giac_eval(string(tag)))
    else
        throw(ArgumentError("unknown assumption `$tag`; use one of $(join(_ASSUMPTION_TAGS, ", ")) or an interval (a, b)"))
    end
    return nothing
end

# Drop whatever Giac assumes about `v` (purging a name it knows nothing about is harmless).
_forget!(v::GiacExpr) = (try Commands.purge(v) catch; end; nothing)

# The assumption a name was declared with: this task's local declaration of it, else the global one
function _declared_tag(name::AbstractString)
    f = _current_frame()
    f !== nothing && haskey(f.tags, name) && return f.tags[name]
    return lock(() -> get(_DECLARED, name, nothing), _DECLARED_LOCK)
end
_declared(v::GiacExpr) = _declared_tag(string(v))

# ---------------------------------------------------------------------------------------------
# Declarations in a local scope
# ---------------------------------------------------------------------------------------------
# GIAC keeps one state per name: `assume(a > 0)` holds for every caller. A declaration in a function
# or `let` belongs to the task that ran it instead, so code on another thread (a notebook runs cells
# in parallel) neither sees it nor disturbs it, the way a Julia local isn't another task's business.
# A task keeps its local names in a frame. When a task takes the GIAC lock, GIAC's state for those
# names is made its own: a name another task's frame holds is saved into that frame and the global
# state put back, and this task's own names are put in place. While one task works alone nothing
# moves. A saved state is whatever GIAC holds for the name (`about`), so conditions added to a local
# declaration, by `assuming` or a derivation, travel with it.

mutable struct _Frame
    tags::Dict{String, Any}           # the task's local declarations, for Julia-side code
    saved::Dict{String, GiacExpr}     # GIAC's state for a local name while another task holds it
end

const _OWNER = Dict{String, _Frame}()            # names whose GIAC state is a frame's, not the global one
const _GLOBAL_STATE = Dict{String, GiacExpr}()   # the global state of those names, to put back

# The calling task's frame. On the GIAC thread, that of the task whose job is running.
function _current_frame()
    _on_giac_thread() && return _ACTIVE_FRAME[]::Union{Nothing, _Frame}
    s = current_task().storage
    s === nothing && return nothing
    return get(s, :giac_frame, nothing)::Union{Nothing, _Frame}
end
_empty_frame() = _Frame(Dict{String, Any}(), Dict{String, GiacExpr}())
# The calling task's frame, made if it has none. On the GIAC thread, the running job's.
function _new_frame!()
    _on_giac_thread() || return get!(_empty_frame, task_local_storage(), :giac_frame)::_Frame
    _ACTIVE_FRAME[] === nothing && (_ACTIVE_FRAME[] = _empty_frame())
    return _ACTIVE_FRAME[]::_Frame
end

# GIAC's state for a name: its assumption (or value), or the name itself when it has neither
_giac_state(name::AbstractString) = Commands.invoke_cmd(:about, giac_identifier(name))
function _set_giac_state!(name::AbstractString, s::GiacExpr)
    v = giac_identifier(name)
    _forget!(v)
    isequal(_node(s), _node(v)) || Commands.invoke_cmd(:sto, s, v)
    return nothing
end

# Make GIAC's state for local names this task's. Run by the outermost `with_giac_lock`, so the
# lock is held.
function _sync_local_names!()
    f = _current_frame()
    isempty(_OWNER) && (f === nothing || isempty(f.tags)) && return nothing
    release = String[]
    for (name, g) in _OWNER
        g === f || push!(release, name)
    end
    for name in release
        _OWNER[name].saved[name] = _giac_state(name)
        _set_giac_state!(name, pop!(_GLOBAL_STATE, name))
        delete!(_OWNER, name)
    end
    f === nothing && return nothing
    for name in keys(f.tags)
        get(_OWNER, name, nothing) === f && continue
        _GLOBAL_STATE[name] = _giac_state(name)
        _set_giac_state!(name, f.saved[name])
        _OWNER[name] = f
    end
    return nothing
end

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

# ---------------------------------------------------------------------------------------------
# Properties of matrices, as facts about their entries: `@giac_matrix Q[2, 2]::orthogonal`
# ---------------------------------------------------------------------------------------------

# name => (relations: polynomials equal to 0, positive: quantities > 0, nonzero: quantities ≠ 0)
const _MATRIX_FACTS = Dict{String, NamedTuple{(:relations, :positive, :nonzero), NTuple{3, Vector{GiacExpr}}}}()

"""
Record what the properties `props` (`:orthogonal`, `:posdef`, `:negdef`, `:invertible`) say about
the entries of the matrix `A` declared as `name`, replacing what was recorded for that name:
relations among the entries (`QᵀQ = QQᵀ = I`), quantities known positive (the leading principal
minors of a positive definite matrix, with its diagonal entries assumed positive) and quantities
known nonzero (a determinant). Used by `@giac_matrix`.
"""
function _declare_matrix!(A::AbstractMatrix, name::String, props::Tuple)
    relations, positive, nonzero = GiacExpr[], GiacExpr[], GiacExpr[]
    n = size(A, 1)
    for p in props
        size(A, 1) == size(A, 2) || throw(ArgumentError("$name is not square, so it cannot be $p"))
        if p === :orthogonal
            At = permutedims(A)
            for M in (At * A, A * At), i in 1:n, j in i:n
                r = Commands.expand(M[i, j] - (i == j ? 1 : 0))
                string(r) == "0" || any(isequal(r), relations) || push!(relations, r)
            end
        elseif p === :posdef || p === :negdef
            for k in 1:n
                minor = Commands.expand(LinearAlgebra.det(A[1:k, 1:k]))
                push!(positive, p === :posdef || iseven(k) ? minor : -minor)
            end
            for i in 1:n
                is_identifier(A[i, i]) && _declare!(A[i, i], p === :posdef ? :positive : :negative)
            end
            push!(nonzero, positive[end])
        elseif p === :invertible
            push!(nonzero, Commands.expand(LinearAlgebra.det(A)))
        else
            throw(ArgumentError("unknown matrix property `$p`; use orthogonal, posdef, negdef or invertible"))
        end
    end
    lock(_DECLARED_LOCK) do
        isempty(relations) && isempty(positive) && isempty(nonzero) ? delete!(_MATRIX_FACTS, name) :
            (_MATRIX_FACTS[name] = (; relations, positive, nonzero))
    end
    return A
end

_facts(field) = lock(_DECLARED_LOCK) do
    reduce(vcat, (getfield(f, field) for f in values(_MATRIX_FACTS)); init = GiacExpr[])
end
"Every recorded relation among matrix entries (polynomials equal to 0)."
_relations() = _facts(:relations)
_positive_facts() = _facts(:positive)
_nonzero_facts() = _facts(:nonzero)
