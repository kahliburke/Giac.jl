# Substitute function for GiacExpr (028-substitute-mechanism, 065-substitute-tier1)
# Provides Symbolics.jl-compatible substitute(expr, Dict(...)) interface backed
# by the direct CxxWrap binding _giac_subst_vec_tier1 (in src/wrapper.jl).

# ============================================================================
# Public API: substitute function
# ============================================================================

"""
    substitute(expr::GiacExpr, dict::AbstractDict{<:GiacExpr}) -> GiacExpr

Substitute variables in a symbolic expression according to a dictionary mapping.

Performs simultaneous substitution of all variables in `dict`. The original
expression is not modified.

# Arguments
- `expr::GiacExpr`: The expression to transform
- `dict::AbstractDict`: Mapping from variables (GiacExpr) to replacement values

# Returns
- `GiacExpr`: New expression with substitutions applied

# Examples
```jldoctest
julia> @giac_var x y;

julia> expr = x^2 + y;

julia> string(substitute(expr, Dict(x => 2)))
"4+y"

julia> string(substitute(expr, Dict(x => 2, y => 3)))
"7"
```

# See also
- [`@giac_var`](@ref): Create symbolic variables
"""
function substitute(expr::GiacExpr, dict::AbstractDict{<:GiacExpr})::GiacExpr
    # Handle empty Dict - return original expression unchanged (no GIAC call).
    isempty(dict) && return expr
    # GIAC would evaluate a held integral: it is substituted into apart
    _any_node(_is_integral, _node(expr)) && return _around_integrals(e -> substitute(e, dict), expr)

    # A sum, product or power as a key is a relation to substitute, which Giac's subst does not
    # do: `x^2 + y^2 => r^2` (see `Rules._substitute_relation`).
    if any(_is_relation_key, keys(dict))
        plain = filter(p -> !_is_relation_key(first(p)), dict)
        expr = substitute(expr, plain)
        for (k, v) in dict
            _is_relation_key(k) && (expr = Rules._substitute_relation(expr, k, v))
        end
        return expr
    end

    # Collect parallel vectors of GiacExpr keys and untyped values, then
    # delegate to the direct-binding shim. Substitution is applied
    # simultaneously by giac_subst, so e.g. Dict(x => y, y => x) swaps.
    vars = collect(GiacExpr, keys(dict))
    vals = collect(values(dict))
    return _take_derivatives(_giac_subst_vec_tier1(expr, vars, vals))
end

# The derivatives a substitution put an expression into, `diff(q(t) + ε*η(t), t)` from
# `q(t) => q(t) + ε*η(t)`, taken: GIAC's subst leaves them as written.
function _take_derivatives(e::GiacExpr)
    n = _node(e)
    _any_node(_is_open_derivative, n) || return e
    return GiacExpr(_replace_nodes(n) do m
        _is_open_derivative(m) || return nothing
        return _node(invoke_cmd(:diff, (GiacExpr(a) for a in m.args)...))
    end)
end

_is_open_derivative(n::Node) = n isa CallNode && n.f == "diff" && length(n.args) >= 2 && !_is_coordinate(n) &&
                               !(n.args[1] isa CallNode && n.args[1].f == "of")
"""
    substitute(expr::GiacExpr, pairs::Pair{<:GiacExpr}...) -> GiacExpr

Substitute variables using one or more pair arguments, aligned with
`Symbolics.substitute`.

Equivalent to `substitute(expr, Dict(pairs))`. All pairs are applied
simultaneously (the canonical swap `substitute(expr, x => y, y => x)` therefore
swaps `x` and `y` rather than collapsing). Calling with zero pairs returns the
input expression unchanged.

# Examples
```jldoctest
julia> @giac_var x y;

julia> substitute(x + 1, x => 5)
GiacExpr: 6

julia> substitute(x*y, x => 1, y => 2)
GiacExpr: 2

julia> substitute(x + 2*y, x => y, y => x)
GiacExpr: y+2*x
```

# See also
- [`substitute(::GiacExpr, ::AbstractDict)`](@ref): Dict-based form
"""
substitute(expr::GiacExpr, pairs::Pair{<:GiacExpr}...)::GiacExpr =
    isempty(pairs) ? expr : substitute(expr, Dict(pairs))

_is_relation_key(k::GiacExpr) = iscall(k) && symb_funcname(k) in ("+", "*", "^", "-", "inv")

# ============================================================================
# GiacMatrix Support: Element-wise Substitution
# ============================================================================

"""
    substitute(m::GiacMatrix, dict::AbstractDict{<:GiacExpr}) -> GiacMatrix

Substitute variables in each element of a symbolic matrix.

Performs element-wise substitution, applying the same variable mappings
to every element of the matrix. Returns a new matrix with the same dimensions.

# Arguments
- `m::GiacMatrix`: The matrix to transform
- `dict::AbstractDict`: Mapping from variables (GiacExpr) to replacement values

# Returns
- `GiacMatrix`: New matrix with substitutions applied element-wise

# Examples
```julia
@giac_var x y
M = GiacMatrix([x+1 2*x; y x*y])
substitute(M, Dict(x => 2))        # Returns matrix with x=2 substituted
substitute(M, Dict(x => 2, y => 3)) # Returns fully numeric matrix
```

# See also
- [`substitute(::GiacExpr, ::AbstractDict)`](@ref): Scalar expression substitution
"""
function substitute(m::GiacMatrix, dict::AbstractDict{<:GiacExpr})::GiacMatrix
    # Handle empty Dict - return a structural copy of the matrix.
    if isempty(dict)
        result = Matrix{Any}(undef, m.rows, m.cols)
        for i in 1:m.rows, j in 1:m.cols
            result[i, j] = m[i, j]
        end
        return GiacMatrix(result)
    end

    # Build the variable/value vectors ONCE outside the element loop, then
    # call _giac_subst_vec_tier1 per element. This avoids re-vectorizing
    # the dict for every matrix entry while preserving simultaneous semantics.
    vars = collect(GiacExpr, keys(dict))
    vals = collect(values(dict))
    result = Matrix{Any}(undef, m.rows, m.cols)
    for i in 1:m.rows, j in 1:m.cols
        result[i, j] = _giac_subst_vec_tier1(m[i, j], vars, vals)
    end
    return GiacMatrix(result)
end

"""
    substitute(m::GiacMatrix, pairs::Pair{<:GiacExpr}...) -> GiacMatrix

Element-wise variant of [`substitute(::GiacExpr, ::Pair{<:GiacExpr}...)`](@ref)
for symbolic matrices.

Equivalent to `substitute(m, Dict(pairs))`. All pairs are applied simultaneously
to every element. Calling with zero pairs returns a structural copy of the
input matrix.

# Examples
```jldoctest
julia> @giac_var x;

julia> M = GiacMatrix([x 2*x; x+1 x^2]);

julia> result = substitute(M, x => 3);

julia> result[1, 1]
GiacExpr: 3

julia> result[2, 2]
GiacExpr: 9
```

# See also
- [`substitute(::GiacMatrix, ::AbstractDict)`](@ref): Dict-based matrix form
"""
substitute(m::GiacMatrix, pairs::Pair{<:GiacExpr}...)::GiacMatrix =
    substitute(m, isempty(pairs) ? Dict{GiacExpr,Any}() : Dict(pairs))

# ============================================================================
# Vectors of variables, mixed pairs, and arrays of expressions
# ============================================================================

# A pair of arrays substitutes entry by entry: `x => [1, 2]` for `x = [x1, x2]`.
_substitution_pairs(p::Pair{<:AbstractArray, <:AbstractArray}) =
    size(first(p)) == size(last(p)) ? Pair.(vec(first(p)), vec(last(p))) :
    throw(DimensionMismatch("cannot substitute values of size $(size(last(p))) for variables of size $(size(first(p)))"))
# a tuple of values keeps exact constants that an array literal would promote: `x => (π, 0)`
_substitution_pairs(p::Pair{<:AbstractArray, <:Tuple}) = _substitution_pairs(first(p) => collect(Any, last(p)))
_substitution_pairs(p::Pair) = (p,)
# an equation `y ~ 2x` puts its right side in place of its left side
_substitution_pairs(eq::GiacExpr) = (lhs(eq) => rhs(eq),)
_substitution_pairs(eqs::AbstractVector{GiacExpr}) = map(eq -> lhs(eq) => rhs(eq), eqs)

"""
    substitute(expr, x => [1, 2], y => 3, ...)
    substitute(expr, y ~ 2x, ...)

Pairs may map a vector (or array) of variables to values of the same shape, entry by entry,
and may be mixed with scalar pairs. Give the values as a tuple, `x => (π, 0)`, to keep exact
constants: the array literal `[π, 0]` is a `Vector{Float64}`.

An equation `a ~ b`, or a vector of them (such as the result of [`isolate`](@ref)), substitutes
`b` for `a`.
"""
function substitute(expr::GiacExpr, rules::Union{Pair, GiacExpr, AbstractVector{GiacExpr}}...)::GiacExpr
    isempty(rules) && return expr
    return substitute(expr, Dict{GiacExpr, Any}(p for q in rules for p in _substitution_pairs(q)))
end

"""
    substitute(A::AbstractArray{GiacExpr}, pairs...) -> Array{GiacExpr}

Substitute in every entry of an array of expressions.
"""
substitute(A::AbstractArray{GiacExpr}, args...) = map(e -> substitute(e, args...), A)
