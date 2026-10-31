# Macro definitions for Giac.jl
# Feature: 011-giac-symbol-macro
# Feature: 012-giac-several-var
# Feature: 033-giac-var-function

# Helper functions for @giac_var function syntax

"""
    _build_function_string(funcname::Symbol, args::Vector{Symbol}) -> String

Build a GIAC-compatible string representation of a function call.
E.g., `_build_function_string(:u, [:t])` returns `"u(t)"`.
"""
function _build_function_string(funcname::Symbol, args::Vector{Symbol})
    return string(funcname, "(", join(string.(args), ","), ")")
end

"""
    _process_giac_var_arg(arg) -> Tuple{Symbol, String}

Process a single argument to @giac_var macro.
Returns (varname, giac_string) where:
- varname is the Julia variable name to bind
- giac_string is the GIAC expression string

For symbols: returns (sym, string(sym))
For function calls: returns (funcname, "funcname(arg1,arg2,...)")

Throws ArgumentError for invalid inputs.
"""
function _process_giac_var_arg(arg)
    if arg isa Symbol
        # Simple variable: x -> ("x", x)
        return (arg, string(arg))
    elseif arg isa Expr && arg.head == :call
        # Function call: u(t) -> ("u(t)", u)
        funcname = arg.args[1]
        funcargs = arg.args[2:end]

        # Validate function name is a symbol
        if !(funcname isa Symbol)
            throw(ArgumentError("Function name must be a symbol, got $(typeof(funcname)): $funcname"))
        end

        # Validate at least one argument
        if isempty(funcargs)
            throw(ArgumentError("Function syntax requires at least one argument: @giac_var $(funcname)() is invalid. Use @giac_var $funcname for a simple variable."))
        end

        # Validate all arguments are symbols
        for (i, farg) in enumerate(funcargs)
            if !(farg isa Symbol)
                throw(ArgumentError("Function arguments must be symbols, got $(typeof(farg)): $farg in position $i"))
            end
        end

        giac_string = _build_function_string(funcname, Symbol.(funcargs))
        return (funcname, giac_string)
    else
        if arg isa String
            throw(ArgumentError("@giac_var arguments must be symbols, not strings. Use giac_eval(\"$arg\") instead."))
        else
            throw(ArgumentError("@giac_var arguments must be symbols or function calls, got $(typeof(arg)): $arg"))
        end
    end
end

# Helper functions for @giac_several_vars

"""
    _needs_separator(dims) -> Bool

Check if any dimension exceeds 9, requiring underscore separators in variable names.
When any dimension > 9, indices could be >= 10 and need separators for clarity.
Also returns true if any index could be negative (requires separator).
"""
function _needs_separator(dims)
    return any(d -> d > 9, dims)
end

"""
    _needs_separator_for_macro_ranges(ranges) -> Bool

Check if underscore separators are needed for ranges in macro context.
Returns true if any range has an index > 9 or < 0.
"""
function _needs_separator_for_macro_ranges(ranges)
    for r in ranges
        if !isempty(r) && (minimum(r) < 0 || maximum(r) > 9)
            return true
        end
    end
    return false
end

"""
    _is_range_expr(expr) -> Bool

Check if an expression is a range expression (a:b or a:b:c).
"""
function _is_range_expr(expr)
    return expr isa Expr && expr.head == :call && expr.args[1] == :(:)
end

"""
    _eval_dim_to_range(dim)

Convert a dimension specification to a range.
Integer n becomes 1:n for backward compatibility.
Range expressions are evaluated to actual ranges.
"""
function _eval_dim_to_range(dim)
    if dim isa Integer
        return 1:dim
    elseif _is_range_expr(dim)
        # Evaluate range expression at macro expansion time
        return eval(dim)
    else
        throw(ArgumentError("Dimension must be integer or range expression, got $(typeof(dim)): $dim"))
    end
end

"""
    _format_index(idx::Integer) -> String

Format a single index for use in variable names.
Negative indices are encoded with 'm' prefix (e.g., -1 → "m1") to avoid
GIAC interpreting the minus sign as a subtraction operator.
"""
function _format_index(idx::Integer)
    if idx < 0
        return "m" * string(-idx)
    else
        return string(idx)
    end
end

"""
    _format_indices(base::Symbol, indices::Tuple, needs_sep::Bool) -> Symbol

Generate a variable name from base name and indices.
If needs_sep is true, uses underscore separators (e.g., `a_1_10_3`).
Otherwise, concatenates indices directly (e.g., `a123`).
Negative indices are encoded with 'm' prefix (e.g., -1 → `a_m1`).
"""
function _format_indices(base::Symbol, indices::Tuple, needs_sep::Bool)
    formatted = [_format_index(idx) for idx in indices]
    name = needs_sep ? Symbol(base, "_", join(formatted, "_")) : Symbol(base, join(formatted))
    # the name says `m1`; it is written with the index, -1
    if any(<(0), indices)
        _TEX_NAMES[replace(string(name), GIAC_RESERVED_LETTERS...)] =
            _tex_name(string(base)) * "_{" * join(string.(indices), ",") * "}"
    end
    return name
end

"""
    giac_identifier(name::AbstractString) -> GiacExpr

The symbolic identifier (or unevaluated call, e.g. `"u(t)"`) `name`, created without evaluating
it. Evaluating a bare name returns whatever value a Giac context has assigned to it, so a
"variable" made that way can silently be a stored value instead.
"""
giac_identifier(name::AbstractString) =
    giac_eval("'" * replace(name, GIAC_RESERVED_LETTERS...) * "'")

# Letters Giac's parser reserves (Δ and Σ are set operators), spelled out instead. Giac typesets
# the spelled-out names as the letters again: `Delta` renders as \Delta.
const GIAC_RESERVED_LETTERS = ("Δ" => "Delta", "Σ" => "Sigma", "∂" => "partial")

"""
    giac_array(base::Symbol, dims...; structure = :general) -> Array{GiacExpr}

A Julia array of symbolic entries named from `base` and their indices, following the same
naming scheme as [`GiacMatrix(base, dims...)`](@ref) (`a12`, `a_1_10`, `a_m1`). Each dimension
is an integer `n` (meaning `1:n`) or a range.

`structure` shapes a square matrix (two equal ranges): `:symmetric` (entry `(j, i)` is entry
`(i, j)`, named with `i ≤ j`), `:skew` (entry `(j, i)` is minus entry `(i, j)`, the diagonal 0),
`:diagonal`, `:upper` or `:lower` (triangular, the rest 0). `symmetric = true` is
`structure = :symmetric`.

The result is an ordinary `Array{GiacExpr}`, so Julia's generic linear algebra, slicing and
broadcasting apply to it.

```julia
giac_array(:x, 2)                         # [x1, x2]
giac_array(:a, 2, 2)                      # [a11 a12; a21 a22]
giac_array(:p, 2, 2; structure = :symmetric)  # [p11 p12; p12 p22]
giac_array(:w, 3, 3; structure = :skew)   # [0 w12 w13; -w12 0 w23; -w13 -w23 0]
```
"""
function giac_array(base::Symbol, dims::Union{Integer, OrdinalRange}...; symmetric::Bool = false,
                    structure::Symbol = symmetric ? :symmetric : :general)
    isempty(dims) && throw(ArgumentError("giac_array needs at least one dimension"))
    ranges = map(_to_range, dims)
    needs_sep = _needs_separator_for_macro_ranges(ranges)
    entry(idx) = giac_identifier(string(_format_indices(base, idx, needs_sep)))
    structure === :general && return GiacExpr[entry(idx) for idx in Iterators.product(ranges...)]
    structure in _STRUCTURES || throw(ArgumentError("unknown structure :$structure; use one of $(join(_STRUCTURES, ", "))"))
    (length(ranges) == 2 && ranges[1] == ranges[2]) ||
        throw(ArgumentError("a $structure array needs two equal index ranges, got $(ranges)"))
    r = ranges[1]
    zero_ = zero(GiacExpr)
    value(i, j) = structure === :symmetric ? entry(i ≤ j ? (i, j) : (j, i)) :
                  structure === :skew ? (i == j ? zero_ : i < j ? entry((i, j)) : -entry((j, i))) :
                  structure === :diagonal ? (i == j ? entry((i, j)) : zero_) :
                  structure === :upper ? (i ≤ j ? entry((i, j)) : zero_) :
                  (i ≥ j ? entry((i, j)) : zero_)
    return GiacExpr[value(i, j) for i in r, j in r]
end

const _STRUCTURES = (:symmetric, :skew, :diagonal, :upper, :lower)

# whether entry (i, j) of a matrix with `structure` is a variable of its own (bound by name)
_own_entry(structure, i, j) = structure === :general || structure === :symmetric && i ≤ j ||
    structure === :skew && i < j || structure === :diagonal && i == j ||
    structure === :upper && i ≤ j || structure === :lower && i ≥ j

"""
Code for an array declaration `name = giac_array(base, dims...)`. When every dimension is a
literal, each distinct entry is also bound to a variable of its own name (`a11`, `a12`, …).
"""
function _array_declaration(name::Symbol, base::Symbol, dims; structure::Symbol = :general)
    decl = :($(esc(name)) = giac_array($(QuoteNode(base)), $(map(esc, dims)...); structure = $(QuoteNode(structure))))
    literal_range(d) = _is_range_expr(d) && all(a -> a isa Integer, d.args[2:end])
    literal = all(d -> d isa Integer || literal_range(d), dims)
    literal || return decl
    ranges = _eval_dim_to_range.(dims)
    needs_sep = _needs_separator_for_macro_ranges(ranges)
    bindings = Expr[]
    seen = Set{Symbol}()
    for (k, idx) in enumerate(Iterators.product(ranges...))
        length(idx) == 2 && !_own_entry(structure, idx...) && continue
        entry = _format_indices(base, idx, needs_sep)
        entry in seen && continue
        push!(seen, entry)
        cart = Tuple(CartesianIndices(Tuple(length.(ranges)))[k])
        push!(bindings, :($(esc(entry)) = $(esc(name))[$(cart...)]))
    end
    return Expr(:block, decl, bindings...)
end

# matrix properties recorded as facts about the entries (see `_declare_matrix!`); `posdef` and
# `negdef` also make the matrix symmetric
const _MATRIX_PROPERTIES = (:orthogonal, :posdef, :negdef, :invertible)

"""
    @giac_matrix A[m, n]
    @giac_matrix P[n, n] symmetric
    @giac_matrix P[n, n]::posdef Q[n, n]::orthogonal W[n, n]::skew

Declare a symbolic matrix `A` (an ordinary `Matrix{GiacExpr}`) whose entries are named with the
lowercased matrix name and their indices, the usual `aᵢⱼ` convention: `@giac_matrix A[2, 2]`
gives `A = [a11 a12; a21 a22]` and binds `a11, a12, a21, a22`. Dimensions may be integers or
ranges, as for [`giac_array`](@ref). Several declarations can be given at once.

A tag after `::` (or a tuple of them) shapes the matrix or records what is known about it:

- structure: `symmetric`, `skew`, `diagonal`, `upper`, `lower` (see [`giac_array`](@ref));
  `symmetric` may also follow the declaration as a word, `@giac_matrix P[2, 2] symmetric`;
- `posdef`, `negdef`: symmetric, with the diagonal entries assumed positive (negative) and each
  leading principal minor recorded as positive (with alternating signs);
- `orthogonal`: the relations `QᵀQ = QQᵀ = I` among the entries are recorded;
- `invertible`: `det A ≠ 0` is recorded;
- an assumption for every entry, as with [`@giac_var`](@ref): `real`, `positive`, ….

Recorded relations are used by [`Rules.simplest`](@ref) and [`Rules.equivalent`](@ref) (a
Gröbner reduction modulo them), and recorded signs by the tests `is_positive`, `is_nonzero` and
the like. A new declaration of the name replaces them.
"""
macro giac_matrix(args...)
    isempty(args) && throw(ArgumentError("@giac_matrix needs a declaration such as A[2, 2]"))
    decls = Expr[]
    names = Symbol[]
    i = 1
    while i <= length(args)
        spec, tag = args[i] isa Expr && args[i].head == :(::) ? (args[i].args[1], args[i].args[2]) : (args[i], nothing)
        (spec isa Expr && spec.head == :ref && spec.args[1] isa Symbol) ||
            throw(ArgumentError("@giac_matrix expects declarations like A[2, 2], got $spec"))
        word = i < length(args) && args[i+1] === :symmetric
        # a tuple of names is a list of tags; anything else (`real`, `(0, 1)`) is one
        tags = tag isa Expr && tag.head === :tuple && all(a -> a isa Symbol, tag.args) ? collect(tag.args) :
               tag === nothing ? Any[] : Any[tag]
        word && push!(tags, :symmetric)
        shapes = filter(t -> t in _STRUCTURES, tags)
        any(t -> t in (:posdef, :negdef), tags) && !(:symmetric in shapes) && push!(shapes, :symmetric)
        length(shapes) <= 1 || throw(ArgumentError("@giac_matrix: a matrix has one structure, got $(join(shapes, " and "))"))
        properties = filter(t -> t in _MATRIX_PROPERTIES, tags)
        entry_tags = filter(t -> !(t in _STRUCTURES) && !(t in _MATRIX_PROPERTIES), tags)
        length(entry_tags) <= 1 || throw(ArgumentError("@giac_matrix: one assumption for the entries, got $(join(entry_tags, ", "))"))
        name = spec.args[1]
        push!(decls, _array_declaration(name, Symbol(lowercase(string(name))), spec.args[2:end];
                                        structure = isempty(shapes) ? :general : only(shapes)))
        tagex = isempty(entry_tags) ? nothing : only(entry_tags) isa Symbol ? QuoteNode(only(entry_tags)) : esc(only(entry_tags))
        push!(decls, :(_declare!($(esc(name)), $tagex)))
        push!(decls, :(_declare_matrix!($(esc(name)), $(string(name)), $(Tuple(properties)))))
        push!(names, name)
        i += word ? 2 : 1
    end
    return Expr(:block, decls..., Expr(:tuple, map(esc, names)...))
end

"""
    @giac_var sym...
    @giac_var func(args...)

Create symbolic variables or function expressions from Julia symbols.

Creates `GiacExpr` variables in the calling scope by internally calling
`giac_eval` with the stringified expression. This provides a cleaner
syntax for variable declaration similar to `@variables` in Symbolics.jl.

# Arguments

- `sym...`: Symbol names to create as simple variables
- `func(args...)`: Function syntax to create function expressions

# Examples

Single variable:
```julia
@giac_var x           # Creates x as a GiacExpr
string(x)             # "x"
x isa GiacExpr        # true
```

Multiple variables:
```julia
@giac_var x y z       # Creates x, y, z as GiacExpr variables
```

Assumptions, declared with the variable:
```julia
@giac_var k::positive n::integer x::real y::(0, 1) w[1:2]::nonnegative
sqrt(k^2)             # k
sin(n * pi)           # 0
```
The tags are `positive`, `negative`, `nonnegative`, `nonpositive`, `real`, `integer`, or an open
interval `(a, b)`. Assumptions are GIAC state, global per name: each declaration replaces what
was assumed about the name before (a declaration without a tag leaves it with none).
`assuming` adds to a declared assumption for the length of a block.

A function variable can carry properties, which `simplest` and `Rules.function_identities()`
use: `odd`, `even`, `involution` (f(f(x)) = x), `additive`, `linear` (additive, and constant
factors come out), `periodic(T)`, `inverse_of(g)`, or a tuple of them.
```julia
@giac_var φ(x)::odd ψ(t)::(even, periodic(2π))
```

Names are written as typed: `x₁`, `x_1` and `x1` are x₁; `x_ref` has an upright subscript;
`x̂`, `V̇`, `ẍ` carry accents; `kₚ`, `Kᵀ`, `x⁺`, `P★`, `x′` have their sub- and superscripts; `Δx`
is Δx. For anything else, give the LaTeX with the name:
```julia
@giac_var m_c => "m_{\\text{cart}}" (k => "k_\\theta")::positive
```

A function of one variable, `q(t)`, followed by its name with dots declares its derivatives:
```julia
@giac_var t q(t) q̇ q̈        # q̇ is diff(q, t), q̈ is diff(q, t, 2)
q̈ + 3q̇ + 2sin(q)            # typeset as  q̈ + 3q̇ + 2 sin q
```
Without such a function, a dotted name is a symbol of its own.

Single-variable function (NEW):
```julia
@giac_var u(t)        # Creates u as GiacExpr representing "u(t)"
string(u)             # "u(t)"

# Use with differentiation
@giac_var t
diff(u, t)            # Returns diff(u(t),t)
```

Multi-variable function (NEW):
```julia
@giac_var f(x, y)     # Creates f as GiacExpr representing "f(x,y)"
string(f)             # "f(x,y)"
```

Mixed declarations (NEW):
```julia
@giac_var t x y u(t) f(x, y)
# Creates: t, x, y as simple variables
#          u as "u(t)", f as "f(x,y)"
```

# Usage
```julia
using Giac

@giac_var x y
expr = giac_eval("x^2 + y^2")
result = giac_diff(expr, x)  # 2*x

# For ODEs
@giac_var t u(t)
# u''(t) + u(t) = 0

# Callable syntax for initial conditions (034-callable-giacexpr)
@giac_var t u(t) tau U0
u(0)                    # Returns GiacExpr: "u(0)"
u(0) ~ 1                # Initial condition: u(0) = 1
diff(u, t)(0) ~ 0       # Derivative condition: u'(0) = 0
desolve([tau * diff(u, t) + u ~ U0, u(0) ~ 1], u)
```

# See also
- [`giac_eval`](@ref): For evaluating string expressions
"""
# a function's property tag as code: names (`odd`) quoted, calls (`periodic(2π)`) evaluated with
# `π` exact rather than a float
_property_code(tag) = tag === nothing ? nothing : tag isa Symbol ? QuoteNode(tag) :
                      tag isa Expr && tag.head === :tuple ? Expr(:tuple, map(_property_code, tag.args)...) : _exact_pi(tag)
_exact_pi(ex::Symbol) = ex in (:π, :pi) ? :($(GlobalRef(Base, :convert))($GiacExpr, $(Base.MathConstants.π))) : esc(ex)
_exact_pi(ex::Expr) = Expr(ex.head, (ex.head === :call && i == 1 ? esc(a) : _exact_pi(a) for (i, a) in enumerate(ex.args))...)
_exact_pi(ex) = ex

macro giac_var(args...)
    # Validate at least one argument
    if isempty(args)
        throw(ArgumentError("@giac_var requires at least one argument"))
    end

    # `x[1:2]` / `x[3]` declares an array of entries x1, x2, … (see `giac_array`); everything
    # else is a variable or an unevaluated function call
    exprs = Expr[]
    varnames = Symbol[]
    for arg in args
        # `k::positive`, `x[1:2]::real`, `y::(0, 1)`: a declaration with an assumption
        decl, tag = arg isa Expr && arg.head == :(::) && length(arg.args) == 2 ? (arg.args[1], arg.args[2]) : (arg, nothing)
        # `m_c => "m_{\\text{cart}}"`: the name and how it is written
        if decl isa Expr && decl.head == :call && decl.args[1] === :(=>) && length(decl.args) == 3
            decl, tex = decl.args[2], decl.args[3]
            name = decl isa Expr && decl.head == :call ? decl.args[1] : decl
            push!(exprs, :(_set_tex_name!($(string(name)), $(esc(tex)))))
        end
        dotted = decl isa Symbol ? _dotted(string(decl)) : nothing
        if dotted !== nothing
            # `q̇` after `q(t)`: the derivative; otherwise a symbol of its own
            base = Symbol(first(dotted))
            push!(exprs, :($(esc(decl)) = _dotted_variable($(string(decl)), $(esc(:(@isdefined $base))) ? $(esc(base)) : nothing, $(last(dotted)))))
            push!(varnames, decl)
            tagex = tag === nothing ? nothing : tag isa Symbol ? QuoteNode(tag) : esc(tag)
            push!(exprs, :(is_identifier($(esc(decl))) && _declare!($(esc(decl)), $tagex)))
            continue
        end
        if decl isa Expr && decl.head == :ref && decl.args[1] isa Symbol
            push!(exprs, _array_declaration(decl.args[1], decl.args[1], decl.args[2:end]))
            push!(varnames, decl.args[1])
        else
            varname, giac_string = _process_giac_var_arg(decl)
            push!(exprs, :($(esc(varname)) = giac_identifier($giac_string)))
            push!(varnames, varname)
            # a function of other variables, `u(t)`, takes properties of functions instead
            if decl isa Expr
                push!(exprs, :(_declare_function!($(esc(varname)), $(_property_code(tag)))))
                continue
            end
        end
        # every declaration starts the name afresh, without assumptions from earlier code
        tagex = tag === nothing ? nothing : tag isa Symbol ? QuoteNode(tag) : esc(tag)
        push!(exprs, :(_declare!($(esc(varnames[end])), $tagex)))
    end

    # Return tuple of created variables
    result = Expr(:tuple, [esc(v) for v in varnames]...)

    return Expr(:block, exprs..., result)
end

"""
    @giac_several_vars base dims...

Create multiple indexed symbolic variables for N-dimensional tensors.

This macro generates multiple `GiacExpr` variables in the calling scope
with names formed from a base name and indices. It supports any number
of dimensions and returns a tuple of all created variables.

# Arguments
- `base`: Symbol - The base name for variables (e.g., `a`, `coeff`, `α`)
- `dims...`: Integer literals or range expressions - Dimensions of the tensor (1 or more)
  - Integer `n`: Creates indices 1:n (backward compatible)
  - UnitRange `a:b`: Creates indices from a to b
  - StepRange `a:s:b`: Creates indices from a to b with step s

# Returns
- `Tuple{GiacExpr...}`: A tuple containing all created variables in lexicographic order

# Naming Convention
- If all indices are in 0-9: indices are concatenated directly (e.g., `a123`)
- If any index > 9: underscore separators are used (e.g., `a_1_10`)
- Negative indices: `m` prefix for minus (e.g., -1 → `m1`, so `c_m1`)

# Examples

1D vector:
```julia
@giac_several_vars a 3
# Creates: a1, a2, a3 and returns (a1, a2, a3)
a1 + a2 + a3  # Symbolic sum

# Capture return value
vars = @giac_several_vars c 4
length(vars)  # 4
```

2D matrix:
```julia
result = @giac_several_vars m 2 3
# Creates: m11, m12, m13, m21, m22, m23
# Returns: (m11, m12, m13, m21, m22, m23)
length(result)  # 6
```

3D tensor:
```julia
@giac_several_vars t 2 2 2
# Creates: t111, t112, t121, t122, t211, t212, t221, t222
# Returns tuple of 8 variables
```

Large dimensions (separator used):
```julia
@giac_several_vars b 2 10 3
# Creates: b_1_1_1, b_1_1_2, ..., b_2_10_3
```

Unicode base names:
```julia
@giac_several_vars α 2
# Creates: α1, α2
```

0-based indexing with UnitRange:
```julia
@giac_several_vars psi 0:2
# Creates: psi0, psi1, psi2 and returns (psi0, psi1, psi2)
```

2D with custom ranges:
```julia
@giac_several_vars T 0:1 0:2
# Creates: T00, T01, T02, T10, T11, T12
```

Negative indices:
```julia
@giac_several_vars c -1:1
# Creates: c_m1, c_0, c_1 (m = minus, to avoid GIAC parsing issues)
```

StepRange:
```julia
@giac_several_vars q 0:2:4
# Creates: q0, q2, q4
```

Mixed arguments (integer and range):
```julia
@giac_several_vars mixed 2 0:1
# First dim: 1:2, second dim: 0:1
# Creates: mixed10, mixed11, mixed20, mixed21
```

Edge cases:
```julia
@giac_several_vars x 0     # Returns empty tuple ()
@giac_several_vars y 1     # Creates y1, returns (y1,)
@giac_several_vars z 2 0   # Returns empty tuple (0 in any dim)
@giac_several_vars e 5:4   # Returns empty tuple (empty range)
```

# See also
- [`@giac_var`](@ref): For creating single symbolic variables
- [`giac_eval`](@ref): For evaluating string expressions
"""
macro giac_several_vars(base, dims...)
    # Validate base name is a symbol
    if !(base isa Symbol)
        throw(ArgumentError("First argument must be a symbol (base name), got $(typeof(base)): $base"))
    end

    # Validate at least one dimension is provided
    if isempty(dims)
        throw(ArgumentError("At least one dimension required"))
    end

    # Validate all dimensions are integers or range expressions, and convert to ranges
    ranges_list = []
    for d in dims
        if d isa Integer
            # Integer: validate non-negative
            if d < 0
                throw(ArgumentError("Dimensions must be non-negative, got $d"))
            end
            push!(ranges_list, 1:d)
        elseif _is_range_expr(d)
            # Range expression: evaluate at macro expansion time
            push!(ranges_list, _eval_dim_to_range(d))
        else
            throw(ArgumentError("Dimensions must be integer literals or range expressions, got $(typeof(d)): $d"))
        end
    end

    # If any range is empty, generate no variables, return empty tuple
    if any(isempty, ranges_list)
        return :(())
    end

    # Determine if we need underscore separators
    needs_sep = _needs_separator_for_macro_ranges(ranges_list)

    # Generate all index combinations using Iterators.product
    # Reverse ranges so that Iterators.product gives row-major order (last dim varies fastest)
    # Then reverse each tuple back to original dimension order
    ranges = reverse(ranges_list)
    index_combinations = (reverse(t) for t in Iterators.product(ranges...))

    # Generate assignment expressions for each variable
    exprs = Expr[]
    varnames = Symbol[]
    for indices in index_combinations
        varname = _format_indices(base, indices, needs_sep)
        push!(varnames, varname)
        push!(exprs, :($(esc(varname)) = giac_identifier($(string(varname)))))
    end

    # Return tuple of created variables
    result = Expr(:tuple, [esc(v) for v in varnames]...)
    return Expr(:block, exprs..., result)
end
