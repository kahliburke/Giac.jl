# build_function: convert a GiacExpr into a native Julia callable
# (066-build-function — Tier 1 wrapper over substitute + to_julia)

# Internal callable wrapper. The fixed-arity Vararg{Any, N} method on
# _BuildFunction{N} guarantees that calls with the wrong number of arguments
# fall through Julia's dispatcher as a real MethodError. Users never reference
# this type directly — they only see the value as a `<: Function` callable.
struct _BuildFunction{N} <: Function
    expr::GiacExpr
    vars::NTuple{N, GiacExpr}
end

(b::_BuildFunction{0})() = to_julia(b.expr)

(b::_BuildFunction{N})(vals::Vararg{Any, N}) where {N} =
    to_julia(b.expr(Pair.(b.vars, vals)...))

Base.show(io::IO, b::_BuildFunction{N}) where {N} =
    print(io, "build_function callable for ", b.expr, " over ", N, " variable(s)")

# Hook for the GiacSymbolicsExt extension. Declared here as a function with
# no methods; the extension supplies the only method when `using Symbolics`
# loads. Avoids the precompile-time "method overwriting" error that would
# occur if a default body lived here and the extension shadowed it.
function _build_function_symbolics_impl end

"""
    build_function(expr::GiacExpr, vars::GiacExpr...; backend::Symbol = :giac) -> Function

Wrap a symbolic `GiacExpr` into a native Julia callable.

The `backend` keyword selects the evaluation engine:
- `backend = :giac` (default, always available): each call runs through GIAC's
  substitution + `to_julia` pipeline. The full behavior is documented below.
- `backend = :symbolics` (requires `using Symbolics`): the expression is
  round-tripped through `Giac.to_symbolics` and compiled into a native Julia
  function via `Symbolics.build_function`. Faster in hot loops and compatible
  with `ForwardDiff` / SciML autodiff. See the "Backends" section below for
  trade-offs and edge cases that differ from `:giac`.

The default `:giac` backend is documented in detail in this docstring; the
`:symbolics` backend is documented in `docs/src/julia_functions.md`.

The returned callable `f` satisfies the equivalence

```julia
f(vals...) ≡ to_julia(expr(Pair.((vars...,), (vals...,))...))
            ≡ to_julia(substitute(expr, vars[1] => vals[1], …, vars[N] => vals[N]))
```

So `build_function` is a thin convenience wrapper. It does not introduce any
new substitution or evaluation mechanism — it just gives the
`substitute` + `to_julia` chain a single named entry point that is suitable as
a drop-in argument to `Plots.plot`, `Plots.surface`, broadcasting (`f.(xs)`),
and matrix comprehensions.

# Arguments
- `expr::GiacExpr`: the symbolic expression to wrap.
- `vars::GiacExpr...`: the symbolic variables to bind, in positional order.

# Returns
A Julia callable (`<: Function`) of fixed arity `length(vars)` that, when
called, returns whatever `to_julia(...)` produces — typically `Bool`,
`Int64`, `BigInt`, `Float64`, `Rational`, `Complex`, `Vector`, `String`, or
`GiacExpr` (see [`to_julia`](@ref)). Calling with the wrong number of
arguments raises `MethodError` (Julia dispatch).

# Examples
```jldoctest
julia> @giac_var x;

julia> f = build_function(x^2 - 1, x);

julia> f(3)
8

julia> f(0)
-1
```

```jldoctest
julia> @giac_var x y;

julia> g = build_function(x + 2*y, x, y);

julia> g(1, 2)
5
```

# Equivalent manual form

`build_function` is a *thin convenience wrapper*. The same result is always
recoverable from the underlying primitives:

```julia
@giac_var x y
expr = x + 2*y

# Either of these gives identical values:
f1 = build_function(expr, x, y);             f1(1, 2)
f2(_x, _y) = to_julia(substitute(expr, x => _x, y => _y));  f2(1, 2)
```

If you need to insert a transformation between substitution and Julia
conversion (e.g., simplify, expand, evalf with custom precision), use
`substitute` and `to_julia` directly rather than expecting `build_function` to
grow new keyword arguments for every variant.

# Edge cases
- **Constant expression (zero `vars`).** `build_function(expr)()` returns
  `to_julia(expr)`. Symbolic constants such as `π` are reduced via
  `Giac.Commands.evalf` automatically (see [`to_julia`](@ref)).
- **Extra (unused) variables.** `build_function(x^2, x, y)(3, 100)` returns
  `9`; the unused `y` is silently ignored, matching `substitute`'s semantics.
- **Missing variables.** `build_function(x*y, x)(2)` returns a `GiacExpr`
  containing the still-free `y` (no error). To get a numeric value you must
  bind every free variable.
- **Mixed numeric input types.** `Int`, `Float64`, `Rational`, `BigInt`
  (and any value `substitute` accepts) flow through unchanged — whatever
  `substitute` + `to_julia` produces is what you get.
- **Argument-count mismatch.** Calling the returned callable with the wrong
  number of positional arguments raises `MethodError` via Julia dispatch.

# Backend `:symbolics`

When `using Symbolics` is in scope, `backend = :symbolics` round-trips the
expression through [`to_symbolics`](@ref) and compiles it via
`Symbolics.build_function`. The returned callable is plain Julia code, so it
runs much faster in hot loops and composes with `ForwardDiff`, SciML solvers,
and other autodiff-aware consumers that the default `:giac` backend cannot.

```julia
using Giac, Symbolics

@giac_var x
f = Giac.build_function(x^3 - 2x + 1, x; backend = :symbolics)
f(2.0)   # 5.0
```

Trade-offs:
- Restricted to expressions whose heads `to_symbolics` can translate. An
  unsupported head raises an error at `build_function` time naming the head.
- Free symbols in `expr` that are not in `vars` raise an `ArgumentError` at
  build time (the `:giac` backend, by contrast, returns a residual `GiacExpr`).
- Argument-count mismatch on the returned callable raises `BoundsError` for
  too-few args and silently ignores extra args (Symbolics' `RuntimeGeneratedFunction`
  semantics; differs from `:giac`'s `MethodError`).
- Without `using Symbolics`, `backend = :symbolics` raises an `ArgumentError`
  pointing the user to install Symbolics or use the default `:giac` backend.

**Naming caveat**: `Symbolics` also exports `build_function`. With both
`using Giac` and `using Symbolics` in scope, the bare name is ambiguous —
write `Giac.build_function(...)` (or `Symbolics.build_function(...)` for
the Symbolics-specific call sites).

See `docs/src/julia_functions.md` for the full backend comparison and a
runtime benchmark.

# See also
- [`substitute`](@ref) — the underlying simultaneous-substitution primitive.
- [`to_julia`](@ref) — the Giac-to-Julia value converter (with auto-`evalf`).
- [`to_symbolics`](@ref) — the bridge used by the `:symbolics` backend.
- [`@giac_var`](@ref) — declare symbolic variables.
- `Symbolics.build_function` — the engine the `:symbolics` backend delegates to.
- `SymPy.lambdify` — the SymPy.jl analogue for users coming from SymPy.
"""
function build_function(expr::GiacExpr, vars::GiacExpr...; backend::Symbol = :giac)
    if backend === :giac
        return _BuildFunction(expr, vars)
    elseif backend === :julia
        return _compile(expr, vars)                    # src/codegen.jl
    elseif backend === :symbolics
        # If GiacSymbolicsExt hasn't loaded, no method is defined and we
        # surface a clear error pointing at Symbolics. Otherwise dispatch
        # to the extension's method.
        if hasmethod(_build_function_symbolics_impl,
                     Tuple{GiacExpr, Tuple{Vararg{GiacExpr}}})
            return _build_function_symbolics_impl(expr, vars)
        end
        throw(ArgumentError(
            "backend = :symbolics requires `using Symbolics`; install with " *
            "Pkg.add(\"Symbolics\") and `using Symbolics` before calling " *
            "build_function. Or use the default backend = :giac."))
    else
        throw(ArgumentError(
            "unknown backend $backend; expected :giac, :julia or :symbolics"))
    end
end
