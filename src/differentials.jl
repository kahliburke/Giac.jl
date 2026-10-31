# The differential, of a scalar in its variables (plain ones, coordinates q(t) and their derivatives,
# matrix symbols through traces and determinants) and of a matrix expression in its matrix symbols.
# The scalar side is built on `partial` (coordinate_diff.jl), the matrix side on the product rule
# (matrix_calculus.jl).

"""
    differential(f, x, y, …)
    differential(f)
    differential(x)

The differential of `f` in the variables given, or in all of its variables when none are:

- a scalar in scalar variables is `∂f/∂x dx + ∂f/∂y dy + …`, each `dx` a symbol written as a
  differential. A variable may be a function of time `q(t)` or one of its derivatives; each
  coefficient is a [`partial`](@ref), so the coefficient of `dt` is the explicit dependence on
  time. The terms are written in the order the variables were given.
- a matrix expression, or a scalar made of traces and determinants, in matrix symbols declared
  with `@matsym`: `d(XY) = dX Y + X dY`, `d(X⁻¹) = -X⁻¹ dX X⁻¹`, `d tr W = tr dW`,
  `d det M = det M tr(M⁻¹ dM)`.

Scalar variables and matrix symbols can be mixed. `differential(x)` of a variable alone is its
`dx`, so a coefficient reads off as `diff(df, differential(x))`.

```julia
@giac_var q q̇ p t L(x, v, s)
dH = differential(p * q̇ - L(q, q̇, t), q, q̇, p, t)
# -∂L/∂q dq + (p - ∂L/∂q̇) dq̇ + q̇ dp - ∂L/∂t dt
diff(dH, differential(p))     # q̇

@matsym X[n, n] A[n, n] x[n]
differential(inv(X))          # -X⁻¹ dX X⁻¹
differential(tr(A*X), X)      # tr(A dX)
differential(x'*A*x, x)       # dxᵀAx + xᵀA dx
```

[`∇`](@ref), `jacobian`, `hessian` and [`derivative`](@ref) are read off the differential.
"""
differential(M::MatExpr, symbols::MatExpr...) = _differential(M, _vary_set(symbols))

function differential(f::GiacExpr, vars::Union{GiacExpr, MatExpr}...)
    if isempty(vars)
        # a variable alone is its differential, dx
        name = _wrt_name(f)
        name === nothing || _is_matrix_atom(f) || return giac_identifier("d" * name)
        return _total_differential(f, _variables_of(f), nothing)
    end
    scalars = GiacExpr[v for v in vars if v isa GiacExpr]
    matrices = [v for v in vars if v isa MatExpr]
    return _total_differential(f, scalars, isempty(matrices) ? Set{Symbol}() : _vary_set(matrices))
end

# ∂f/∂x dx summed over the scalar variables, written in their order, and the matrix part over
# `vary` (nothing: every matrix symbol; an empty set: none)
function _total_differential(f::GiacExpr, scalars::Vector{GiacExpr}, vary)
    dxs = [differential(v) for v in scalars]
    lock(_DIFFERENTIAL_LOCK) do
        foreach(((k, dx),) -> _DIFFERENTIAL_ORDER[string(dx)] = k, enumerate(dxs))
    end
    d = isempty(scalars) ? zero(GiacExpr) : sum(partial(f, v) * dx for (v, dx) in zip(scalars, dxs))
    (vary === nothing || !isempty(vary)) && (d += _differential(f, vary))
    return d
end

_is_matrix_atom(x::GiacExpr) = is_identifier(x) && lock(() -> haskey(_ATOMS, string(x)), _ATOMS_LOCK)

# The scalar variables of `f`: its coordinates (q(t), q̇), then the plain symbols outside them, not
# counting the names of functions, named constants or traces and determinants of matrices
function _variables_of(f::GiacExpr)
    n = _node(f)
    vars = GiacExpr[GiacExpr(c) for c in _coordinates!(Node[], n)]
    names = String[]
    _plain_variables!(names, n)
    sort!(names; by = s -> (_declared_order(s), s))
    append!(vars, giac_identifier.(names))
    return vars
end

_plain_variables!(out, n::Node) = out
function _plain_variables!(out, n::SymNode)
    s = n.name
    (s in _GIAC_NAMED_VALUES || s in out || lock(() -> haskey(_ATOMS, s), _ATOMS_LOCK)) || push!(out, s)
    return out
end
function _plain_variables!(out, n::CallNode)
    _is_coordinate(n) && return out
    foreach(a -> _plain_variables!(out, a), n.f == "of" ? n.args[2:end] : n.args)
    return out
end
_plain_variables!(out, n::AddNode) = (foreach(t -> _plain_variables!(out, first(t)), n.terms); out)
_plain_variables!(out, n::MulNode) = (foreach(((b, e),) -> (_plain_variables!(out, b); _plain_variables!(out, e)), n.factors); out)
