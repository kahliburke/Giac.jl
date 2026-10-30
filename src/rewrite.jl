# Rewriting expressions into a preferred form.

"""
    express(expr, name => value, ...; eliminate = nothing)

Rewrite `expr` in terms of named quantities, e.g. a matrix's trace and determinant:

```julia
@giac_matrix A[2, 2]
@giac_var τ Δ
express(a11^2 + 2a11*a22 + a22^2, τ => tr(A), Δ => det(A))     # τ^2
```

Each pair says `name = value`, where `value` is a polynomial in other variables. The rewrite is a
Gröbner reduction modulo the relations `value - name`, with a lexicographic order that eliminates
the variables of the values first (in the order given by `eliminate`, if supplied), so their
combinations are traded for the names wherever the relations allow. Numerator and denominator
are reduced separately and the result is factored.

Applies entry by entry to arrays: `express.(M, τ => tr(A), Δ => det(A))`.
"""
function express(expr::GiacExpr, defs::Pair...; eliminate = nothing)
    isempty(defs) && return expr
    names = [convert(GiacExpr, first(d)) for d in defs]
    values = [convert(GiacExpr, last(d)) for d in defs]
    old = eliminate === nothing ? _variables_of(values) : collect(GiacExpr, eliminate)
    order = vcat(old, names)
    basis = Commands.invoke_cmd(:gbasis, GiacExpr[v - n for (n, v) in zip(names, values)], order, giac_eval("plex"))
    reduce(p) = is_constant(p) ? p : Commands.invoke_cmd(:greduce, p, basis, order)
    rational = Commands.normal(expr)
    num, den = Commands.invoke_cmd(:numer, rational), Commands.invoke_cmd(:denom, rational)
    return Commands.factor(reduce(num) / reduce(den))
end

express(x::Number, defs::Pair...; kwargs...) = convert(GiacExpr, x)

"The distinct free variables of `exprs`, in order of first appearance."
function _variables_of(exprs)
    seen = GiacExpr[]
    for e in exprs, v in free_symbols(e)
        any(isequal(v), seen) || push!(seen, v)
    end
    return seen
end
