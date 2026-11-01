# Rewriting expressions into a preferred form.

"""
    express(expr, name => value, ...; eliminate = nothing)

Rewrite `expr` in terms of named quantities, e.g. a matrix's trace and determinant:

```julia
@giac_matrix A[2, 2]
@giac_var τ Δ
express(a11^2 + 2a11*a22 + a22^2, τ => tr(A), Δ => det(A))     # τ^2
```

Each pair says `name = value`, where `value` is a polynomial in other variables. When every value
is affine in its variables (`L => x₂ - x₁`, `p => y - x₁`), the definitions are a change of
variables: they are solved for as many of the old variables as there are names, keeping the ones
shared by most definitions (`x₁` here), and substituted. Otherwise the rewrite is a Gröbner
reduction modulo the relations `value - name`, with a lexicographic order that eliminates the
variables of the values first (in the order given by `eliminate`, if supplied), so their
combinations are traded for the names wherever the relations allow. Numerator and denominator
are reduced separately and the result is factored.

A Gröbner reduction can take long; `timeout = seconds` stops it with a [`GiacTimeout`](@ref).

Applies entry by entry to arrays: `express.(M, τ => tr(A), Δ => det(A))`.
"""
function express(expr::GiacExpr, defs::Pair...; eliminate = nothing, timeout = nothing)
    isempty(defs) && return expr
    names = [convert(GiacExpr, first(d)) for d in defs]
    values = [convert(GiacExpr, last(d)) for d in defs]
    old = eliminate === nothing ? _variables_of(values) : collect(GiacExpr, eliminate)
    solved = _affine_change(names, values, old)
    solved === nothing || return Commands.factor(Commands.normal(substitute(expr, solved...)))
    order = vcat(old, names)
    function reduced()
        basis = Commands.invoke_cmd(:gbasis, GiacExpr[v - n for (n, v) in zip(names, values)], order, giac_eval("plex"))
        reduce(p) = is_constant(p) ? p : Commands.invoke_cmd(:greduce, p, basis, order)
        rational = Commands.normal(expr)
        num, den = Commands.invoke_cmd(:numer, rational), Commands.invoke_cmd(:denom, rational)
        return Commands.factor(reduce(num) / reduce(den))
    end
    return timeout === nothing ? reduced() : with_timeout(reduced, timeout)
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

# The old variables in terms of the names, when each value is affine in the old variables with
# rational coefficients and the values are independent: `x₂ => L + x₁`, … as pairs, or `nothing`.
# Each definition is solved for the variable that appears in the fewest definitions, so the
# variables shared by many (a common origin) are the ones kept.
function _affine_change(names, values, old)
    oldnames = [string(v) for v in old]
    k, n = length(values), length(oldnames)
    k <= n || return nothing
    A = zeros(Rational{BigInt}, k, n + k + 1)              # [old | names | constant] = 0
    for (i, v) in enumerate(values)
        row = _affine_row(_expand(_node(v)), oldnames)
        row === nothing && return nothing
        A[i, 1:n] .= row[1:n]
        A[i, n + i] = -1
        A[i, end] = row[end]
    end
    uses = [count(!iszero, A[:, j]) for j in 1:n]
    pivots = Int[]
    for i in 1:k
        candidates = [j for j in 1:n if !iszero(A[i, j]) && !(j in pivots)]
        isempty(candidates) && return nothing                 # dependent definitions
        j = argmin(c -> (uses[c], c), candidates)
        A[i, :] ./= A[i, j]
        for r in 1:k
            r != i && !iszero(A[r, j]) && (A[r, :] .-= A[r, j] .* A[i, :])
        end
        push!(pivots, j)
    end
    columns = vcat(convert.(GiacExpr, old), convert.(GiacExpr, names))
    return [old[j] => -sum((A[i, c] * columns[c] for c in eachindex(columns) if c != j && !iszero(A[i, c])); init = zero(GiacExpr)) - A[i, end]
            for (i, j) in enumerate(pivots)]
end

# The coefficients of an expanded affine expression in `oldnames`, then its constant; `nothing`
# if it is not affine in them with rational coefficients
function _affine_row(n::Node, oldnames)
    row = zeros(Rational{BigInt}, length(oldnames) + 1)
    terms = n isa AddNode ? n.terms : n isa MulNode ? [_mul(MulNode(1, n.factors)) => n.coef] : [n => 1]
    for (m, c) in terms
        c isa Union{Integer, Rational} || return nothing
        if m isa NumNode
            row[end] += c * m.v
        elseif m isa SymNode && (j = findfirst(==(m.name), oldnames)) !== nothing
            row[j] += c
        else
            return nothing
        end
    end
    return row
end
