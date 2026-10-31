# Rewriting, simplifying and comparing symbolic matrices (`@matsym`): through their coefficients,
# the scalars multiplying each product of matrices.

using ..Giac: MatExpr, map_coefficients

"""
    simplest(M::MatExpr, transforms...; kwargs...)
    rewrite(M::MatExpr, rules...; kwargs...)

A symbolic matrix with each coefficient simplified (or rewritten), as for a scalar:
`simplest((sin(t)^2 + cos(t)^2) * A)` is `A`. Terms whose coefficients become equal-looking
combine.
"""
function simplest(M::MatExpr, transforms...; trace::Bool = false, depth::Integer = 4, width::Integer = 6, kwargs...)
    trace && throw(ArgumentError("simplest(…; trace = true) is for a single scalar expression"))
    ids = isempty(transforms) ? matrix_identities() : [t for t in transforms if t isa MatrixIdentity]
    scalar = isempty(transforms) ? () : Tuple(t for t in transforms if !(t isa MatrixIdentity))
    M = _search_identities(M, ids, depth, width)
    return map_coefficients(c -> simplest(c, scalar...; kwargs...), M)
end

# a symbolic matrix's size for the search: its terms and factors, an inverse counting one more,
# and a sum under an inverse counted in full
_matrix_cost(M::MatExpr) = sum((1 + sum(_factor_cost, w; init = 0) for (w, _) in M.terms); init = 0)
_factor_cost(f) = 1 + (f.power < 0 ? 1 : 0) + (f.base isa MatExpr ? _matrix_cost(f.base) : 0)

# beam search over the matrix identities (each applied where it first fits), for the expression of
# least cost; identities that lengthen (Woodbury) are kept only where they lead somewhere shorter
function _search_identities(M::MatExpr, ids, depth, width)
    isempty(ids) && return M
    score(X) = (_matrix_cost(X), length(string(X)))
    best, best_score = M, score(M)
    seen = Set([string(M)])
    frontier = [M]
    for _ in 1:depth
        next = MatExpr[]
        for X in frontier, id in ids
            Y = try id(X) catch; nothing end
            Y isa MatExpr || continue
            k = string(Y)
            k in seen && continue
            push!(seen, k)
            push!(next, Y)
        end
        isempty(next) && break
        sort!(next; by = score)
        frontier = next[1:min(width, end)]
        score(frontier[1]) < best_score && ((best, best_score) = (frontier[1], score(frontier[1])))
    end
    return best
end
simplest(X::AbstractArray{MatExpr}, transforms...; kwargs...) = map(M -> simplest(M, transforms...; kwargs...), X)

function rewrite(M::MatExpr, rules...; trace = false, maxpasses::Integer = 100, kwargs...)
    trace === false || throw(ArgumentError("rewrite(…; trace) is for a single scalar expression"))
    ids = [r for r in rules if r isa MatrixIdentity]
    scalar = Tuple(r for r in rules if !(r isa MatrixIdentity))
    seen = Set([string(M)])
    for _ in 1:maxpasses
        i = findfirst(id -> id(M) isa MatExpr, ids)
        i === nothing && break
        M = ids[i](M)
        string(M) in seen && throw(ErrorException("the matrix identities return to $M"))
        push!(seen, string(M))
    end
    return isempty(scalar) ? M : map_coefficients(c -> rewrite(c, scalar...; kwargs...), M)
end
rewrite(X::AbstractArray{MatExpr}, rules...; kwargs...) = map(M -> rewrite(M, rules...; kwargs...), X)

apply_relations(M::MatExpr) = map_coefficients(apply_relations, M)

"""
    equivalent(A::MatExpr, B::MatExpr)

Whether two symbolic matrices are equal: of the same size, with every coefficient of `A - B`
[`equivalent`](@ref) to 0.
"""
equivalent(A::MatExpr, B::MatExpr; kwargs...) =
    size(A) == size(B) && all(c -> equivalent(c, 0; kwargs...), last.((A - B).terms))
