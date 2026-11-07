# GIAC's cSolve gives the roots of a polynomial of degree above 4 as decimals, and so the solutions
# of a polynomial system that needs one. Where the equations are exact they are solved exactly
# instead. A system in shape position (its lexicographic Gröbner basis is one polynomial in the
# last variable and, for each other variable, that variable as a polynomial in the last) comes down
# to one variable. A polynomial in xᵏ is solved in xᵏ and the kth roots are taken; other roots are
# Nemo's exact algebraic numbers (algebraic_roots), so this needs Nemo.

const _PLAIN_CSOLVE = :giac_plain_csolve

# `invoke_cmd(:cSolve, args...)`
function _csolve_cmd(args...)
    r = task_local_storage(() -> invoke_cmd(:cSolve, args...), _PLAIN_CSOLVE, true)
    (_has_decimal(r) && !isempty(methods(algebraic_roots))) || return r
    v = try
        _exact_csolve(args...)
    catch err
        err isa Union{ArgumentError, GiacError, ErrorException} || rethrow()
        nothing
    end
    return v === nothing ? r : v
end
_plain_csolve() = get(task_local_storage(), _PLAIN_CSOLVE, false)

# whether a value, or an entry of a (nested) list, holds a decimal
_has_decimal(e::GiacExpr) =
    giac_type(e) == GenTypes.VECT ? any(i -> _has_decimal(e[i]), 1:length(e)) : _has_float(_node(e))

_as_polynomial(e::GiacExpr) = _is_equation(e) ? lhs(e) - rhs(e) : e
_as_vector(a::AbstractVector) = collect(GiacExpr, a)
_as_vector(a) = GiacExpr[convert(GiacExpr, a)]

function _exact_csolve(eqs, vars)
    ps = _as_polynomial.(_as_vector(eqs))
    vs = _as_vector(vars)
    any(p -> _has_float(_node(p)), ps) && return nothing
    if length(vs) == 1 && length(ps) == 1
        roots = _exact_polynomial_roots(ps[1], vs[1])
        return roots === nothing ? nothing : invoke_cmd(:eval, roots)
    end
    sols = _shape_position_solutions(ps, vs)
    return sols === nothing ? nothing : invoke_cmd(:eval, sols)
end

# the distinct roots of the polynomial `p` in `x`, exactly, or nothing when its coefficients are
# not rational
function _exact_polynomial_roots(p::GiacExpr, x::GiacExpr)
    c = try
        _rational_coefficients(p, x)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    # a polynomial in xᵏ: the kth roots of the roots in xᵏ
    k = reduce(gcd, (j - 1 for j in 2:length(c) if !iszero(c[j])); init = 0)
    if k > 1
        u = _fresh_symbol(_node(p), "u")
        inner = _exact_polynomial_roots(_polynomial_in(c[1:k:end], u), u)
        inner === nothing && return nothing
        out = GiacExpr[]
        for r in inner
            if k == 2
                push!(out, sqrt(r), -sqrt(r))
            else
                ρ = r^(1 // k)
                append!(out, (invoke_cmd(:simplify, ρ * exp(2j * giac_eval("i") * giac_eval("pi") / k)) for j in 0:(k - 1)))
            end
        end
        return out
    end
    # GIAC's own answer where it is exact (its rootof forms of a cubic's roots are compact)
    plain = task_local_storage(() -> invoke_cmd(:cSolve, p, x), _PLAIN_CSOLVE, true)
    _has_decimal(plain) || return [plain[i] for i in 1:length(plain)]
    return unique(algebraic_roots(p, x))
end

# the solutions of polynomial equations `ps` in `vs` (one list of values per solution, in the
# order of `vs`) when a lexicographic Gröbner basis in some order of the variables is in shape
# position, of the orders tried the one whose other variables are the shortest polynomials in the
# last; nothing otherwise
function _shape_position_solutions(ps, vs)
    shapes = filter(!isnothing, [_shape_position(ps, order) for order in unique([vs, reverse(vs)])])
    isempty(shapes) && return nothing
    uni, last_, forms = argmin(((u, l, f),) -> sum(length(string(e)) for e in values(f); init = 0), shapes)
    roots = _exact_polynomial_roots(uni, last_)
    roots === nothing && return nothing
    return [GiacExpr[isequal(v, last_) ? r : substitute(forms[string(v)], last_ => r) for v in vs] for r in roots]
end

# (the univariate polynomial, its variable, each other variable as a polynomial in it) for the
# lexicographic Gröbner basis of `ps` in `order`, when that is in shape position; nothing otherwise
function _shape_position(ps, order)
    G = invoke_cmd(:gbasis, ps, order, giac_eval("plex"))
    basis = [G[i] for i in 1:length(G)]
    length(basis) == length(order) || return nothing
    names = [string(v) for v in order]
    uni = findfirst(g -> _free_names(g) ⊆ Set([names[end]]), basis)
    uni === nothing && return nothing
    # each other variable v as c v + f(last), c a number
    forms = Dict{String, GiacExpr}()
    for (j, g) in enumerate(basis)
        j == uni && continue
        vars_in = intersect(_free_names(g), Set(names[1:(end - 1)]))
        length(vars_in) == 1 || return nothing
        v = giac_identifier(only(vars_in))
        d = invoke_cmd(:diff, g, v)
        isempty(_free_names(d)) || return nothing
        forms[only(vars_in)] = -substitute(g, v => 0) / d
    end
    length(forms) == length(order) - 1 || return nothing
    return basis[uni], order[end], forms
end
