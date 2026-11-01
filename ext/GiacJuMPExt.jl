# ---------------------------------------------------------------------------
# GiacJuMPExt: choosing among a decomposition's solutions by linear programming.
#
# Loaded with JuMP, HiGHS and Nemo. `certify` looks for weights of a Decomposition's free
# directions that make every multiplier's coefficients (after the gap substitution of an
# ordering) of the wanted sign. HiGHS solves in floating point; the weights are then made
# rational and the result is checked exactly in Nemo, so a returned certificate does not rest on
# the floating-point solve.
# ---------------------------------------------------------------------------

module GiacJuMPExt

using Giac
using Nemo
using JuMP
import HiGHS

function Giac.certify(d::Giac.Decomposition; order = (), positive = (), param = (), signs = 1, bound = 1.0e6)
    nb, nd = length(d.blocks), length(d.directions)
    s = signs isa Number ? fill(Int(signs), nb) : collect(Int, signs)
    length(s) == nb || throw(ArgumentError("certify: one sign per block"))
    wanted = copy(s)
    gaps, known = Giac._gap_substitution(order, positive)
    prepared(e) = Giac.Commands.normal(_substituted(_substituted(e, param), gaps))
    exprs = GiacExpr[]
    for k in 1:nb
        push!(exprs, prepared(d.multipliers[k]))
        foreach(j -> push!(exprs, prepared(d.directions[j][k])), 1:nd)
    end
    values = Giac.to_nemo(exprs)
    # per block: the numerators over a common denominator of known sign
    rows = Vector{Vector{Dict{Vector{Int}, QQFieldElem}}}()      # rows[k][1 + j]: monomial => coefficient
    for k in 1:nb
        members = values[(k - 1) * (nd + 1) .+ (1:(nd + 1))]
        nums, σ = _common_numerators(members)
        σ == 0 && return nothing                                  # a denominator of unknown sign
        s[k] *= σ
        all(p -> _known_variables(p, known), nums) || return nothing
        push!(rows, [Dict(collect(e) => c for (c, e) in zip(coefficients(p), exponent_vectors(p))) for p in nums])
    end
    λ = nd == 0 ? Rational{BigInt}[] : _lp_weights(rows, s, nd, bound)
    λ === nothing && return nothing
    # small denominators first: the simplest weights that keep the signs
    for tol in (1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 0.0)
        q = Rational{BigInt}[rationalize(BigInt, Float64(v); tol) for v in λ]
        if _signs_hold(rows, s, q)
            multipliers = Giac.combination(d, q)
            checks = [positivity(wanted[k] * multipliers[k]; order, positive, param) for k in 1:nb]
            return (λ = q, multipliers, checks)
        end
    end
    return nothing
end

_substituted(e, pairs) = isempty(pairs) ? e : substitute(e, pairs...)

# numerators of the members over their common denominator, and that denominator's sign by its
# coefficients (0 when mixed)
function _common_numerators(members)
    first(members) isa Nemo.FracElem || return members, 1
    L = reduce(lcm, (denominator(m) for m in members))
    cs = collect(coefficients(L))
    σ = all(>(0), cs) ? 1 : all(<(0), cs) ? -1 : 0
    return [divexact(numerator(m) * L, denominator(m)) for m in members], σ
end

# whether every variable of p is known positive
_known_variables(p, known) = all(((v, g),) -> degree(p, g) == 0 || string(v) in known || Giac._known_nonnegative(Giac.SymNode(string(v))),
                                 zip(symbols(parent(p)), gens(parent(p))))

# weights λ maximizing the smallest signed coefficient, in floating point
function _lp_weights(rows, s, nd, bound)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, -bound <= λ[1:nd] <= bound)
    @variable(model, t <= 1)
    for (k, members) in enumerate(rows)
        monomials = union(keys.(members)...)
        for m in monomials
            c0 = Float64(get(members[1], m, QQ(0)))
            cs = [Float64(get(members[1 + j], m, QQ(0))) for j in 1:nd]
            @constraint(model, s[k] * (c0 + sum(cs[j] * λ[j] for j in 1:nd)) >= t)
        end
    end
    @objective(model, Max, t)
    optimize!(model)
    (is_solved_and_feasible(model) && value(t) >= -1.0e-9) || return nothing
    # then the smallest weights keeping half that margin, which rationalize simply
    margin = value(t)
    @variable(model, w[1:nd] >= 0)
    @constraint(model, [j in 1:nd], w[j] >= λ[j])
    @constraint(model, [j in 1:nd], w[j] >= -λ[j])
    @constraint(model, t >= (margin > 0 ? margin / 2 : margin))
    @objective(model, Min, sum(w))
    optimize!(model)
    is_solved_and_feasible(model) || return nothing
    return value.(λ)
end

# whether the multipliers at the rational weights q have coefficients of the wanted signs, and
# none is zero
function _signs_hold(rows, s, q)
    for (k, members) in enumerate(rows)
        coeffs = Dict{Vector{Int}, QQFieldElem}()
        for (j, member) in enumerate(members), (m, c) in member
            w = j == 1 ? QQ(1) : QQ(numerator(q[j - 1]), denominator(q[j - 1]))
            coeffs[m] = get(coeffs, m, QQ(0)) + w * c
        end
        nonzero = filter(!iszero, collect(values(coeffs)))
        (isempty(nonzero) || all(c -> s[k] * c > 0, nonzero)) || return false
    end
    return true
end

end # module GiacJuMPExt
