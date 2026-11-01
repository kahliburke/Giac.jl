# ---------------------------------------------------------------------------
# GiacClarabelExt: sums of squares, found by semidefinite programming and checked exactly.
#
# Loaded with JuMP, Clarabel and Nemo. A polynomial p is a sum of squares when p = mᵀ Q m for a
# vector m of monomials and a positive semidefinite Gram matrix Q. Clarabel finds a Q well inside
# the cone; its entries are rounded to rationals, moved by the least amount that makes mᵀ Q m
# equal p exactly, and factored Q = L D Lᵀ over the rationals. When every pivot is nonnegative,
# p = Σ dₖ (Lₖᵀ m)² exactly; the identity is checked in Nemo.
# ---------------------------------------------------------------------------

module GiacClarabelExt

using Giac
using Giac: GiacExpr
using Nemo
using JuMP
import Clarabel
using LinearAlgebra: I

const _Exp = Vector{Int}

# ---------------------------------------------------------------------------------------------
# Monomials and Gram matrices
# ---------------------------------------------------------------------------------------------

# The monomials that can occur in the squares of a polynomial with support `support`: half its
# exponents' range in each variable and in total degree, less those whose square cannot appear
# (2m outside the support, and not a sum of two other candidates), removed until none is left.
function _half_basis(support::Vector{_Exp}, n::Int; mindeg = nothing, maxdeg = nothing)
    isempty(support) && return _Exp[]
    lo = [minimum(e[i] for e in support) for i in 1:n]
    hi = [maximum(e[i] for e in support) for i in 1:n]
    dlo = mindeg === nothing ? minimum(sum, support) : mindeg
    dhi = maxdeg === nothing ? maximum(sum, support) : maxdeg
    ranges = [cld(lo[i], 2):fld(hi[i], 2) for i in 1:n]
    basis = _Exp[collect(m) for m in Iterators.product(ranges...) if dlo <= 2sum(m) <= dhi]
    inside = Set(support)
    while true
        sums = Dict{_Exp, Int}()
        for a in eachindex(basis), b in (a + 1):length(basis)
            s = basis[a] .+ basis[b]
            sums[s] = get(sums, s, 0) + 1
        end
        keep = [m for m in basis if (2 .* m) in inside || haskey(sums, 2 .* m)]
        length(keep) == length(basis) && return basis
        basis = keep
    end
end

# The pairs (a ≤ b) of basis monomials, by the exponent of their product
function _classes(basis)
    classes = Dict{_Exp, Vector{Tuple{Int, Int}}}()
    for a in eachindex(basis), b in a:length(basis)
        push!(get!(classes, basis[a] .+ basis[b], Tuple{Int, Int}[]), (a, b))
    end
    return classes
end

_weight(a, b) = a == b ? 1 : 2

# Constraints making mᵀ Q m equal the polynomial with coefficients `coeffs` (exponent => number
# or affine expression); false when the polynomial has a term no product of the basis reaches.
function _match!(model, Q, classes, coeffs)
    for (α, c) in coeffs
        haskey(classes, α) && continue
        c isa Number ? (iszero(c) || return false) : @constraint(model, c == 0)
    end
    for (α, pairs) in classes
        @constraint(model, sum(_weight(a, b) * Q[a, b] for (a, b) in pairs) == get(coeffs, α, 0))
    end
    return true
end

# ---------------------------------------------------------------------------------------------
# Exact rounding
# ---------------------------------------------------------------------------------------------

# Q rounded to rationals with error below 2^-bits, then moved to satisfy the coefficient
# constraints exactly: in each class, the residual is spread evenly over the entries, which is
# the least change in the Frobenius norm.
function _rounded(Qf, classes, coeffs::Dict{_Exp, Rational{BigInt}}, bits)
    N = size(Qf, 1)
    Q = zeros(Rational{BigInt}, N, N)
    for a in 1:N, b in a:N
        Q[a, b] = Q[b, a] = rationalize(BigInt, Qf[a, b]; tol = 2.0^-bits)
    end
    for (α, pairs) in classes
        total = sum(_weight(a, b) for (a, b) in pairs)
        r = get(coeffs, α, zero(Rational{BigInt})) - sum(_weight(a, b) * Q[a, b] for (a, b) in pairs)
        iszero(r) && continue
        for (a, b) in pairs
            Q[a, b] += r // total
            Q[b, a] = Q[a, b]
        end
    end
    return Q
end

# Q = L D Lᵀ over the rationals: the pivots and the columns of L, or nothing when Q is not
# positive semidefinite
function _ldl(Q::Matrix{Rational{BigInt}})
    A = copy(Q)
    N = size(A, 1)
    D = Rational{BigInt}[]
    L = Vector{Vector{Rational{BigInt}}}()
    for k in 1:N
        d = A[k, k]
        d < 0 && return nothing
        if iszero(d)
            all(iszero, @view A[k, (k + 1):N]) || return nothing
            continue
        end
        col = [j < k ? zero(d) : A[j, k] / d for j in 1:N]
        for j in (k + 1):N, l in (k + 1):N
            A[j, l] -= col[j] * d * col[l]
        end
        push!(D, d)
        push!(L, col)
    end
    return D, L
end

_coeffs(P) = Dict{_Exp, Rational{BigInt}}(collect(e) => Rational{BigInt}(c) for (c, e) in zip(coefficients(P), exponent_vectors(P)))

function _poly(R, terms)
    ctx = MPolyBuildCtx(R)
    for (c, e) in terms
        iszero(c) || push_term!(ctx, QQ(c), e)
    end
    return finish(ctx)
end

# An exact certificate from a floating-point Gram matrix, trying finer roundings in turn
function _certificate(P, basis, classes, Qf, maxbits, label)
    R = parent(P)
    coeffs = _coeffs(P)
    for bits in 8:8:maxbits
        f = _ldl(_rounded(Qf, classes, coeffs, bits))
        f === nothing && continue
        D, L = f
        squares = [_poly(R, [(l[j], basis[j]) for j in eachindex(basis)]) for l in L]
        sum((QQ(d) * q^2 for (d, q) in zip(D, squares)); init = zero(R)) == P || continue
        # each square with a positive leading coefficient, the weight absorbing the scale
        return Giac.SOSCertificate(label, D, [Giac.to_giac(q) for q in squares])
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# The semidefinite programs
# ---------------------------------------------------------------------------------------------

_model() = (m = Model(Clarabel.Optimizer); set_silent(m); m)

# A Gram matrix for the polynomial with coefficients `coeffs` in `basis`, as far inside the cone
# as the problem allows (the largest t with Q − tI ⪰ 0); nothing if the solver finds none.
function _gram(coeffs, basis)
    N = length(basis)
    N == 0 && return nothing
    model = _model()
    @variable(model, Q[1:N, 1:N], Symmetric)
    @variable(model, t <= 1)
    @constraint(model, Q - t * I(N) in PSDCone())
    _match!(model, Q, _classes(basis), Dict(α => Float64(c) for (α, c) in coeffs)) || return nothing
    @objective(model, Max, t)
    optimize!(model)
    is_solved_and_feasible(model; allow_almost = true) || return nothing
    return value.(Q)
end

function _sos(P, label; bits = 48)
    iszero(P) && return Giac.SOSCertificate(label, Rational{BigInt}[], GiacExpr[])
    support = [collect(e) for e in exponent_vectors(P)]
    isodd(maximum(sum, support)) && return nothing
    basis = _half_basis(support, nvars(parent(P)))
    Qf = _gram(_coeffs(P), basis)
    Qf === nothing && return nothing
    return _certificate(P, basis, _classes(basis), Qf, bits, label)
end

function _nemo_polynomial(p, vars)
    p = convert(GiacExpr, p)
    if vars === nothing
        v = only(Giac.to_nemo((p,)))
        v isa Nemo.FracElem && throw(ArgumentError("sos_decomposition: $p is not a polynomial"))
        return v
    end
    R, _ = polynomial_ring(QQ, Symbol.(string.(vars)))
    return Giac.to_nemo(p, R)
end

function Giac.sos_decomposition(p; vars = nothing, bits = 48)
    P = _nemo_polynomial(p, vars)
    return _sos(P, Giac.to_giac(P); bits)
end

# ---------------------------------------------------------------------------------------------
# Lyapunov functions
# ---------------------------------------------------------------------------------------------

# The exponents of total degree lo..hi in n variables
_monomials(n, lo, hi) = _Exp[collect(m) for m in Iterators.product(fill(0:hi, n)...) if lo <= sum(m) <= hi]

function Giac.lyapunov(f, x; degree::Integer = 2, strict::Bool = true, ε = 1 // 1000, bits = 48)
    iseven(degree) && degree >= 2 || throw(ArgumentError("lyapunov: degree must be even and at least 2"))
    n = length(x)
    length(f) == n || throw(ArgumentError("lyapunov: one component of f per state variable"))
    R, _ = polynomial_ring(QQ, Symbol.(string.(x)))
    F = [Giac.to_nemo(convert(GiacExpr, fi), R) for fi in f]
    all(Fi -> iszero(constant_coefficient(Fi)), F) || throw(ArgumentError("lyapunov: the origin is not an equilibrium, f(0) ≠ 0"))
    ε, δ = Rational{BigInt}(ε), strict ? Rational{BigInt}(ε) : zero(Rational{BigInt})
    monos = _monomials(n, 2, degree)
    unit(i) = [j == i ? 1 : 0 for j in 1:n]

    # V = Σ cₖ x^αₖ and V̇ = ∇V · f, as exponent => affine expression in the cₖ
    model = _model()
    @variable(model, c[1:length(monos)])
    Vc = Dict{_Exp, AffExpr}(α => 1.0 * c[k] for (k, α) in enumerate(monos))
    Vd = Dict{_Exp, AffExpr}()
    for (k, α) in enumerate(monos), i in 1:n
        α[i] == 0 && continue
        for (fc, β) in zip(coefficients(F[i]), exponent_vectors(F[i]))
            γ = α .- unit(i) .+ β
            Vd[γ] = add_to_expression!(get(Vd, γ, zero(AffExpr)), α[i] * Float64(fc), c[k])
        end
    end
    p1 = copy(Vc)
    p2 = Dict{_Exp, AffExpr}(γ => -e for (γ, e) in Vd)
    for i in 1:n
        e2 = 2 .* unit(i)
        p1[e2] = get(p1, e2, zero(AffExpr)) - Float64(ε)
        p2[e2] = get(p2, e2, zero(AffExpr)) - Float64(δ)
    end
    dF = maximum(total_degree, F)
    b1 = _monomials(n, 1, degree ÷ 2)
    b2 = _monomials(n, 1, (degree - 1 + dF) ÷ 2)
    N1, N2 = length(b1), length(b2)
    @variable(model, Q1[1:N1, 1:N1], Symmetric)
    @variable(model, Q2[1:N2, 1:N2], Symmetric)
    @variable(model, t <= 1)
    @constraint(model, Q1 - t * I(N1) in PSDCone())
    @constraint(model, Q2 - t * I(N2) in PSDCone())
    _match!(model, Q1, _classes(b1), p1)
    _match!(model, Q2, _classes(b2), p2)
    @constraint(model, sum(Q1[a, a] for a in 1:N1) <= 100)       # V's scale
    @objective(model, Max, t)
    optimize!(model)
    is_solved_and_feasible(model; allow_almost = true) || return nothing
    # any positive multiple of V will do: the largest coefficient 1 keeps the rationals short
    cf = value.(c)
    cf ./= maximum(abs, cf)

    # V made rational, then each condition certified exactly on its own
    for vbits in 8:8:bits
        cr = [rationalize(BigInt, v; tol = 2.0^-vbits) for v in cf]
        V = _poly(R, [(cr[k], monos[k]) for k in eachindex(monos)])
        Vdot = sum(Nemo.derivative(V, i) * F[i] for i in 1:n)
        sq = sum(gens(R)[i]^2 for i in 1:n)
        pos = _sos(V - QQ(ε) * sq, Giac.to_giac(V - QQ(ε) * sq); bits)
        pos === nothing && continue
        dec = _sos(-Vdot - QQ(δ) * sq, Giac.to_giac(-Vdot - QQ(δ) * sq); bits)
        dec === nothing && continue
        return Giac.LyapunovCertificate(Giac.to_giac(V), Giac.to_giac(Vdot), ε, δ, pos, dec)
    end
    return nothing
end

end # module
