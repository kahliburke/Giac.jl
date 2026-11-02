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

function _sos(P, label; bits = 48, maxbasis = typemax(Int))
    iszero(P) && return Giac.SOSCertificate(label, Rational{BigInt}[], GiacExpr[])
    support = [collect(e) for e in exponent_vectors(P)]
    isodd(maximum(sum, support)) && return nothing
    basis = _half_basis(support, nvars(parent(P)))
    length(basis) > maxbasis && return nothing
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

# for positivity: declines a semidefinite program over more than `maxbasis` monomials
Giac._sos_certificate(P::QQMPolyRingElem; maxbasis = 120) = _sos(P, Giac.to_giac(P); maxbasis)

# ---------------------------------------------------------------------------------------------
# Lyapunov functions
# ---------------------------------------------------------------------------------------------

# The exponents of total degree lo..hi in n variables
_monomials(n, lo, hi) = _Exp[collect(m) for m in Iterators.product(fill(0:hi, n)...) if lo <= sum(m) <= hi]

# ẋ = f(x) as polynomials of a ring in the state variables, with the origin an equilibrium
function _dynamics(f, x, name)
    n = length(x)
    length(f) == n || throw(ArgumentError("$name: one component of f per state variable"))
    R, _ = polynomial_ring(QQ, Symbol.(string.(x)))
    F = [Giac.to_nemo(convert(GiacExpr, fi), R) for fi in f]
    all(Fi -> iszero(constant_coefficient(Fi)), F) || throw(ArgumentError("$name: the origin is not an equilibrium, f(0) ≠ 0"))
    return n, R, F
end

function Giac.lyapunov(f, x; degree::Integer = 2, strict::Bool = true, ε = 1 // 1000, bits = 48)
    iseven(degree) && degree >= 2 || throw(ArgumentError("lyapunov: degree must be even and at least 2"))
    n, R, F = _dynamics(f, x, "lyapunov")
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

# ---------------------------------------------------------------------------------------------
# Regions of attraction
# ---------------------------------------------------------------------------------------------

# A with f(x) = A x + higher terms
function _linear_part(F, n)
    A = zeros(Rational{BigInt}, n, n)
    for i in 1:n, (c, e) in zip(coefficients(F[i]), exponent_vectors(F[i]))
        sum(e) == 1 && (A[i, findfirst(==(1), e)] += Rational{BigInt}(c))
    end
    return A
end

# P with AᵀP + PA = −I, solved exactly; nothing unless it is positive definite
function _lyapunov_matrix(A)
    n = size(A, 1)
    pairs = [(i, j) for i in 1:n for j in i:n]
    idx = Dict(p => k for (k, p) in enumerate(pairs))
    at(i, j) = idx[(min(i, j), max(i, j))]
    M = zero_matrix(QQ, length(pairs), length(pairs))
    b = zero_matrix(QQ, length(pairs), 1)
    for (row, (i, j)) in enumerate(pairs)
        for k in 1:n           # (AᵀP)[i, j] + (PA)[i, j]
            M[row, at(k, j)] = M[row, at(k, j)] + QQ(A[k, i])
            M[row, at(i, k)] = M[row, at(i, k)] + QQ(A[k, j])
        end
        b[row, 1] = i == j ? QQ(-1) : QQ(0)
    end
    ok, sol = can_solve_with_solution(M, b; side = :right)
    ok || return nothing
    P = [Rational{BigInt}(sol[at(i, j), 1]) for i in 1:n, j in 1:n]
    f = _ldl(P)
    return f !== nothing && length(first(f)) == n ? P : nothing
end

# nᵀ S n for an exactly positive semidefinite rational S, as a certificate; nothing if S is not
function _gram_certificate(S, basis, R)
    f = _ldl(S)
    f === nothing && return nothing
    D, L = f
    squares = [_poly(R, [(l[j], basis[j]) for j in eachindex(basis)]) for l in L]
    s = sum((QQ(d) * q^2 for (d, q) in zip(D, squares)); init = zero(R))
    return s, Giac.SOSCertificate(Giac.to_giac(s), D, [Giac.to_giac(q) for q in squares])
end

# exponent => affine expression, for a polynomial whose coefficients are those of nᵀ S n
function _gram_polynomial(S, basis)
    p = Dict{_Exp, AffExpr}()
    for a in eachindex(basis), b in a:length(basis)
        α = basis[a] .+ basis[b]
        p[α] = add_to_expression!(get(p, α, zero(AffExpr)), _weight(a, b), S[a, b])
    end
    return p
end

function Giac.region_of_attraction(f, x; V = nothing, multiplier_degree::Integer = 2, ε = 1 // 1000,
                                   tolerance = 1 // 1000, bits = 48)
    iseven(multiplier_degree) && multiplier_degree >= 2 ||
        throw(ArgumentError("region_of_attraction: multiplier_degree must be even and at least 2"))
    n, R, F = _dynamics(f, x, "region_of_attraction")
    xs = gens(R)
    if V === nothing
        P = _lyapunov_matrix(_linear_part(F, n))
        P === nothing && throw(ArgumentError("region_of_attraction: the linearization at the origin is not " *
                                             "asymptotically stable; pass a V"))
        Vp = sum(QQ(P[i, j]) * xs[i] * xs[j] for i in 1:n, j in 1:n)
    else
        Vp = Giac.to_nemo(convert(GiacExpr, V), R)
        iszero(constant_coefficient(Vp)) || throw(ArgumentError("region_of_attraction: V(0) ≠ 0"))
    end
    ε = Rational{BigInt}(ε)
    sq = sum(g^2 for g in xs)
    pos = _sos(Vp - QQ(ε) * sq, Giac.to_giac(Vp - QQ(ε) * sq); bits)
    pos === nothing && throw(ArgumentError("region_of_attraction: V − ε|x|² is not a sum of squares"))
    Vdot = sum(Nemo.derivative(Vp, i) * F[i] for i in 1:n)

    sbasis = _monomials(n, 1, multiplier_degree ÷ 2)
    D = max(total_degree(Vdot), multiplier_degree + total_degree(Vp))
    qbasis = _monomials(n, 1, cld(D, 2))
    base = _coeffs(-Vdot - QQ(ε) * sq)
    Vterms = [(Float64(c), collect(e)) for (c, e) in zip(coefficients(Vp), exponent_vectors(Vp))]

    # a multiplier's Gram matrix for c, well inside the cone, or nothing
    function multiplier(c)
        model = _model()
        Ns, Nq = length(sbasis), length(qbasis)
        @variable(model, S[1:Ns, 1:Ns], Symmetric)
        @variable(model, Q[1:Nq, 1:Nq], Symmetric)
        @variable(model, t <= 1)
        @constraint(model, S - t * I(Ns) in PSDCone())
        @constraint(model, Q - t * I(Nq) in PSDCone())
        @constraint(model, sum(S[a, a] for a in 1:Ns) <= 100)
        s = _gram_polynomial(S, sbasis)
        p = Dict{_Exp, AffExpr}(α => AffExpr(Float64(v)) for (α, v) in base)
        for (α, e) in s                            # − s·c + s·V
            p[α] = add_to_expression!(get(p, α, zero(AffExpr)), -Float64(c), e)
            for (v, β) in Vterms
                γ = α .+ β
                p[γ] = add_to_expression!(get(p, γ, zero(AffExpr)), v, e)
            end
        end
        _match!(model, Q, _classes(qbasis), p) || return nothing
        @objective(model, Max, t)
        optimize!(model)
        is_solved_and_feasible(model; allow_almost = true) && value(t) > 1.0e-9 || return nothing
        return value.(S)
    end

    # bisection on c, then the largest c tried that certifies exactly
    lo, hi = 0.0, 1.0
    while multiplier(hi) !== nothing
        lo, hi = hi, 2hi
        hi > 2.0^30 && break
    end
    lo == 0.0 && multiplier(hi / 1024) === nothing && return nothing
    lo == 0.0 && (lo = hi / 1024)
    while (hi - lo) > Float64(tolerance) * hi
        mid = (lo + hi) / 2
        multiplier(mid) === nothing ? (hi = mid) : (lo = mid)
    end
    for shrink in (1, 0.99, 0.9, 0.5)
        c = rationalize(BigInt, lo * shrink; tol = Float64(tolerance) * lo / 10)
        Sf = multiplier(Float64(c))
        Sf === nothing && continue
        for sbits in 8:8:bits
            Sr = [rationalize(BigInt, Sf[a, b]; tol = 2.0^-sbits) for a in axes(Sf, 1), b in axes(Sf, 2)]
            got = _gram_certificate(Sr, sbasis, R)
            got === nothing && continue
            s, scert = got
            rest = -Vdot - QQ(ε) * sq - s * (QQ(c) - Vp)
            dec = _sos(rest, Giac.to_giac(rest); bits)
            dec === nothing && continue
            return Giac.AttractionCertificate(Giac.to_giac(Vp), Giac.to_giac(Vdot), c, ε, pos, scert, dec)
        end
    end
    return nothing
end

end # module
