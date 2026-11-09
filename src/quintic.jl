# The roots of a solvable quintic in radicals. An irreducible quintic is solvable when its Galois
# group lies in the Frobenius group F₂₀, the affine maps k ↦ ak + b of the roots x₀, …, x₄ taken in
# a suitable cyclic order. That order is the one where
#     θ = (Σ xₖxₖ₊₁ - Σ xₖxₖ₊₂)²
# is rational, θ being a root of Cayley's resolvent sextic. In that order the Lagrange resolvents
# rⱼ = Σ ζ^(jk) xₖ (ζ = e^(2πi/5)) have fifth powers uⱼ that the group and ζ ↦ ζᶜ only permute,
# j ↦ ca⁻¹j, so they are the roots of a rational quartic, and u₁ + u₄, u₂ + u₃ of a rational
# quadratic; u₁ is found from them with square roots. Each rⱼ/r₁ʲ is fixed where u₁ is, so it is a
# polynomial in u₁ with rational coefficients, and
#     xₖ = (r₀ + Σⱼ ζ^(-jk) r₁ʲ qⱼ(u₁))/5,  r₁ = u₁^(1/5).
# The rational numbers (θ, the quadratic's coefficients, the qⱼ) are read from the roots computed
# to high precision and recognized as fractions; the roots in radicals are checked against the
# numeric ones.

const _QUINTIC_BITS = 512

# the five roots of the quintic with rational coefficients `a` (from x⁵ down, a[1] ≠ 0) in
# radicals, or nothing when it is not solvable or the construction does not apply
function _quintic_radicals(a::Vector{<:Rational})
    length(a) == 6 && !iszero(a[1]) || return nothing
    a = a ./ a[1]
    setprecision(BigFloat, _QUINTIC_BITS) do
        xs = _precise_roots(a)
        xs === nothing && return nothing
        tol = BigFloat(2)^(-_QUINTIC_BITS ÷ 2)
        order = _f20_order(xs, tol)
        order === nothing && return nothing
        x = xs[order]
        ζ = cispi(big(2) / 5)
        r = [sum(ζ^(j * k) * x[k + 1] for k in 0:4) for j in 0:4]
        u = [r[j + 1]^5 for j in 1:4]
        minimum(abs(u[i] - u[j]) for i in 1:4 for j in (i + 1):4) > tol || return nothing
        # qⱼ(u₁) = rⱼ/r₁ʲ, its conjugates r_(mj)/r_mʲ at u_m
        V = [u[m]^k for m in 1:4, k in 0:3]
        qs = Vector{Rational{BigInt}}[]
        for j in 2:4
            c = _rationals(V \ [r[mod(m * j, 5) + 1] / r[m + 1]^j for m in 1:4], tol)
            c === nothing && return nothing
            push!(qs, c)
        end
        U = _resolvent_root(u, tol)
        U === nothing && return nothing
        five = convert(GiacExpr, 5)
        Z = [_root_of_unity(k) for k in 0:4]
        # r₁ = ζˢ u₁^(1/5), s the power that gives the r₁ of this order
        R = nothing
        for s in 0:4
            cand = Z[s + 1] * _fifth_root(U)
            abs(_giac_complex(cand) - Complex{Float64}(r[2])) < 1e-8 * (1 + abs(r[2])) && (R = cand; break)
        end
        R === nothing && return nothing
        q(c) = invoke_cmd(:normal, sum(c[k + 1] * U^k for k in 0:3))
        terms = [R, R^2 * q(qs[1]), R^3 * q(qs[2]), R^4 * q(qs[3])]
        r0 = convert(GiacExpr, -a[2])
        roots = [(r0 + sum(Z[mod(-j * k, 5) + 1] * terms[j] for j in 1:4)) / five for k in 0:4]
        # each root is one of the numeric ones
        for (k, e) in enumerate(roots)
            abs(_giac_complex(e) - Complex{Float64}(x[k])) < 1e-8 * (1 + abs(x[k])) || return nothing
        end
        return roots
    end
end

# the roots of the monic polynomial `a` in BigFloat: GIAC's, refined by Newton's method
function _precise_roots(a)
    A = Complex{BigFloat}.(a)
    dA = [A[k] * (length(A) - k) for k in 1:(length(A) - 1)]
    horner(c, z) = foldl((v, ck) -> v * z + ck, c; init = zero(z))
    seeds = invoke_cmd(:proot, giac_eval("[" * join(string.(numerator.(a)) .* "/" .* string.(denominator.(a)), ",") * "]"))
    giac_type(seeds) == GenTypes.VECT && length(seeds) == 5 || return nothing
    xs = Complex{BigFloat}[]
    for i in 1:5
        z = Complex{BigFloat}(to_julia(invoke_cmd(:evalf, seeds[i])))
        for _ in 1:40
            step = horner(A, z) / horner(dA, z)
            z -= step
            abs(step) < BigFloat(2)^(-precision(BigFloat) + 8) * (1 + abs(z)) && break
        end
        push!(xs, z)
    end
    minimum(abs(xs[i] - xs[j]) for i in 1:5 for j in (i + 1):5) > BigFloat(2)^(-precision(BigFloat) ÷ 4) || return nothing
    return xs
end

# a cyclic order of the roots in which θ is rational, as a permutation, or nothing
function _f20_order(xs, tol)
    θ(o) = (sum(xs[o[k]] * xs[o[mod1(k + 1, 5)]] for k in 1:5) - sum(xs[o[k]] * xs[o[mod1(k + 2, 5)]] for k in 1:5))^2
    for p in _permutations([2, 3, 4, 5])
        o = vcat(1, p)
        t = θ(o)
        abs(imag(t)) < tol * (1 + abs(t)) || continue
        _rational(real(t), tol) === nothing || return o
    end
    return nothing
end

_permutations(v) = length(v) <= 1 ? [copy(v)] :
    [vcat(v[i], p) for i in eachindex(v) for p in _permutations([v[j] for j in eachindex(v) if j != i])]

# a fraction within tol of the real number v, with a denominator below 2^64; nothing if none
function _rational(v::BigFloat, tol)
    q = rationalize(BigInt, v; tol = tol * (1 + abs(v)))
    return denominator(q) < big(2)^64 ? q : nothing
end
function _rationals(vs, tol)
    out = Rational{BigInt}[]
    for v in vs
        abs(imag(v)) < tol * (1 + abs(v)) || return nothing
        q = _rational(real(v), tol)
        q === nothing && return nothing
        push!(out, q)
    end
    return out
end

# u₁ in square roots: s = u₁ + u₄ is a root of the rational quadratic of s and u₂ + u₃, and
# u₁u₄ = α + βs; then u₁ = (s ± √(s² - 4u₁u₄))/2
function _resolvent_root(u, tol)
    s1, s2, p1, p2 = u[1] + u[4], u[2] + u[3], u[1] * u[4], u[2] * u[3]
    sd = _rationals([s1 + s2, (s1 - s2)^2], tol)
    sd === nothing && return nothing
    σ, Δ = sd
    two = convert(GiacExpr, 2)
    S = nothing
    for ε in (1, -1)
        cand = (convert(GiacExpr, σ) + ε * sqrt(convert(GiacExpr, Δ))) / two
        abs(_giac_complex(cand) - Complex{Float64}(s1)) < 1e-10 * (1 + abs(s1)) && (S = cand; break)
    end
    S === nothing && return nothing
    if abs(s1 - s2) > tol * (1 + abs(s1))
        αβ = _rationals([(p2 * s1 - p1 * s2) / (s1 - s2), (p1 - p2) / (s1 - s2)], tol)
        αβ === nothing && return nothing
        P = convert(GiacExpr, αβ[1]) + convert(GiacExpr, αβ[2]) * S
    else
        pr = _rationals([p1], tol)
        pr === nothing && return nothing
        P = convert(GiacExpr, pr[1])
    end
    disc = invoke_cmd(:normal, S^2 - 4P)
    for ε in (1, -1)
        cand = (S + ε * sqrt(disc)) / two
        abs(_giac_complex(cand) - Complex{Float64}(u[1])) < 1e-10 * (1 + abs(u[1])) && return cand
    end
    return nothing
end

# U^(1/5), the real fifth root for real negative U
function _fifth_root(U::GiacExpr)
    v = _giac_complex(U)
    abs(imag(v)) < 1e-12 * (1 + abs(v)) && real(v) < 0 && return -(-U)^(1 // 5)
    return U^(1 // 5)
end

# e^(2πik/5) in square roots
_root_of_unity(k) = k == 0 ? convert(GiacExpr, 1) :
    in_radicals(giac_eval("cos(2*pi*$k/5)")) + convert(GiacExpr, im) * in_radicals(giac_eval("sin(2*pi*$k/5)"))

_giac_complex(e::GiacExpr) = Complex{Float64}(to_julia(invoke_cmd(:evalf, e, 30)))
