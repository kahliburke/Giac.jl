# Cube roots of numbers in Q(∛m) denested, as in Ramanujan's ∛(∛2 - 1) = (1 - ∛2 + ∛4)/∛9.
# ∛α, α = a + bt + ct² with t = ∛m, denests to β/∛r with β in Q(t) when rα is a cube β³ there for
# a rational r. The r tried are made of the primes of 3, m and α's norm, each to a power below 3.
# β = p + qt + st² comes from cube roots of rα at the three embeddings t ↦ ωᵏt, its rational
# coordinates recognized from high precision, and β³ = rα is checked exactly.

# e with each cube root of a positive number of Q(∛m) that denests, a factor of a product, written
# as a fresh name, and the names' denested values
function _cube_root_placeholders(e::GiacExpr)
    denested = Dict{GiacExpr, GiacExpr}()
    _any_node(_has_cubic_cube_root, _node(e)) || return (e, denested)
    taken = _node(e)
    r = _replace_nodes(taken) do n
        _has_cubic_cube_root(n) || return nothing
        parts, changed = Node[NumNode(n.coef)], false
        for (b, k) in n.factors
            v = _is_cube_root(k) ? _denest_cube_root(b) : nothing
            if v === nothing
                push!(parts, _pow(b, k))
            else
                name = _fresh_symbol(_mul(taken, (_node(p) for p in keys(denested))...), "cuberoot")
                denested[name] = v
                push!(parts, _node(name))
                changed = true
            end
        end
        return changed ? _mul(parts...) : nothing
    end
    return (GiacExpr(r), denested)
end

_is_cube_root(k::Node) = k isa NumNode && k.v == 1 // 3

# whether n is a product with a factor ∛α, α a sum of rationals times powers k/3 of one integer
_has_cubic_cube_root(n::Node) = n isa MulNode && any(((b, k),) -> _is_cube_root(k) && _cubic_coordinates(b) !== nothing, n.factors)

# (m, [a, b, c]) for α = a + b∛m + c∛m², m > 1 cube-free; nothing for other α
function _cubic_coordinates(α::Node)
    terms = α isa AddNode ? [(t, c) for (t, c) in α.terms] : [(α, 1)]
    m, coords = nothing, Rational{BigInt}[0, 0, 0]
    for (t, c) in terms
        c isa Union{Integer, Rational} || return nothing
        if t isa NumNode && t.v isa Union{Integer, Rational}
            coords[1] += c * t.v
            continue
        end
        t isa MulNode && length(t.factors) == 1 && t.coef isa Union{Integer, Rational} || return nothing
        b, e = only(t.factors)
        (b isa NumNode && b.v isa Integer && b.v > 1 && e isa NumNode && e.v isa Rational && denominator(e.v) == 3) || return nothing
        m === nothing && (m = BigInt(b.v))
        b.v == m || return nothing
        k = numerator(e.v)
        coords[mod(k, 3) + 1] += c * t.coef * big(m)^fld(k, 3)
    end
    m === nothing && return nothing
    round(BigInt, cbrt(big(m)))^3 == m && return nothing
    all(iszero, coords[2:3]) && return nothing
    return (m, coords)
end

function _denest_cube_root(α::Node)
    mc = _cubic_coordinates(α)
    mc === nothing && return nothing
    m, (a, b, c) = mc
    setprecision(BigFloat, 256) do
        t = cbrt(big(m))
        a + b * t + c * t^2 > 0 || return nothing
        norm = a^3 + b^3 * m + c^3 * m^2 - 3a * b * c * m
        primes = Set{BigInt}([3])
        for v in (m, numerator(norm), denominator(norm))
            iszero(v) || union!(primes, keys(_factor_int(abs(v))))
        end
        ps = sort!(collect(primes))
        length(ps) <= 5 || return nothing
        best = nothing
        for es in Iterators.product((0:2 for _ in ps)...)
            r = prod(big(p)^e for (p, e) in zip(ps, es); init = big(1))
            β = _cube_in_field(r .* [a, b, c], m, t)
            β === nothing && continue
            e = (β[1] + β[2] * convert(GiacExpr, m)^(1 // 3) + β[3] * convert(GiacExpr, m)^(2 // 3)) / convert(GiacExpr, r)^(1 // 3)
            (best === nothing || length(string(e)) < length(string(best))) && (best = e)
        end
        return best
    end
end

# β in Q(∛m) with β³ = the number with coordinates γ, β real at the real embedding; nothing if none
function _cube_in_field(γ, m, t)
    ω = cispi(big(2) / 3)
    g(k) = γ[1] + γ[2] * (ω^k * t) + γ[3] * (ω^k * t)^2
    g0 = real(g(0))
    β0 = sign(g0) * abs(g0)^(1 // 3)
    M = [ω^(j * k) * t^j for k in 0:2, j in 0:2]
    for root in 0:2
        β1 = g(1)^(1 // 3) * ω^root
        q = M \ [Complex(β0), β1, conj(β1)]
        all(v -> abs(imag(v)) < big(2.0)^-200, q) || continue
        β = [rationalize(BigInt, real(v); tol = big(2.0)^-180) for v in q]
        all(v -> denominator(v) < big(2)^32, β) || continue
        _cubic_cube(β, m) == γ && return β
    end
    return nothing
end

# the coordinates of β³ for β with coordinates (p, q, s) in Q(∛m)
function _cubic_cube(β, m)
    mul(x, y) = [x[1] * y[1] + m * (x[2] * y[3] + x[3] * y[2]),
                 x[1] * y[2] + x[2] * y[1] + m * x[3] * y[3],
                 x[1] * y[3] + x[2] * y[2] + x[3] * y[1]]
    return mul(mul(β, β), β)
end

function _factor_int(n::BigInt)
    out = Dict{BigInt, Int}()
    for p in big.((2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47))
        while n % p == 0
            out[p] = get(out, p, 0) + 1
            n ÷= p
        end
    end
    n > 1 && (out[n] = 1)
    return out
end
