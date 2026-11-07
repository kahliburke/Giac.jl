# cos(2πk/q) in square roots when q is a power of 2 times 1 or a Fermat prime: the regular q-gon is
# constructible. For a Fermat prime p, by Gauss's periods. With g a primitive root mod p, the
# period ηⱼ,ᵢ is the sum of ζ^(g^e) over the exponents e ≡ i (mod 2ʲ), ζ = e^(2πi/p). The two
# periods of level j+1 inside ηⱼ,ᵢ are the roots of t² − ηⱼ,ᵢ t + P, P their product, which is a
# whole combination of the level-j periods, so each level is written in square roots of the one
# before. At level m−1 (p − 1 = 2ᵐ) a period is ζᵃ + ζ⁻ᵃ = 2cos(2πa/p). A factor 2 in q is a half
# angle: cos x = ±√((1 + cos 2x)/2).

"""
    in_radicals(e)

`e` with each `cos` and `sin` of a rational multiple of π written in square roots where the
regular polygon it belongs to is constructible: the denominator of the angle as a fraction of a
turn is a power of 2 times 1 or a Fermat prime up to 257 (Gauss's construction).

```julia
Giac.in_radicals(giac_eval("cos(2*pi/17)"))   # nested square roots of 17
```
"""
in_radicals(e::GiacExpr) = GiacExpr(_in_radicals(_node(e)))

function _in_radicals(n::Node)
    if n isa CallNode
        args = Node[_in_radicals(a) for a in n.args]
        if n.f in ("cos", "sin") && length(args) == 1
            v = _constructible_trig(n.f, args[1])
            v === nothing || return _node(v)
        end
        return _call(n.f, args)
    elseif n isa AddNode
        return _add((_scale(_in_radicals(m), c) for (m, c) in n.terms)...)
    elseif n isa MulNode
        return _mul(NumNode(n.coef), (_pow(_in_radicals(b), _in_radicals(e)) for (b, e) in n.factors)...)
    end
    return n
end

const _GAUSS_PRIMES = (1, 3, 5, 17, 257)

# cos rπ or sin rπ in square roots, or nothing when the polygon is not constructible
function _constructible_trig(f::String, a::Node)
    r = _pi_coefficient(a)
    r === nothing && return nothing
    # sin x = cos(π/2 − x); cos rπ = cos(2π · r/2), the angle as a fraction k/q of a turn
    t = (f == "sin" ? 1 // 2 - r : r) / 2
    q = denominator(t)
    q >> trailing_zeros(q) in _GAUSS_PRIMES || return nothing
    return _cos_turn(numerator(t), q)
end

function _pi_coefficient(a::Node)
    a isa SymNode && a.name == "pi" && return 1 // 1
    (a isa MulNode && length(a.factors) == 1 && isequal(a.factors[1], SymNode("pi") => _ONE)) || return nothing
    return a.coef isa Union{Integer, Rational} ? Rational{BigInt}(a.coef) : nothing
end

# cos(2πk/q), q a power of 2 times 1 or a Fermat prime
function _cos_turn(k::Integer, q::Integer)
    k = mod(k, q)
    g = gcd(k, q)
    k, q = k ÷ g, q ÷ g
    q == 1 && return one(GiacExpr)
    isodd(q) && return _cos_two_pi(Int(q), Int(k))
    q == 2 && return convert(GiacExpr, -1)
    q == 4 && return zero(GiacExpr)
    # cos x = ±√((1 + cos 2x)/2), the sign that of cos x
    c = _cos_turn(2k, q)
    return (cospi(2k / q) < 0 ? -1 : 1) * sqrt((1 + c) / 2)
end

# cos(2πk/p) for a Fermat prime p, in square roots
function _cos_two_pi(p::Int, k::Int)
    k == 0 && return one(GiacExpr)
    g = _primitive_root(p)
    m = trailing_zeros(p - 1)
    ζ(a) = cispi(2a / p)
    # level 0: the one period, the sum of all the roots, -1
    periods = [Set(powermod(g, e, p) for e in 0:(p - 2))]
    exprs = GiacExpr[convert(GiacExpr, -1)]
    for j in 0:(m - 2)
        # the periods of level j+1: exponents ≡ i (mod 2^(j+1))
        step = 2^(j + 1)
        next = [Set(powermod(g, e, p) for e in i:step:(p - 2)) for i in 0:(step - 1)]
        value(S) = sum(ζ(a) for a in S)
        nexprs = Vector{GiacExpr}(undef, step)
        for i in 0:(2^j - 1)
            A, B = next[i + 1], next[i + 1 + 2^j]
            # A·B as a constant plus a combination of the level-j periods
            counts = zeros(Int, p)
            for a in A, b in B
                counts[mod(a + b, p) + 1] += 1
            end
            P = convert(GiacExpr, counts[1]) + sum(counts[first(S) + 1] * exprs[l] for (l, S) in enumerate(periods))
            # the periods are real (each holds a and -a), so the roots are (η ± √(η² − 4P))/2;
            # A is the one with + when it is the larger
            sign = real(value(A)) > real(value(B)) ? 1 : -1
            root = sqrt(exprs[i + 1]^2 - 4P)
            nexprs[i + 1] = (exprs[i + 1] + sign * root) / 2
            nexprs[i + 1 + 2^j] = (exprs[i + 1] - sign * root) / 2
        end
        periods, exprs = next, nexprs
    end
    # level m-1: ζᵃ + ζ⁻ᵃ for a in the period; k's period holds k or -k
    l = findfirst(S -> k in S, periods)
    return invoke_cmd(:normal, exprs[l] / 2)
end

function _primitive_root(p::Int)
    factors = [2]               # p - 1 is a power of 2
    for g in 2:(p - 1)
        all(q -> powermod(g, (p - 1) ÷ q, p) != 1, factors) && return g
    end
    error("no primitive root mod $p")
end
