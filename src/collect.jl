# Grouping an expression by monomials in chosen variables.

"""
    coefficients(e, vars) -> Dict{GiacExpr, GiacExpr}

The coefficient of each monomial in the variables `vars` (a variable or an array of them) in the
expanded `e`, as `monomial => coefficient`; the term free of `vars` is under `1`. A factor that
depends on `vars` without being a power of one (`sin(x1)`) is part of the monomial. Coefficients
are normalised, and monomials whose coefficient is zero are left out.

Not exported, as other algebra packages export a `coefficients` of their own:
`using Giac: coefficients`, or `Giac.coefficients(…)`.

```julia
@giac_var x[1:2] a b
c = Giac.coefficients(a * x1^2 + b * x1 * x2 + 2a * x1^2 + 3, x)
c[x1^2]          # 3*a
c[x1 * x2]       # b
c[1]             # 3
```
"""
function coefficients(e::GiacExpr, vars)
    return Dict(m => c for (m, c) in _collected(e, vars))
end

"""
    collect_terms(e, vars)

`e` as a sum over monomials in the variables `vars`, each with its coefficient collected and
normalised: `(a + 2b)*x1^2 + (a - b)*x1*x2 + …`. Higher degrees come first, and within a degree,
higher powers of the earlier variables. For an array, entry by entry. See [`coefficients`](@ref) for the pieces.
"""
function collect_terms(e::GiacExpr, vars)
    terms = [isone(c) ? m : isone(m) ? c : c * m for (m, c) in _collected(e, vars)]
    return isempty(terms) ? zero(GiacExpr) : reduce(+, terms)
end
collect_terms(A::AbstractArray, vars) = map(e -> collect_terms(e, vars), A)

# `monomial => coefficient` pairs in graded order, zero coefficients dropped
function _collected(e::GiacExpr, vars)
    varnames = [string(v) for v in (vars isa GiacExpr ? (vars,) : vec(collect(vars)))]
    depends(f) = !isdisjoint(_symbol_names(_node(f)), varnames)
    # key => (monomial, its order key, coefficient terms)
    groups = Dict{String, Tuple{GiacExpr, Vector{Int}, Vector{GiacExpr}}}()
    for term in _sum_terms(GiacExpr(_expand(_node(e))))
        fs = _product_factors(term)
        # powers of the variables in their order, then other factors (`sin(x1)`)
        mono = sort!(filter(depends, fs); by = f -> (_base_index(f, varnames), string(f)))
        coef = filter(!depends, fs)
        key = join(string.(mono), "*")
        m, _, cs = get!(groups, key) do
            powers = [_power(mono, v) for v in varnames]
            # total degree first, then the powers of the variables in order, all descending
            order = vcat(-sum(_degree, mono; init = 0), -powers)
            (isempty(mono) ? one(GiacExpr) : reduce(*, mono), order, GiacExpr[])
        end
        push!(cs, isempty(coef) ? one(GiacExpr) : reduce(*, coef))
    end
    out = Pair{GiacExpr, GiacExpr}[]
    for key in sort!(collect(keys(groups)); by = k -> (groups[k][2], k))
        m, _, cs = groups[key]
        c = reduce(+, cs)
        _has_denominator(_node(c)) && (c = Commands.normal(c))
        # a coefficient that cancels is the number 0 by now (summed in normal form, or normal'd)
        (n = _node(c); n isa NumNode && iszero(n.v)) || push!(out, m => c)
    end
    return out
end

_sum_terms(e::GiacExpr) = iscall(e) && symb_funcname(e) == "+" ? arguments(e) : [e]

function _product_factors(t::GiacExpr)
    iscall(t) || return [t]
    head = symb_funcname(t)
    head == "*" && return arguments(t)
    head == "-" && return vcat(convert(GiacExpr, -1), _product_factors(only(arguments(t))))
    return [t]
end

# the degree a factor adds to a monomial: its integer exponent, or 1
function _degree(f::GiacExpr)
    iscall(f) && symb_funcname(f) == "^" || return 1
    n = arguments(f)[2]
    return is_integer(n) ? Int(to_julia(n)) : 1
end

# the position in `varnames` of a factor's variable, if it is a variable or a power of one
function _base_index(f::GiacExpr, varnames)
    base = iscall(f) && symb_funcname(f) == "^" ? string(arguments(f)[1]) : string(f)
    return something(findfirst(==(base), varnames), length(varnames) + 1)
end

# the power of the variable named `v` among the factors of a monomial
function _power(factors, v::String)
    for f in factors
        string(f) == v && return 1
        iscall(f) && symb_funcname(f) == "^" && string(arguments(f)[1]) == v && return _degree(f)
    end
    return 0
end

# A polynomial in its names is expanded by GIAC's `normal`, which multiplies in its sparse
# polynomials; GIAC's `expand` multiplies symbolic sums, and is many times slower on large ones.
function _expand_cmd(e::GiacExpr)
    _is_polynomial(_node(e)) && return _invoke_cmd_direct(:normal, (e,))
    return _invoke_cmd_direct(:expand, (e,))
end

# Whether a tree is a polynomial in its names with exact coefficients
_is_polynomial(n::Node) = false
_is_polynomial(n::SymNode) = true
_is_polynomial(n::NumNode) = n.v isa Union{Integer, Rational}
_is_polynomial(n::AddNode) = all(((m, c),) -> c isa Union{Integer, Rational} && _is_polynomial(m), n.terms)
_is_polynomial(n::MulNode) =
    n.coef isa Union{Integer, Rational} &&
    all(((b, k),) -> k isa NumNode && k.v isa Integer && k.v >= 0 && _is_polynomial(b), n.factors)

# `factor(e)` over the field of e's coefficients, as other systems factor: GIAC's `withsqrt` mode,
# on by default, splits quadratic factors with square roots (x⁴ + 1 over ℚ(√2), Φ₅ over ℚ(√5)).
# A factorization over an extension is asked for with it: `factor(x^4 + 1, sqrt(2))`.
_factor_cmd(e::GiacExpr) = with_settings(() -> _invoke_cmd_direct(:factor, (e,)), :withsqrt => false)
# Primality of Mersenne and Fermat numbers. GIAC proves primality through PARI, whose APRCL test errs
# on some large primes (2^521 - 1); for these two forms there are complete tests that are also
# much faster: Lucas–Lehmer for 2^p - 1 (s₀ = 4, sₖ₊₁ = sₖ² - 2: prime exactly when s_(p-2) ≡ 0)
# and Pépin's for 2^(2^k) + 1 (prime exactly when 3^((F-1)/2) ≡ -1).

# `isprime(n)`: by these tests for the two forms, else GIAC's
function _isprime_cmd(n)
    m = _integer_value(n)
    m === nothing && return _invoke_cmd_direct(:isprime, (convert(GiacExpr, n),))
    r = _special_form_prime(m)
    r === nothing && return _invoke_cmd_direct(:isprime, (convert(GiacExpr, n),))
    return giac_eval(r ? "true" : "false")
end

_integer_value(n::Integer) = big(n)
function _integer_value(n::GiacExpr)
    v = _node(n)
    return v isa NumNode && v.v isa Integer ? big(v.v) : nothing
end
_integer_value(n) = nothing

# whether m is prime, for m a Mersenne number 2^p - 1 or a Fermat number 2^(2^k) + 1 past the
# sizes GIAC settles at once; nothing for other numbers
function _special_form_prime(m::BigInt)
    m < big(2)^64 && return nothing
    if ispow2(m + 1)
        p = trailing_zeros(m + 1)
        _small_prime(p) || return false                 # 2^ab - 1 has the factor 2^a - 1
        s = big(4)
        for _ in 1:(p - 2)
            s = mod(s * s - 2, m)
        end
        return iszero(s)
    elseif ispow2(m - 1) && ispow2(trailing_zeros(m - 1))
        return powermod(big(3), (m - 1) ÷ 2, m) == m - 1
    end
    return nothing
end

_small_prime(p::Integer) = p >= 2 && all(d -> p % d != 0, 2:isqrt(p))
