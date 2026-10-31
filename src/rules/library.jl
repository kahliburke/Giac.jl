# A catalog of identities Giac does not apply on its own, each written `longer ~ shorter` with the
# conditions under which it holds. The conditions ask Giac, so declared assumptions decide where
# they apply: with `@giac_var p::positive q::positive`, `log(p) + log(q)` combines and
# `log(x) + log(y)` does not. Families (the trigonometric identities and their hyperbolic
# counterparts) are generated, and every entry is checked by `verify` in the test suite.

"""
    IdentityCatalog

Rules from the identity catalog, with the topic of each: a vector of rules for [`rewrite`](@ref)
and [`simplest`](@ref) that displays grouped by topic. See [`identities`](@ref).
"""
struct IdentityCatalog <: AbstractVector{Rule}
    rules::Vector{Rule}
    topics::Vector{Symbol}
end
Base.size(c::IdentityCatalog) = size(c.rules)
Base.getindex(c::IdentityCatalog, i::Int) = c.rules[i]
Base.IndexStyle(::Type{IdentityCatalog}) = IndexLinear()

function Base.show(io::IO, ::MIME"text/latex", c::IdentityCatalog)
    rows = String[]
    for topic in unique(c.topics)
        push!(rows, "\\textbf{" * replace(string(topic), "_" => " ") * "} & ")
        append!(rows, "& " * _rule_latex(r) for (r, t) in zip(c.rules, c.topics) if t === topic)
    end
    print(io, "\$\$\\begin{array}{ll}", join(rows, " \\\\ "), "\\end{array}\$\$")
end

const _TOPICS = (:trig, :hyperbolic, :inverse_trig, :exp, :log, :power, :abs, :complex, :integer,
                 :trig_linear, :hyperbolic_linear)
# topics whose identities make an expression shorter; the `_linear` ones turn products and
# powers of trigonometric functions into sums instead
const _SIMPLIFYING = filter(t -> !endswith(string(t), "_linear"), _TOPICS)

const _CATALOG = Ref{Any}(nothing)
const _LIBRARY = Dict{Any, IdentityCatalog}()
const _LIBRARY_LOCK = ReentrantLock()

"""
    identities(topics...; direction = :simplify)

The identities of the catalog in the given topics (all that simplify, without topics), as rules:
`direction = :simplify` rewrites the longer side to the shorter, `:expand` the other way (for
the identities that can be turned around). Topics:

- `:trig`, `:hyperbolic`: Pythagorean forms, `tan`, multiple angles, sums and differences of angles.
- `:trig_linear`, `:hyperbolic_linear`: products and powers to sums (power reduction,
  product-to-sum); these lengthen, for linear forms.
- `:inverse_trig`: a function of an inverse function, such as `cosh(asinh x) = √(1 + x²)`.
- `:exp`, `:log`, `:power`: products of exponentials, sums of logarithms of positive quantities,
  products of powers, each under the conditions where they hold.
- `:abs`, `:complex`: absolute values and conjugates of quantities known nonnegative or real.
- `:integer`: `sin(nπ) = 0`, `cos(x + nπ) = (-1)ⁿcos x` and the like, for `n` declared `::integer`.

The result displays grouped by topic.
"""
function identities(topics::Symbol...; direction::Symbol = :simplify)
    direction in (:simplify, :expand) || throw(ArgumentError("direction is :simplify or :expand, got :$direction"))
    ts = isempty(topics) ? _SIMPLIFYING : topics
    for t in ts
        t in _TOPICS || throw(ArgumentError("unknown topic :$t; the topics are $(join(_TOPICS, ", "))"))
    end
    return lock(_LIBRARY_LOCK) do
        get!(_LIBRARY, (ts, direction)) do
            entries = [(r, t) for (r, t) in _catalog() if t in ts]
            direction === :simplify && return IdentityCatalog(first.(entries), last.(entries))
            turned = [(r, t) for (r, t) in ((_inverse_or_nothing(r), t) for (r, t) in entries) if r !== nothing]
            IdentityCatalog(first.(turned), last.(turned))
        end
    end
end

# the rule turned around, or `nothing` if it cannot be (or would match any expression)
function _inverse_or_nothing(r::Rule)
    inv = try
        inverse(r)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    return inv.lhs isa Slot || inv.lhs isa Leaf ? nothing : inv
end

_catalog() = lock(_LIBRARY_LOCK) do
    _CATALOG[] === nothing && (_CATALOG[] = _build_catalog())
    _CATALOG[]
end

function _build_catalog()
    # variables private to the catalog, so assumptions on a user's `x` don't change it; shown
    # as x, y, a, b, n
    v = Dict(n => giac_identifier("identity_" * string(n)) for n in (:x, :y, :a, :b, :n))
    x, y, a, b, n = v[:x], v[:y], v[:a], v[:b], v[:n]
    names = Dict(e => n for (n, e) in v)
    π_ = convert(GiacExpr, π)
    i_ = Giac.giac_eval("i")
    out = Tuple{Rule, Symbol}[]
    function add!(topic, eq, vars...)
        vs = GiacExpr[p isa Pair ? first(p) : p for p in vars]
        _key(Giac.lhs(eq)) == _key(Giac.rhs(eq)) && return
        r = _rule(eq, vs, Any[p isa Pair ? last(p) : nothing for p in vars], [names[e] for e in vs])
        _giac_does_it(r) || push!(out, (r, topic))
    end

    # Circular and hyperbolic functions differ by the sign σ wherever a product of two sines
    # appears (Osborn's rule): one generator writes both families.
    for (topic, S, C, T, σ) in ((:trig, sin, cos, tan, -1), (:hyperbolic, sinh, cosh, tanh, 1))
        add!(topic, C(x)^2 - σ * S(x)^2 ~ 1, x)
        add!(topic, 1 + σ * S(x)^2 ~ C(x)^2, x)
        add!(topic, C(x)^2 - 1 ~ σ * S(x)^2, x)
        add!(topic, S(x) / C(x) ~ T(x), x)
        add!(topic, 1 - σ * T(x)^2 ~ 1 / C(x)^2, x)
        add!(topic, 2S(x) * C(x) ~ S(2x), x)
        add!(topic, C(x)^2 + σ * S(x)^2 ~ C(2x), x)
        add!(topic, 3S(x) + 4σ * S(x)^3 ~ S(3x), x)
        add!(topic, 4C(x)^3 - 3C(x) ~ C(3x), x)
        add!(topic, 2T(x) / (1 + σ * T(x)^2) ~ T(2x), x)
        add!(topic, S(x) * C(y) + C(x) * S(y) ~ S(x + y), x, y)
        add!(topic, S(x) * C(y) - C(x) * S(y) ~ S(x - y), x, y)
        add!(topic, C(x) * C(y) + σ * S(x) * S(y) ~ C(x + y), x, y)
        add!(topic, C(x) * C(y) - σ * S(x) * S(y) ~ C(x - y), x, y)
        add!(topic, (T(x) + T(y)) / (1 + σ * T(x) * T(y)) ~ T(x + y), x, y)

        linear = Symbol(topic, :_linear)
        add!(linear, S(x)^2 ~ (C(2x) - 1) / (2σ), x)
        add!(linear, C(x)^2 ~ (1 + C(2x)) / 2, x)
        add!(linear, S(x) * S(y) ~ σ * (C(x + y) - C(x - y)) / 2, x, y)
        add!(linear, C(x) * C(y) ~ (C(x + y) + C(x - y)) / 2, x, y)
        add!(linear, S(x) * C(y) ~ (S(x + y) + S(x - y)) / 2, x, y)
    end

    # a function of an inverse function, as an algebraic expression
    one_minus, one_plus = sqrt(1 - x^2), sqrt(1 + x^2)
    for (lhs, rhs, test) in ((sin(acos(x)), one_minus, nothing), (cos(asin(x)), one_minus, nothing),
                             (tan(asin(x)), x / one_minus, nothing), (tan(acos(x)), one_minus / x, nothing),
                             (sin(atan(x)), x / one_plus, nothing), (cos(atan(x)), 1 / one_plus, nothing),
                             (cosh(asinh(x)), one_plus, nothing), (tanh(asinh(x)), x / one_plus, nothing),
                             (sinh(atanh(x)), x / one_minus, in_interval(-1, 1)),
                             (cosh(atanh(x)), 1 / one_minus, in_interval(-1, 1)))
        # where Giac has already written the left side algebraically, there is nothing to match
        _count(e -> _head(e) in ("sin", "cos", "tan", "sinh", "cosh", "tanh"), lhs) > 0 || continue
        add!(:inverse_trig, lhs ~ rhs, test === nothing ? x : x => test)
    end

    add!(:exp, exp(x) * exp(y) ~ exp(x + y), x, y)
    add!(:log, log(x) + log(y) ~ log(x * y), x => is_positive, y => is_positive)
    add!(:log, log(x) - log(y) ~ log(x / y), x => is_positive, y => is_positive)
    add!(:log, log(exp(x)) ~ x, x => is_real)
    add!(:power, x^a * x^b ~ x^(a + b), x, a, b)
    add!(:power, (x^a)^b ~ x^(a * b), x => is_positive, a => is_real, b => is_real)
    add!(:power, x^a * y^a ~ (x * y)^a, x => is_nonnegative, y => is_nonnegative, a => is_real)
    add!(:power, x^n * y^n ~ (x * y)^n, x, y, n => is_integer_valued)
    add!(:abs, abs(x) ~ x, x => is_nonnegative)
    add!(:abs, abs(x) ~ -x, x => is_negative)
    add!(:complex, conj(x) ~ x, x => is_real)
    add!(:complex, Giac.invoke_cmd(:re, x) ~ x, x => is_real)
    add!(:complex, Giac.invoke_cmd(:im, x) ~ 0, x => is_real)

    add!(:integer, sin(n * π_) ~ 0, n => is_integer_valued)
    add!(:integer, cos(n * π_) ~ (-1)^n, n => is_integer_valued)
    add!(:integer, tan(n * π_) ~ 0, n => is_integer_valued)
    add!(:integer, sin(x + n * π_) ~ (-1)^n * sin(x), x, n => is_integer_valued)
    add!(:integer, cos(x + n * π_) ~ (-1)^n * cos(x), x, n => is_integer_valued)
    add!(:integer, exp(2π_ * i_ * n) ~ 1, n => is_integer_valued)
    add!(:integer, convert(GiacExpr, -1)^(2n) ~ 1, n => is_integer_valued)
    add!(:integer, Giac.invoke_cmd(:floor, n) ~ n, n => is_integer_valued)
    add!(:integer, Giac.invoke_cmd(:ceil, n) ~ n, n => is_integer_valued)
    return out
end

# Whether Giac already rewrites the rule's left side itself, under the conditions of its tests: the
# left side, instantiated with fresh variables assumed what the tests ask, no longer matches.
function _giac_does_it(r::Rule)
    slots = _slots(r.lhs)
    fresh = Dict(s.name => giac_identifier("catalog_check_$(s.name)") for s in slots)
    try
        for s in slots
            Giac._declare!(fresh[s.name], s isa Slot ? _assumption_for(s) : nothing)
        end
        b = foldl((b, (n, v)) -> Bindings(b, n, v), fresh; init = Bindings())
        return r(_instantiate(r.lhs, b)) === nothing
    finally
        foreach(v -> Giac._declare!(v, nothing), values(fresh))
    end
end

# ---------------------------------------------------------------------------------------------
# Rule sets by topic and direction
# ---------------------------------------------------------------------------------------------

"""
    trig_simplify()

Shorter forms of trigonometric and hyperbolic expressions: `identities(:trig, :hyperbolic)`.
Sums match up to a common factor, so `3sin²t + 3cos²t` becomes `3` and `sin t cos t` becomes
`sin(2t)/2`.
"""
trig_simplify() = identities(:trig, :hyperbolic)

"""
    log_expand()
    log_combine()

Logarithms of products split into sums of logarithms, and back: `log(a*b) = log(a) + log(b)` and
`log(a/b) = log(a) - log(b)`, for positive `a` and `b`. (Giac itself writes `log(a^n)` as
`n*log(a)` for positive `a`.)
"""
log_expand() = identities(:log; direction = :expand)

"See [`log_expand`](@ref)."
log_combine() = identities(:log)

"""
    power_expand()
    power_combine()

Powers of products as products of powers and back, `(a*b)^n = a^n * b^n` (for an integer `n`,
or nonnegative `a` and `b`), and `x^a x^b = x^(a+b)`. Square roots are powers `1/2`, so
`√a·√b = √(ab)` for nonnegative `a` and `b`.
"""
power_expand() = identities(:power; direction = :expand)

"See [`power_expand`](@ref)."
power_combine() = identities(:power)

"""
    exp_expand()
    exp_combine()

`exp(a + b) = exp(a) * exp(b)`, and back.
"""
exp_expand() = identities(:exp; direction = :expand)

"See [`exp_expand`](@ref)."
exp_combine() = identities(:exp)
