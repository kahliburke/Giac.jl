# Sums over the roots of a polynomial, rootsum(p(z), z, g(z)) = Σ_{p(r)=0} g(r), for the
# antiderivative of a rational function whose denominator has a factor Q with no roots GIAC writes:
# ∫ B/Q = Σ_{Q(r)=0} B(r)/Q'(r) ln(x - r). GIAC keeps the call as it is. Its derivative is taken
# inside, its value is the sum over the roots in floating point, and `simplify` writes a sum of a
# rational function of z in closed form, from the power sums of the roots.

const _ROOTSUM = "rootsum"

"""
    rootsum(p, z, g)

Σ g(r) over the roots r of the squarefree polynomial `p` in `z`. `integrate` gives one for a rational
function whose denominator has no roots in radicals:

```julia
@giac_var x
integrate(1 / (x^5 - x + 1), x)     # rootsum(z^5-z+1, z, ln(x-z)/(5*z^4-1))
```
"""
rootsum(p::GiacExpr, z::GiacExpr, g::GiacExpr) = GiacExpr(_rootsum(_node(p), _node(z), _node(g)))

_rootsum(p::Node, z::Node, g::Node) = CallNode("of", Node[SymNode(_ROOTSUM), p, z, g])

# (p, z, g) of a rootsum, else nothing
_rootsum_parts(n::Node) = nothing
function _rootsum_parts(n::CallNode)
    (n.f == "of" && length(n.args) == 4 && n.args[1] isa SymNode && n.args[1].name == _ROOTSUM &&
     n.args[3] isa SymNode) || return nothing
    return (n.args[2], n.args[3], n.args[4])
end

_has_rootsum(n::Node) = _any_node(m -> _rootsum_parts(m) !== nothing, n)
_has_rootsum(e::GiacExpr) = _has_rootsum(_node(e))

# the free names of p and g, z being bound
function _rootsum_names!(out, p::Node, z::SymNode, g::Node)
    inner = _symbol_names!(_symbol_names!(Set{String}(), p), g)
    delete!(inner, z.name)
    return union!(out, inner)
end

# d/dx of the sum, taken inside, for p free of x
function _rootsum_derivative(p::Node, z::SymNode, g::Node, x::String)
    (z.name == x || x in _symbol_names(p)) && return nothing
    d = _derivative(g, x)
    (d === nothing || _is_zero_node(d)) && return d
    return _rootsum(p, z, d)
end

# the sum in floating point, where p and g have no free names but z; nothing otherwise. GIAC's
# roots are refined by Newton's method in BigFloat, and g is evaluated as Julia code at them.
function _rootsum_value(p::Node, z::SymNode, g::Node)
    isempty(setdiff(_symbol_names(p) ∪ _symbol_names(g), (z.name,))) || return nothing
    Z, G = GiacExpr(z), GiacExpr(g)
    coefficients = invoke_cmd(:symb2poly, GiacExpr(p), Z)
    a = [to_julia(c) for c in collect(coefficients)]
    all(c -> c isa Number, a) || return nothing
    roots = invoke_cmd(:proot, coefficients)
    giac_type(roots) == GenTypes.VECT || return nothing
    h = try
        _compile(G, (Z,))
    catch e
        e isa ArgumentError || rethrow()
        nothing
    end
    total = setprecision(BigFloat, 256) do
        A = Complex{BigFloat}.(a)
        da = [A[k] * (length(A) - k) for k in 1:(length(A) - 1)]
        horner(c, r) = foldl((v, ck) -> v * r + ck, c; init = zero(r))
        s = zero(Complex{BigFloat})
        for i in 1:length(roots)
            r = Complex{BigFloat}(to_julia(invoke_cmd(:evalf, roots[i])))
            for _ in 1:4
                r -= horner(A, r) / horner(da, r)
            end
            v = h === nothing ? to_julia(invoke_cmd(:evalf, substitute(G, Z => ComplexF64(r)))) : h(r)
            v isa Number || return nothing
            s += v
        end
        ComplexF64(s)
    end
    total === nothing && return nothing
    return abs(imag(total)) <= 1e-12 * (1 + abs(total)) ? real(total) : total
end

# e with each sum of a rational function of z in closed form: g reduced modulo p to a polynomial
# Σ cₖ zᵏ, whose sum is Σ cₖ sₖ for the power sums sₖ of the roots
_rootsums_summed(e::GiacExpr) = _has_rootsum(e) ? GiacExpr(_replace_nodes(_node(e)) do m
    parts = _rootsum_parts(m)
    parts === nothing && return nothing
    v = _rootsum_closed(parts...)
    return v === nothing ? nothing : _node(v)
end) : e

function _rootsum_closed(p::Node, z::SymNode, g::Node)
    _rational_in(g, z.name) || return nothing
    P, Z = GiacExpr(p), GiacExpr(z)
    G = invoke_cmd(:normal, GiacExpr(g))
    N, M = invoke_cmd(:numer, G), invoke_cmd(:denom, G)
    # 1/M modulo P from uM + vP = d, d free of z where M and P have no common root
    u, _, d = collect(invoke_cmd(:egcd, M, P, Z))
    z.name in _free_names(d) && return nothing
    R = invoke_cmd(:rem, N * u, P, Z)
    a = collect(invoke_cmd(:symb2poly, P, Z))            # aₙ, …, a₀
    n = length(a) - 1
    c = reverse(collect(invoke_cmd(:symb2poly, R, Z)))   # c₀, c₁, …
    # Newton's identities: aₙsₖ + aₙ₋₁sₖ₋₁ + … + aₙ₋ₖ₊₁s₁ + k aₙ₋ₖ = 0
    s = GiacExpr[convert(GiacExpr, n)]
    for k in 1:(length(c) - 1)
        t = k * (k <= n ? a[k + 1] : convert(GiacExpr, 0))
        for i in 1:(k - 1)
            i <= n && (t += a[i + 1] * s[k - i + 1])
        end
        push!(s, -t / a[1])
    end
    return invoke_cmd(:normal, sum(c[k] * s[k] for k in eachindex(c)) / d)
end

# whether the tree is a rational function of the variable named z
_rational_in(n::Union{NumNode, SymNode}, z::String) = true
_rational_in(n::AddNode, z::String) = all(t -> _rational_in(first(t), z), n.terms)
_rational_in(n::MulNode, z::String) =
    all(((b, e),) -> !(z in _symbol_names(e)) && (!(z in _symbol_names(b)) || (_rational_in(b, z) && e isa NumNode && e.v isa Integer)),
        n.factors)
_rational_in(n::Node, z::String) = !(z in _symbol_names(n))

# ∫ f dx for a rational function f by its partial fractions over the rationals: GIAC's integral
# of each, and for a fraction B/Qᵏ GIAC leaves, Hermite's reduction to B₁/Q and the sum of
# B₁(r)/Q'(r) ln(x - r) over the roots of Q; nothing for other f
function _rootsum_integral(f::GiacExpr, x::GiacExpr)
    xs = _node(x)
    (xs isa SymNode && _rational_in(_node(f), xs.name)) || return nothing
    pf = _node(invoke_cmd(:partfrac, f, x))
    terms = pf isa AddNode ? [_scale(t, c) for (t, c) in pf.terms] : [pf]
    total, summed = convert(GiacExpr, 0), false
    for t in terms
        T = GiacExpr(t)
        r = invoke_cmd(:integrate, T, x)
        if !_integral_left(r)
            total += r
            continue
        end
        g = _fraction_integral(T, x)
        g === nothing && return nothing
        total += g
        summed = true
    end
    summed || return nothing
    _is_zero_expr(_simplified(_rootsums_summed(Commands.diff(total, x)) - f)) || return nothing
    return total
end

# ∫ B/Qᵏ dx for Q squarefree: B = SQ + TQ' gives ∫ B/Qᵏ = -T/((k-1)Qᵏ⁻¹) + ∫ (S + T'/(k-1))/Qᵏ⁻¹
function _fraction_integral(F::GiacExpr, x::GiacExpr)
    B, D = invoke_cmd(:numer, F), invoke_cmd(:denom, F)
    fs = collect(invoke_cmd(:factors, D))               # [f₁, m₁, f₂, m₂, …]
    nonconstant = [(fs[i], fs[i + 1]) for i in 1:2:length(fs) if string(x) in _free_names(fs[i])]
    length(nonconstant) == 1 || return nothing
    Q, k = nonconstant[1][1], Int(to_julia(nonconstant[1][2]))
    B = B / invoke_cmd(:normal, D / Q^k)                 # the constant factor of D into B
    s, t, d = collect(invoke_cmd(:egcd, Q, Commands.diff(Q, x), x))
    part = convert(GiacExpr, 0)
    while k > 1
        T = invoke_cmd(:rem, B * t / d, Q, x)
        S = invoke_cmd(:quo, B - T * Commands.diff(Q, x), Q, x)
        part -= T / ((k - 1) * Q^(k - 1))
        B = invoke_cmd(:normal, S + Commands.diff(T, x) / (k - 1))
        k -= 1
    end
    q, B = invoke_cmd(:quo, B, Q, x), invoke_cmd(:rem, B, Q, x)
    _is_zero_expr(q) || (part += invoke_cmd(:integrate, q, x))
    _is_zero_expr(B) && return part
    z = _fresh_symbol(_node(F), "z")
    c = invoke_cmd(:normal, substitute(B / Commands.diff(Q, x), x => z))
    return part + rootsum(substitute(Q, x => z), z, c * log(x - z))
end
