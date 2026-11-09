# Bessel, Hankel, Airy and Struve functions. GIAC's BesselJ and its kin take only an integer order and a
# real argument, and have no derivatives, so these are kept as calls GIAC does not interpret,
# named as SpecialFunctions names them: besselj(ν, x), …, airyaiprime(x). Their derivatives are
# taken on the tree, their values come from SpecialFunctions, and an order that is half an odd
# integer is written with elementary functions, as are negative whole orders with positive ones.
# SpecialFunctions has no Struve functions: `Giac.struveh` and `Giac.struvel` are defined here, with
# their values by quadrature.

const _BESSEL = ("besselj", "bessely", "besseli", "besselk", "hankelh1", "hankelh2")
const _AIRY = ("airyai", "airybi", "airyaiprime", "airybiprime")
const _STRUVE = ("struveh", "struvel")
# the functions of one argument, analytic everywhere; erfi, which GIAC does not know, writes the
# erf of an imaginary argument: erf(iu) = i erfi(u)
const _ENTIRE = (_AIRY..., "erfi")

# the Julia function of the name, from SpecialFunctions or, for the Struve functions and the
# polylogarithm, Giac
_special_module(f::AbstractString) = f in _STRUVE || f == "polylog" ? (@__MODULE__) : SpecialFunctions

_special(f::String, args::Node...) = CallNode("of", Node[SymNode(f), args...])

# the name of a call of one of the functions, or nothing
_special_name(n::Node) = nothing
function _special_name(n::CallNode)
    (n.f == "of" && length(n.args) >= 2 && n.args[1] isa SymNode) || return nothing
    f = n.args[1].name
    f in _BESSEL && length(n.args) == 3 && return f
    f in _ENTIRE && length(n.args) == 2 && return f
    f in _STRUVE && length(n.args) == 3 && return f
    f == "polylog" && length(n.args) == 3 && return f
    return nothing
end

_has_special(n::Node) = _any_node(m -> _special_name(m) !== nothing, n)
_has_special(e::GiacExpr) = _has_special(_node(e))

# constant, with a numeric value: a Bessel or Airy function at numbers is, though GIAC sees its name
_has_value(e::GiacExpr) = is_constant(e) || (_has_special(e) && isempty(_free_names(e)))

# the value of a constant with the functions in it, computed in Julia to a Float64's precision;
# nothing for any other expression, or one Julia has no code for
function _special_constant_value(e::GiacExpr)
    (_has_special(e) && isempty(_free_names(e))) || return nothing
    try
        return _compile(e, ())()
    catch err
        err isa Union{ArgumentError, DomainError} || rethrow()
        return nothing
    end
end

for f in Symbol.(_BESSEL)
    @eval SpecialFunctions.$f(ν::GiacExpr, x::GiacExpr) = GiacExpr(_bessel($(string(f)), _node(ν), _node(x)))
    @eval SpecialFunctions.$f(ν::Number, x::GiacExpr) = SpecialFunctions.$f(convert(GiacExpr, ν), x)
    @eval SpecialFunctions.$f(ν::GiacExpr, x::Number) = SpecialFunctions.$f(ν, convert(GiacExpr, x))
end
for f in Symbol.(_AIRY)
    @eval SpecialFunctions.$f(x::GiacExpr) = GiacExpr(_airy($(string(f)), _node(x)))
end
SpecialFunctions.erfi(x::GiacExpr) = GiacExpr(_erfi(_node(x)))

# SpecialFunctions' functions GIAC has, by GIAC's names, for code written with SpecialFunctions
for (f, g) in ((:erf, :erf), (:erfc, :erfc), (:gamma, :Gamma), (:sinint, :Si), (:cosint, :Ci),
               (:expinti, :Ei), (:digamma, :Psi), (:zeta, :Zeta))
    @eval SpecialFunctions.$f(x::GiacExpr) = invoke_cmd($(QuoteNode(g)), x)
end

# erfi(u), odd: -erfi(-u) for u with a negative coefficient, 0 at 0, and the value at a float
function _erfi(u::Node)
    _is_zero_node(u) && return _ZERO
    if _inexact(u)
        v = _special_value("erfi", u.v)
        v === nothing || return NumNode(v)
    end
    c = u isa NumNode ? u.v : u isa MulNode ? u.coef : 1
    c isa Real && c < 0 && return _neg(_special("erfi", _neg(u)))
    return _special("erfi", u)
end

# e with each erf of an imaginary multiple of u, erf(ibu), written i erfi(bu), and erf(-u) as -erf(u)
_erfi_form(e::GiacExpr) = _any_node(m -> m isa CallNode && m.f == "erf", _node(e)) ?
    GiacExpr(_replace_nodes(_node(e)) do m
        (m isa CallNode && m.f == "erf" && length(m.args) == 1) || return nothing
        u = m.args[1]
        c = u isa NumNode ? u.v : u isa MulNode ? u.coef : nothing
        c isa Real && c < 0 && return _neg(CallNode("erf", Node[_neg(u)]))
        (c isa Complex && iszero(real(c)) && !iszero(imag(c))) || return nothing
        return _mul(NumNode(im), _erfi(_node(GiacExpr(u) / convert(GiacExpr, im))))
    end) : e

"""
    struveh(ν, x)
    struvel(ν, x)

The Struve function H_ν(x) and the modified Struve function L_ν(x). At real numbers they are
numbers, from the integral
`H_ν(x) = 2(x/2)^ν/(√π Γ(ν+1/2)) ∫₀¹ (1-t²)^(ν-1/2) sin(xt) dt` (`sinh` for L_ν). With a Giac
expression they are symbolic, as Bessel functions are: half-integer orders are written with
elementary functions, `struveh(1//2, x)` as `√(2/(πx)) (1 - cos x)`.
"""
function struveh end
@doc (@doc struveh) function struvel end

for f in Symbol.(_STRUVE)
    @eval $f(ν::Real, x::Real) = _struve_number($(string(f)), ν, x)
    @eval $f(ν::GiacExpr, x::GiacExpr) = GiacExpr(_struve($(string(f)), _node(ν), _node(x)))
    @eval $f(ν::Number, x::GiacExpr) = $f(convert(GiacExpr, ν), x)
    @eval $f(ν::GiacExpr, x::Number) = $f(ν, convert(GiacExpr, x))
end

# H_ν(x) or L_ν(x): half-integer and negative whole orders from the recurrences below, 0 at 0
# for a whole order, and the value at a float
function _struve(f::String, ν::Node, x::Node)
    if _exact_number(ν) && (denominator(ν.v) == 2 || ν.v isa Integer && ν.v < 0)
        return _node(_struve_order(f, Rational{Int}(ν.v), GiacExpr(x)))
    elseif _exact_number(ν) && ν.v isa Integer && _is_zero_node(x)
        return _ZERO
    elseif ν isa NumNode && x isa NumNode && (_inexact(ν) || _inexact(x))
        v = _special_value(f, ν.v, x.v)
        v === nothing || return NumNode(v)
    end
    return _special(f, ν, x)
end

# orders ±1/2 in closed form, and the others from orders 0 and 1 or -1/2 and 1/2 by
#     H_(m+1) = (2m/x) H_m - H_(m-1) + c_m,   L_(m+1) = L_(m-1) - (2m/x) L_m - c_m,
# c_m = (x/2)^m/(√π Γ(m+3/2)), run upward or downward
function _struve_order(f::String, ν::Rational, x::GiacExpr)
    π_ = convert(GiacExpr, π)
    H = f == "struveh"
    c(m) = (x / 2)^m / (sqrt(π_) * invoke_cmd(:Gamma, m + 3 // 2))
    if denominator(ν) == 2
        r = sqrt(2 / (π_ * x))
        lo, hi = H ? (r * sin(x), r * (1 - cos(x))) : (r * sinh(x), r * (cosh(x) - 1))
        base = -1 // 2
    else
        lo, hi = GiacExpr(_special(f, _ZERO, _node(x))), GiacExpr(_special(f, _ONE, _node(x)))
        base = 0 // 1
    end
    ν == base && return lo
    ν == base + 1 && return hi
    if ν > base + 1
        a, b, m = lo, hi, base + 1                 # orders m - 1 and m
        while m < ν
            a, b, m = b, H ? 2m / x * b - a + c(m) : a - 2m / x * b - c(m), m + 1
        end
        return _expand_cmd(b)
    else
        a, b, m = lo, hi, base                     # orders m and m + 1
        while m > ν
            a, b, m = H ? 2m / x * a - b + c(m) : b + 2m / x * a + c(m), a, m - 1
        end
        return _expand_cmd(a)
    end
end

_inexact(n::Node) = n isa NumNode && (n.v isa AbstractFloat || n.v isa Complex{<:AbstractFloat})
_exact_number(n::Node) = n isa NumNode && n.v isa Union{Integer, Rational}

# f_ν(x), written with elementary functions where ν is half an odd integer, with a positive order
# where ν is a negative whole number, and as its value at a float
function _bessel(f::String, ν::Node, x::Node)
    if _exact_number(ν) && denominator(ν.v) == 2 && f != "hankelh1" && f != "hankelh2"
        return _node(_half_order(f, ν.v, GiacExpr(x)))
    elseif f in ("hankelh1", "hankelh2") && _exact_number(ν) && denominator(ν.v) == 2
        J, Y = _half_order("besselj", ν.v, GiacExpr(x)), _half_order("bessely", ν.v, GiacExpr(x))
        i = convert(GiacExpr, im)
        return _node(f == "hankelh1" ? J + i * Y : J - i * Y)
    elseif _exact_number(ν) && ν.v isa Integer && ν.v < 0
        n = -ν.v
        f == "besselk" && return _bessel(f, NumNode(n), x)
        f == "besseli" && return _bessel(f, NumNode(n), x)
        return _mul(NumNode(isodd(n) ? -1 : 1), _bessel(f, NumNode(n), x))
    elseif _exact_number(ν) && ν.v isa Integer && _is_zero_node(x) && f in ("besselj", "besseli")
        return iszero(ν.v) ? _ONE : _ZERO
    elseif ν isa NumNode && x isa NumNode && (_inexact(ν) || _inexact(x))
        v = _special_value(f, ν.v, x.v)
        v === nothing || return NumNode(v)
    end
    return _special(f, ν, x)
end

# Ai(0) = 1/(3^(2/3) Γ(2/3)), Ai'(0) = -1/(3^(1/3) Γ(1/3)), Bi(0) = 1/(3^(1/6) Γ(2/3)),
# Bi'(0) = 3^(1/6)/Γ(1/3)
function _airy(f::String, x::Node)
    if _is_zero_node(x)
        three, Γ = convert(GiacExpr, 3), p -> invoke_cmd(:Gamma, p)
        v = f == "airyai" ? 1 / (three^(2 // 3) * Γ(2 // 3)) : f == "airyaiprime" ? -1 / (three^(1 // 3) * Γ(1 // 3)) :
            f == "airybi" ? 1 / (three^(1 // 6) * Γ(2 // 3)) : three^(1 // 6) / Γ(1 // 3)
        return _node(v)
    end
    if _inexact(x)
        v = _special_value(f, x.v)
        v === nothing || return NumNode(v)
    end
    return _special(f, x)
end

# the value from SpecialFunctions, real where the function is real at a real argument
function _special_value(f::String, args::Number...)
    if f in _STRUVE
        all(a -> a isa Real, args) || return nothing
        return _struve_number(f, args...)
    end
    if f == "polylog"
        s, z = args
        (s isa Integer || s isa Real && isinteger(s)) || return nothing
        return _polylog_number(Int(s), z)
    end
    fn = getfield(SpecialFunctions, Symbol(f))
    v = try
        fn(map(a -> a isa Rational || a isa Integer ? float(a) : a, args)...)
    catch e
        e isa Union{DomainError, ArgumentError, SpecialFunctions.AmosException} || rethrow()
        try
            fn(map(a -> complex(float(a)), args)...)
        catch e
            e isa Union{DomainError, ArgumentError, SpecialFunctions.AmosException} || rethrow()
            return nothing
        end
    end
    v isa Complex && iszero(imag(v)) && !(f in ("hankelh1", "hankelh2")) && return real(v)
    return v
end

# orders ±1/2 in closed form, and the others from the recurrences
#     J_(ν+1) = (2ν/x) J_ν - J_(ν-1),   I_(ν+1) = I_(ν-1) - (2ν/x) I_ν,   K_(ν+1) = K_(ν-1) + (2ν/x) K_ν
function _half_order(f::String, ν::Rational, x::GiacExpr)
    π_ = convert(GiacExpr, π)
    r = sqrt(2 / (π_ * x))
    lo, hi = f == "besselj" ? (r * cos(x), r * sin(x)) :          # orders -1/2 and 1/2
             f == "bessely" ? (r * sin(x), -r * cos(x)) :
             f == "besseli" ? (r * cosh(x), r * sinh(x)) :
             (sqrt(π_ / (2x)) * exp(-x), sqrt(π_ / (2x)) * exp(-x))
    step(m, a, b) = f == "besseli" ? a - 2m / x * b : f == "besselk" ? a + 2m / x * b : 2m / x * b - a
    down(m, a, b) = f == "besseli" ? b + 2m / x * a : f == "besselk" ? b - 2m / x * a : 2m / x * a - b
    ν == 1 // 2 && return hi
    ν == -1 // 2 && return lo
    if ν > 0
        a, b, m = lo, hi, 1 // 2                   # orders m - 1 and m
        while m < ν
            a, b, m = b, step(m, a, b), m + 1
        end
        return _expand_cmd(b)
    else
        a, b, m = lo, hi, -1 // 2                  # orders m and m + 1
        while m > ν
            a, b, m = down(m, a, b), a, m - 1
        end
        return _expand_cmd(a)
    end
end

# d/dx of f(x) for the function as called: (J_(ν-1) - J_(ν+1))/2, and Ai' for Ai, x Ai for Ai'
function _special_derivative(n::CallNode, x::String)
    parts = _rootsum_parts(n)
    parts === nothing || return _rootsum_derivative(parts..., x)
    f = _special_name(n)
    f === nothing && return x in _symbol_names(n) ? nothing : _ZERO
    if f == "erfi"                                      # 2 e^(u²)/√π
        u = n.args[2]
        du = _derivative(u, x)
        (du === nothing || _is_zero_node(du)) && return du
        return _mul(du, _node(2 * exp(GiacExpr(u)^2) / sqrt(convert(GiacExpr, π))))
    end
    if f in _AIRY
        u = n.args[2]
        du = _derivative(u, x)
        (du === nothing || _is_zero_node(du)) && return du
        d = f == "airyai" ? _special("airyaiprime", u) : f == "airybi" ? _special("airybiprime", u) :
            _mul(u, _special(f == "airyaiprime" ? "airyai" : "airybi", u))
        return _mul(du, d)
    end
    ν, u = n.args[2], n.args[3]
    x in _symbol_names(ν) && return nothing
    du = _derivative(u, x)
    (du === nothing || _is_zero_node(du)) && return du
    if f == "polylog"                                   # Li_s'(u) = Li_(s-1)(u)/u
        return _mul(du, _node(GiacExpr(_polylog(_add(ν, _MINUS_ONE), u)) / GiacExpr(u)))
    end
    if f in _STRUVE
        # H_ν' = (ν/u) H_ν - H_(ν+1) + c_ν and L_ν' = (ν/u) L_ν + L_(ν+1) + c_ν, from (u^(-ν) H_ν)'
        V, U = GiacExpr(ν), GiacExpr(u)
        c = (U / 2)^V / (sqrt(convert(GiacExpr, π)) * invoke_cmd(:Gamma, V + 3 // 2))
        next = GiacExpr(_struve(f, _add(ν, _ONE), u))
        d = V / U * GiacExpr(n) + (f == "struveh" ? -next : next) + c
        return _mul(du, _node(d))
    end
    below, above = _bessel(f, _add(ν, _MINUS_ONE), u), _bessel(f, _add(ν, _ONE), u)
    d = _is_zero_node(ν) ? (f == "besseli" ? _bessel(f, _ONE, u) : _neg(_bessel(f, _ONE, u))) :
        f == "besseli" ? _mul(NumNode(1 // 2), _add(below, above)) :
        f == "besselk" ? _mul(NumNode(-1 // 2), _add(below, above)) :
        _mul(NumNode(1 // 2), _add(below, _neg(above)))
    return _mul(du, d)
end

# `evalf` of an expression with the functions in it: each at numbers replaced by its value first.
# A value that is the whole expression is returned as it is: GIAC's evalf of a double moves it by
# more than its rounding.
function _evalf_special(e::GiacExpr, rest...)
    v = _special_values(_node(e))
    _inexact(v) && return GiacExpr(v)
    return _invoke_cmd_direct(:evalf, (GiacExpr(v), rest...))
end

_special_values(n::Node) = _replace_nodes(n) do m
    parts = _rootsum_parts(m)
    if parts !== nothing
        v = _rootsum_value(parts...)
        return v === nothing ? nothing : NumNode(v)
    end
    f = _special_name(m)
    f === nothing && return nothing
    args = Node[_node(_invoke_cmd_direct(:evalf, (GiacExpr(_special_values(a)),))) for a in m.args[2:end]]
    all(a -> a isa NumNode, args) || return nothing
    v = _special_value(f, (a.v for a in args)...)
    return v === nothing ? nothing : NumNode(v)
end

# `simplify`, also with the Bessel functions of three or more orders in a family written in the
# two lowest where that is simpler (trig_simplify.jl for the rest)
function _simplify_cmd(e::GiacExpr)
    e = _rootsums_summed(e)
    r = _trig_safe_simplify(e)
    c = _contiguous(e)
    c === nothing && return r
    q = _trig_safe_simplify(c)
    return Rules.expression_size(q) < Rules.expression_size(r) ? q : r
end

# The contiguous relations: in a family f(ν + k, u) with whole k, of three orders or more, each
# order written in the two lowest from
#     J_(m+1) = (2m/u) J_m - J_(m-1),  I_(m+1) = I_(m-1) - (2m/u) I_m,  K_(m+1) = K_(m-1) + (2m/u) K_m
# and the same for Y and the Hankel functions as for J; nothing where there is no such family
function _contiguous(e::GiacExpr)
    calls = Node[]
    _any_node(_node(e)) do m
        f = _special_name(m)
        f !== nothing && f in _BESSEL && !(m in calls) && push!(calls, m)
        return false
    end
    families = Vector{Vector{Tuple{Int, CallNode}}}()
    for m in calls
        family = findfirst(families) do fam
            r = fam[1][2]
            r.args[1] == m.args[1] && r.args[3] == m.args[3] && _whole_difference(m.args[2], r.args[2]) !== nothing
        end
        if family === nothing
            push!(families, [(0, m)])
        else
            fam = families[family]
            push!(fam, (fam[1][1] + _whole_difference(m.args[2], fam[1][2].args[2]), m))
        end
    end
    replacements = Dict{Node, Node}()
    for fam in families
        length(unique(first.(fam))) >= 3 || continue
        lo = minimum(first, fam)
        f, ν0, u = fam[1][2].args[1].name, fam[1][2].args[2], GiacExpr(fam[1][2].args[3])
        base = GiacExpr(ν0) + (lo - fam[1][1])
        R = Dict(0 => GiacExpr(_bessel(f, _node(base), _node(u))), 1 => GiacExpr(_bessel(f, _node(base + 1), _node(u))))
        for k in 2:(maximum(first, fam) - lo)
            m = base + (k - 1)
            R[k] = f == "besseli" ? R[k - 2] - 2m / u * R[k - 1] :
                   f == "besselk" ? R[k - 2] + 2m / u * R[k - 1] : 2m / u * R[k - 1] - R[k - 2]
        end
        for (k, m) in fam
            k - lo >= 2 && (replacements[m] = _node(R[k - lo]))
        end
    end
    isempty(replacements) && return nothing
    return GiacExpr(_replace_nodes(m -> get(replacements, m, nothing), _node(e)))
end

# a - b where that is a whole number
function _whole_difference(a::Node, b::Node)
    d = _node(GiacExpr(a) - GiacExpr(b))
    return d isa NumNode && d.v isa Integer ? Int(d.v) : nothing
end

# ---------------------------------------------------------------------------------------------
# Antiderivatives
# ---------------------------------------------------------------------------------------------

# ∫ f dx term by term where f has the functions in it: theirs from the rules below, the others by
# the whole integration; nothing where a term is not integrated
function _special_terms_integral(f::GiacExpr, x::GiacExpr)
    _has_special(f) || return nothing
    parts = GiacExpr[]
    for m in _terms(_node(f))
        g = _has_special(m) ? _special_integral(GiacExpr(m), x) : _integrate_or_nothing(GiacExpr(m), x)
        g === nothing && return nothing
        push!(parts, g)
    end
    return sum(parts)
end

# ∫ c xᵖ Z_ν(kx) dx, with p - ν even ending in Struve functions (_struve_antiderivative), from
#     (x^μ Z_μ(kx))' = k x^μ Z_(μ-1)(kx),   (x^(-μ) Z_μ(kx))' = -k x^(-μ) Z_(μ+1)(kx)
# for Z = J, Y, H⁽¹⁾, H⁽²⁾ (I takes + in the second, K - in the first), by parts down to one of
# them: where p - ν is odd and positive, ∫ xᵖ Z_ν = xᵖ Z_(ν+1)/k - (p-ν-1)/k ∫ xᵖ⁻¹ Z_(ν+1),
# and where p + ν is, ∫ xᵖ Z_ν = -xᵖ Z_(ν-1)/k + (p+ν-1)/k ∫ xᵖ⁻¹ Z_(ν-1). Airy functions:
# ∫ Ai' = Ai and ∫ x Ai = Ai'. Nothing for other integrands.
function _special_integral(f::GiacExpr, x::GiacExpr)
    n = _node(f)
    c, rest = n isa MulNode ? (GiacExpr(NumNode(n.coef)), n.factors) : (one(GiacExpr), Pair{Node, Node}[n => _ONE])
    p, call = 0, nothing
    for (b, e) in rest
        factor = _pow(b, e)
        if !(string(x) in _symbol_names(factor))
            c *= GiacExpr(factor)
        elseif b isa SymNode && b.name == string(x) && e isa NumNode && e.v isa Integer
            p += e.v
        elseif call === nothing && _is_one_node(e) && _special_name(b) !== nothing
            call = b
        else
            return nothing
        end
    end
    call === nothing && return nothing
    g = _special_antiderivative(_special_name(call), call, p, x)
    return g === nothing ? nothing : c * g
end

_is_one_node(e::Node) = e isa NumNode && isone(e.v)

function _special_antiderivative(f::String, call::CallNode, p::Integer, x::GiacExpr)
    name = string(x)
    f == "erfi" && return nothing
    if f in _AIRY
        u = call.args[2]
        (u isa SymNode && u.name == name) || return nothing
        f == "airyaiprime" && p == 0 && return GiacExpr(_special("airyai", u))
        f == "airybiprime" && p == 0 && return GiacExpr(_special("airybi", u))
        f == "airyai" && p == 1 && return GiacExpr(_special("airyaiprime", u))
        f == "airybi" && p == 1 && return GiacExpr(_special("airybiprime", u))
        return nothing
    end
    ν, u = GiacExpr(call.args[2]), call.args[3]
    _depends_on(ν, x) && return nothing
    (_linear_in(u, name) && _is_zero_node(_node(substitute(GiacExpr(u), x => 0)))) || return nothing
    return _bessel_antiderivative(f, ν, u, GiacExpr(_derivative(u, name)), p, x)
end

function _bessel_antiderivative(f::String, ν::GiacExpr, u::Node, k::GiacExpr, p::Integer, x::GiacExpr)
    Z(m) = GiacExpr(_bessel(f, _node(m), u))
    up = _whole_difference(_node(p - ν), _ZERO)          # p - ν
    down = _whole_difference(_node(p + ν), _ZERO)        # p + ν
    if up !== nothing && up >= 0 && (isodd(up) || f != "besselk")
        up == 0 && return _struve_antiderivative(f, ν, u, k, x)
        sign = f == "besselk" ? -1 : 1                   # (x^μ Z_μ)' = sign k x^μ Z_(μ-1)
        rest = up == 1 ? zero(GiacExpr) : _bessel_antiderivative(f, ν + 1, u, k, p - 1, x)
        rest === nothing && return nothing
        return sign * (x^p * Z(ν + 1) - (up - 1) * rest) / k
    elseif down !== nothing && down > 0 && isodd(down)
        sign = f == "besseli" ? 1 : -1                   # (x^(-μ) Z_μ)' = sign k x^(-μ) Z_(μ+1)
        rest = down == 1 ? zero(GiacExpr) : _bessel_antiderivative(f, ν - 1, u, k, p - 1, x)
        rest === nothing && return nothing
        return sign * (x^p * Z(ν - 1) - (down - 1) * rest) / k
    end
    return nothing
end

# ∫ x^ν Z_ν(kx) dx = √π 2^(ν-1) Γ(ν+1/2) x (Z_ν H_(ν-1) - Z_(ν-1) H_ν)(kx) / k^ν, for Z = J, Y
# and the Hankel functions with the Struve function H, and for I with L (DLMF 10.22.2, 10.43.2)
function _struve_antiderivative(f::String, ν::GiacExpr, u::Node, k::GiacExpr, x::GiacExpr)
    f == "besselk" && return nothing
    S = f == "besseli" ? "struvel" : "struveh"
    Z(m) = GiacExpr(_bessel(f, _node(m), u))
    W(m) = GiacExpr(_struve(S, _node(m), u))
    C = sqrt(convert(GiacExpr, π)) * 2^(ν - 1) * invoke_cmd(:Gamma, ν + 1 // 2)
    return C * x * (Z(ν) * W(ν - 1) - Z(ν - 1) * W(ν)) / k^ν
end

# ---------------------------------------------------------------------------------------------
# Equations
# ---------------------------------------------------------------------------------------------

# The general solution of y'' + P y' + Q y = 0 where it is of Bessel's or Airy's form: with
# a = (1 - xP)/2 constant and x²Q = A + B x^m,
#     y = x^a (C₁ Z_ν(b x^c) + C₂ W_ν(b x^c)),   c = m/2, b = √|B|/c, ν = √(a² - A)/c,
# Z, W = J, Y where B > 0 and I, K where B < 0; and for P = 0, Q = -k x, y = C₁ Ai(k^(1/3) x) +
# C₂ Bi(k^(1/3) x). Nothing for other equations.
function _bessel_equation(eq::GiacExpr, x::GiacExpr, y::GiacExpr)
    c = _linear_coefficients(eq, x, y)
    (c === nothing || length(c.q) != 3 || !_is_zero_expr(c.f)) && return nothing
    P = invoke_cmd(:simplify, c.q[2] / c.q[3])
    Q = invoke_cmd(:simplify, c.q[1] / c.q[3])
    C1, C2 = GiacExpr(SymNode("c_0")), GiacExpr(SymNode("c_1"))
    if _is_zero_expr(P)
        k = -invoke_cmd(:simplify, Q / x)
        if !_depends_on(k, x)
            r = cbrt(k)
            return C1 * SpecialFunctions.airyai(r * x) + C2 * SpecialFunctions.airybi(r * x)
        end
    end
    a = invoke_cmd(:simplify, (1 - x * P) / 2)
    _depends_on(a, x) && return nothing
    X = invoke_cmd(:expand, x^2 * Q)
    A = substitute(X, x => 0)
    rest = invoke_cmd(:simplify, X - A)
    m = _monomial_degree(rest, x)
    (m === nothing || iszero(m)) && return nothing
    B = invoke_cmd(:simplify, rest / x^m)
    half = GiacExpr(NumNode(m // 2))
    positive = _positive_coefficient(B)
    b = _square_root(positive ? B : -B) / abs(half)
    ν = _square_root(invoke_cmd(:simplify, (a^2 - A) / half^2))
    u = b * x^half
    Z, W = positive ? (SpecialFunctions.besselj, SpecialFunctions.bessely) : (SpecialFunctions.besseli, SpecialFunctions.besselk)
    return x^a * (C1 * Z(ν, u) + C2 * W(ν, u))
end

# √e with the even powers in it halved, n for n², as an order or a scale may be taken either sign
function _square_root(e::GiacExpr)
    n = _node(e)
    if n isa MulNode && n.coef isa Union{Integer, Rational} && n.coef > 0 &&
       all(((b, k),) -> k isa NumNode && k.v isa Integer && iseven(k.v), n.factors)
        r = sqrt(GiacExpr(NumNode(n.coef)))
        return r * GiacExpr(_mul((_pow(b, NumNode(k.v ÷ 2)) for (b, k) in n.factors)...))
    end
    return sqrt(e)
end

# the exponent m of a monomial B xᵐ in x, B free of x; nothing for anything else
function _monomial_degree(e::GiacExpr, x::GiacExpr)
    n = _node(e)
    n isa SymNode && n.name == string(x) && return 1
    n isa MulNode || return nothing
    m = nothing
    for (b, k) in n.factors
        if b isa SymNode && b.name == string(x)
            (k isa NumNode && k.v isa Union{Integer, Rational}) || return nothing
            m = k.v
        elseif string(x) in _symbol_names(_pow(b, k))
            return nothing
        end
    end
    return m
end

# ---------------------------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------------------------

# The Taylor polynomial of degree n of f in x at a, from its derivatives on the tree, where f has
# the functions in it and they are analytic at a; nothing at a point where a Bessel function of
# the second kind, a K or a Hankel function, or a J or I of an order that is not whole, is not
function _special_taylor(f::GiacExpr, x::GiacExpr, a, n::Integer)
    _has_special(f) || return nothing
    name = string(x)
    terms = GiacExpr[]
    g = _node(f)
    for k in 0:n
        v = GiacExpr(_node(substitute(GiacExpr(g), x => a)))
        (_singular_special(_node(v)) || !_is_finite_value(v)) && return nothing
        push!(terms, v * (x - a)^k / factorial(big(k)))
        k == n && break
        g = _derivative(g, name)
        g === nothing && return nothing
    end
    return sum(terms)
end

# a call of the functions where it is not analytic: at 0, of Y, K or a Hankel function, or of J
# or I of an order that is not a whole number
_singular_special(n::Node) = _any_node(n) do m
    f = _special_name(m)
    (f === nothing || f in _ENTIRE) && return false
    _is_zero_node(m.args[3]) || return false
    f in ("bessely", "besselk", "hankelh1", "hankelh2") && return true
    ν = m.args[2]
    return !(ν isa NumNode && ν.v isa Integer)
end

# the functions in a tree read again through their rules, as after a substitution: J₀(0) = 1,
# J_(1/2) in sin, a float argument to its value
_rebuilt_specials(e::GiacExpr) = _has_special(e) ? GiacExpr(_rebuilt_specials(_node(e))) : e
_rebuilt_specials(n::Node) = _replace_nodes(n) do m
    f = _special_name(m)
    f === nothing && return nothing
    args = Node[_rebuilt_specials(a) for a in m.args[2:end]]
    return f == "erfi" ? _erfi(args[1]) : f in _AIRY ? _airy(f, args[1]) : f in _STRUVE ? _struve(f, args[1], args[2]) :
           f == "polylog" ? _polylog(args[1], args[2]) : _bessel(f, args[1], args[2])
end

# `series(f, x, a, n)` of an expression with the functions in it, as GIAC writes a series: the
# Taylor polynomial and (x - a)ⁿ⁺¹ order_size(x - a); nothing for GIAC to answer
function _special_series(f, x, a, n)
    (f isa GiacExpr && x isa GiacExpr && _node(x) isa SymNode && n isa Integer && n >= 0) || return nothing
    calls = _special_calls(_node(f))
    any(m -> _vanishes_at(m, x, a), calls) && return _series_by_expansion(f, calls, x, a, n)
    p = _special_taylor(f, x, a, n)
    p === nothing && return nothing
    h = x - a
    return p + h^(n + 1) * invoke_cmd(:order_size, h)
end

_special_calls(n::Node) = (calls = CallNode[]; _any_node(m -> (_special_name(m) === nothing || m in calls || push!(calls, m); false), n); calls)

# whether the argument of a Bessel or Struve function call is 0 at x = a
function _vanishes_at(m::CallNode, x::GiacExpr, a)
    _special_name(m) in _ENTIRE && return false
    return _is_zero_node(_node(substitute(GiacExpr(m.args[3]), x => a)))
end

# GIAC's series of f with each of the functions in it written out: at a zero of its argument as
# its expansion there (_expansion_at_zero), elsewhere as its Taylor polynomial, both to a degree
# past n by enough for the negative powers of Y and K; nothing where one cannot be
function _series_by_expansion(f::GiacExpr, calls, x::GiacExpr, a, n::Integer)
    orders = [_approx_value(GiacExpr(m.args[2])) for m in calls if !(_special_name(m) in _ENTIRE)]
    N = n + 2 + 2 * min(10, maximum((abs(v) for v in orders if v isa Real); init = 0) |> ceil |> Int)
    replacements = Dict{Node, Node}()
    for m in calls
        any(_has_special, m.args[2:end]) && return nothing
        E = if _vanishes_at(m, x, a)
            _expansion_at_zero(_special_name(m), GiacExpr(m.args[2]), GiacExpr(m.args[3]), N)
        else
            _special_taylor(GiacExpr(m), x, a, N)
        end
        E === nothing && return nothing
        replacements[m] = _node(E)
    end
    g = GiacExpr(_replace_nodes(m -> get(replacements, m, nothing), _node(f)))
    r = try
        _invoke_cmd_direct(:series, (g, x, convert(GiacExpr, a), convert(GiacExpr, n)))
    catch e
        _stopped(e) && rethrow()
        return nothing
    end
    return _unevaluated(r, "series") ? nothing : r
end

# ---------------------------------------------------------------------------------------------
# Definite integrals
# ---------------------------------------------------------------------------------------------

# ∫ₐᵇ f dx for f with the functions in it: on [0, ∞) the limit at s → 0⁺ of its Laplace
# transform, where the integral converges (_bessel_integral_converges); elsewhere F(b) - F(a) for
# the antiderivative F, where both are finite and f has no singularity at 0 between a and b
function _special_definite_integral(f::GiacExpr, x::GiacExpr, a, b)
    (_has_special(f) && _node(x) isa SymNode) || return nothing
    A, B = convert(GiacExpr, a), convert(GiacExpr, b)
    if _is_zero_node(_node(A)) && _is_plus_infinity(_node(B))
        _bessel_integral_converges(f, x) || return nothing
        s = _fresh_symbol(_node(f), "s")
        F = invoke_cmd(:laplace, f, x, s)
        _has_transform_call(_node(F)) && return nothing
        v = _giac_or_nothing(:limit, F, s, 0, 1)
        (v === nothing || !_is_finite_value(v) || _unevaluated(v, "limit")) && return nothing
        return invoke_cmd(:simplify, v)
    end
    (_is_finite_value(A) && _is_finite_value(B)) || return nothing
    F = _special_terms_integral(f, x)
    F === nothing && return nothing
    lo, hi = _approx_value(A), _approx_value(B)
    inside = !(lo isa Real && hi isa Real) || min(lo, hi) <= 0 <= max(lo, hi)
    if inside
        f0 = substitute(f, x => 0)
        (_is_finite_value(f0) && !_singular_special(_node(f0))) || return nothing
    end
    Fa, Fb = substitute(F, x => A), substitute(F, x => B)
    for v in (Fa, Fb)
        (_is_finite_value(v) && !_singular_special(_node(v))) || return nothing
    end
    return _smaller_form(Fb - Fa)
end

# whether ∫₀^∞ c xᵖ e^(-αx) Z_ν(kx) dx converges, for numbers p, α, ν and k > 0: at 0 where
# p + ν > -1 for J and I, p - |ν| > -1 for Y, K and the Hankel functions (p > -1 at ν = 0); at ∞,
# as J, Y and the Hankel functions fall off as x^(-1/2) with oscillation, where α > 0 or α = 0 and
# p < 1/2, and as I and K grow and fall as e^(±kx) x^(-1/2), where α > k (α > -k for K), or at
# equality p < -1/2
function _bessel_integral_converges(f::GiacExpr, x::GiacExpr)
    c, a, g = _exponential_factor(f, x)
    m = _power_and_bessel(g, x)
    m === nothing && return false
    p, Z, ν, u = m
    shape = _argument_shape(u, x)
    (shape !== nothing && shape[1] === :linear) || return false
    vals = map(_approx_value, (-a, ν, shape[2]))
    all(v -> v isa Real, vals) || return false
    α, ν, k = vals
    k > 0 || return false
    p = float(p)
    at0 = Z in ("besselj", "besseli") ? p + ν > -1 : iszero(ν) ? p > -1 : p - abs(ν) > -1
    at∞ = Z == "besseli" ? (α > k || α == k && p < -1 // 2) :
          Z == "besselk" ? (α > -k || α == -k && p < -1 // 2) :
          (α > 0 || α == 0 && p < 1 // 2)
    return at0 && at∞
end

# H_ν(x) and L_ν(x) at real numbers, for ν ≥ 1/2 from
#     H_ν(x) = 2(x/2)^ν/(√π Γ(ν+1/2)) ∫₀¹ (1-t²)^(ν-1/2) sin(xt) dt
# (sinh for L_ν) by tanh-sinh quadrature, where the integrand is bounded, with steps fine enough
# for the oscillation, and lower
# orders from the recurrences
#     H_(ν-1) = (2ν/x) H_ν - H_(ν+1) + (x/2)^ν/(√π Γ(ν+3/2)),
#     L_(ν-1) = L_(ν+1) + (2ν/x) L_ν + (x/2)^ν/(√π Γ(ν+3/2))
function _struve_number(f::String, ν::Real, x::Real)
    ν, x = float(ν), float(x)
    if ν < 1 / 2
        a, b = _struve_number(f, ν + 1, x), _struve_number(f, ν + 2, x)
        m = ν + 1
        term = (x / 2)^m / (sqrt(π) * SpecialFunctions.gamma(m + 3 / 2))
        return f == "struveh" ? 2m / x * a - b + term : b + 2m / x * a + term
    end
    iszero(x) && return 0.0
    h = 1 / (32 * max(1, ceil(Int, abs(x) / 30)))
    total = 0.0
    for u in -4.5:h:4.5
        w = π / 2 * sinh(u)
        t, s = 1 / (1 + exp(-2w)), 1 / (1 + exp(2w))      # t and 1 - t
        dt = π / 4 * cosh(u) / cosh(w)^2
        g = (s * (1 + t))^(ν - 1 / 2) * (f == "struveh" ? sin(x * t) : sinh(x * t))
        isfinite(g) && (total += g * dt)
    end
    return 2 * (x / 2)^ν / (sqrt(π) * SpecialFunctions.gamma(ν + 1 / 2)) * h * total
end

# ---------------------------------------------------------------------------------------------
# Expansions at a zero of the argument
# ---------------------------------------------------------------------------------------------

# f(ν, u) as the sum of its expansion at u = 0 to the power uᴺ, in the closed forms that hold
# for all u (DLMF 10.2.2, 10.8.1, 10.25.2, 10.31.1, 11.2.1, 11.2.2): power series for J, I and the
# Struve functions, and for Y and K of a whole order n the log terms
#     Y_n = -(u/2)⁻ⁿ/π Σₖ₌₀ⁿ⁻¹ (n-k-1)!/k! (u²/4)ᵏ + (2/π) ln(u/2) J_n
#           - (u/2)ⁿ/π Σₖ (ψ(k+1) + ψ(n+k+1)) (-u²/4)ᵏ/(k!(n+k)!),
#     K_n = (u/2)⁻ⁿ/2 Σₖ₌₀ⁿ⁻¹ (n-k-1)!/k! (-u²/4)ᵏ + (-1)ⁿ⁺¹ ln(u/2) I_n
#           + (-1)ⁿ (u/2)ⁿ/2 Σₖ (ψ(k+1) + ψ(n+k+1)) (u²/4)ᵏ/(k!(n+k)!);
# Y and K of other orders from J_(±ν) and I_(±ν), and the Hankel functions as J ± iY
function _expansion_at_zero(f::String, ν::GiacExpr, u::GiacExpr, N::Integer)
    π_ = convert(GiacExpr, π)
    Γ(z) = invoke_cmd(:Gamma, z)
    v = _approx_value(ν)
    K = v isa Real ? max(0, ceil(Int, (N - v) / 2)) + 1 : cld(N, 2) + 1
    whole = _node(ν) isa NumNode && _node(ν).v isa Integer && _node(ν).v >= 0
    if f == "polylog"                                    # Σₖ uᵏ/kˢ
        return sum(u^k / convert(GiacExpr, k)^ν for k in 1:max(N, 1))
    end
    if f in ("besselj", "besseli")
        sgn(k) = f == "besselj" && isodd(k) ? -1 : 1
        return sum(sgn(k) * (u / 2)^(2k + ν) / (factorial(big(k)) * Γ(k + ν + 1)) for k in 0:K)
    elseif f in _STRUVE
        sgn2(k) = f == "struveh" && isodd(k) ? -1 : 1
        return sum(sgn2(k) * (u / 2)^(2k + ν + 1) / (Γ(convert(GiacExpr, k) + 3 // 2) * Γ(k + ν + 3 // 2)) for k in 0:K)
    elseif f in ("hankelh1", "hankelh2")
        J, Y = _expansion_at_zero("besselj", ν, u, N), _expansion_at_zero("bessely", ν, u, N)
        Y === nothing && return nothing
        i = convert(GiacExpr, im)
        return f == "hankelh1" ? J + i * Y : J - i * Y
    elseif whole
        n = Int(_node(ν).v)
        γ = giac_eval("euler_gamma")
        ψ(m) = -γ + sum((1 // j for j in 1:(m - 1)); init = 0 // 1)          # ψ(m) for whole m ≥ 1
        q = u^2 / 4
        if f == "bessely"
            head = n == 0 ? zero(GiacExpr) :
                   -(u / 2)^(-n) / π_ * sum(factorial(big(n - k - 1)) // factorial(big(k)) * q^k for k in 0:(n - 1))
            tail = -(u / 2)^n / π_ * sum((ψ(k + 1) + ψ(n + k + 1)) * (-q)^k / (factorial(big(k)) * factorial(big(n + k))) for k in 0:K)
            return head + 2 / π_ * log(u / 2) * _expansion_at_zero("besselj", ν, u, N) + tail
        else
            head = n == 0 ? zero(GiacExpr) :
                   (u / 2)^(-n) / 2 * sum(factorial(big(n - k - 1)) // factorial(big(k)) * (-q)^k for k in 0:(n - 1))
            tail = (-1)^n * (u / 2)^n / 2 * sum((ψ(k + 1) + ψ(n + k + 1)) * q^k / (factorial(big(k)) * factorial(big(n + k))) for k in 0:K)
            return head + (-1)^(n + 1) * log(u / 2) * _expansion_at_zero("besseli", ν, u, N) + tail
        end
    elseif v isa Real && isinteger(v)
        return nothing                     # a negative whole order is rewritten with its positive
    else
        s = sin(ν * π_)
        if f == "bessely"
            return (_expansion_at_zero("besselj", ν, u, N) * cos(ν * π_) - _expansion_at_zero("besselj", -ν, u, N)) / s
        else
            return π_ * (_expansion_at_zero("besseli", -ν, u, N) - _expansion_at_zero("besseli", ν, u, N)) / (2s)
        end
    end
end

# ---------------------------------------------------------------------------------------------
# The polylogarithm
# ---------------------------------------------------------------------------------------------

"""
    polylog(s, x)

The polylogarithm Li_s(x) = Σₙ₌₁^∞ xⁿ/nˢ, for a whole order s. At numbers it is a number; with a
Giac expression it is symbolic, written with logarithms for s ≤ 1, and with its values at 0, 1, -1
and ½ (Li₂(½) = π²/12 - ln²2/2) where they are known. A sum `Σ xⁿ/nˢ` is written with it.
"""
polylog(s::Integer, x::Number) = _polylog_number(s, x)
polylog(s::GiacExpr, x::GiacExpr) = GiacExpr(_polylog(_node(s), _node(x)))
polylog(s::Number, x::GiacExpr) = polylog(convert(GiacExpr, s), x)
polylog(s::GiacExpr, x::Number) = polylog(s, convert(GiacExpr, x))

function _polylog(s::Node, x::Node)
    if s isa NumNode && s.v isa Integer
        S, X = Int(s.v), GiacExpr(x)
        if S <= 0                                       # (x d/dx)⁻ˢ x/(1 - x)
            e = X / (1 - X)
            for _ in 1:(-S)
                e = invoke_cmd(:normal, X * invoke_cmd(:diff, e, X))
            end
            return _node(e)
        end
        S == 1 && return _node(-log(1 - X))
        _is_zero_node(x) && return _ZERO
        ζ = invoke_cmd(:Zeta, convert(GiacExpr, S))
        x isa NumNode && isone(x.v) && return _node(ζ)
        x isa NumNode && x.v == -1 && return _node(-(1 - convert(GiacExpr, 2)^(1 - S)) * ζ)
        if x isa NumNode && x.v == 1 // 2 && S in (2, 3)
            π_, l = convert(GiacExpr, π), log(convert(GiacExpr, 2))
            return _node(S == 2 ? π_^2 / 12 - l^2 / 2 : 7ζ / 8 - π_^2 * l / 12 + l^3 / 6)
        end
        if _inexact(x)
            v = _polylog_number(S, x.v)
            v === nothing || return NumNode(v)
        end
    end
    return _special("polylog", s, x)
end

# Li_s(z) at a number: the series for |z| ≤ 1/2; for |ln z| < 5 (< 2π) the expansion in μ = ln z,
#     Li_s(e^μ) = μ^(s-1)/(s-1)! (H_(s-1) - ln(-μ)) + Σ_(k ≠ s-1) ζ(s-k) μᵏ/k!;
# beyond it the inversion Li_s(z) = -(-1)ˢ Li_s(1/z) - (2πi)ˢ/s! B_s(1/2 + ln(-z)/(2πi)). Real for
# real z ≤ 1.
function _polylog_number(s::Integer, x::Number)
    z = complex(float(x))
    v = if s <= 1
        s == 1 ? -log(1 - z) : _polylog_negative(s, z)
    elseif z == 1
        complex(SpecialFunctions.zeta(s))
    elseif abs(z) <= 0.5
        total, term, k = zero(z), z, 1
        while abs(term) > 1e-18 * max(abs(total), 1e-300) && k < 200
            total += term / k^s
            k += 1
            term *= z
        end
        total
    elseif abs(log(z)) < 5
        μ = log(z)
        # -μ + 0i: for real z > 1, -μ is -0.0i below the cut; Im Li_s(z) = -π lnˢ⁻¹z/(s-1)! there
        total = μ^(s - 1) / factorial(s - 1) * (sum(1 / j for j in 1:(s - 1); init = 0.0) - log(-μ + 0.0im))
        small = 0                     # ζ vanishes at the negative even numbers: two small terms end it
        for k in 0:300
            k == s - 1 && continue
            t = SpecialFunctions.zeta(s - k) * μ^k / factorial(big(k))
            total += ComplexF64(t)
            small = abs(t) < 1e-18 * max(abs(total), 1) ? small + 1 : 0
            k > s && small >= 2 && break
        end
        total
    else
        w = log(-z) / (2π * im)
        ComplexF64(-(-1)^s * _polylog_number(s, 1 / z) - (2π * im)^s / factorial(s) * _bernoulli_polynomial(s, 1 / 2 + w))
    end
    return x isa Real && x <= 1 ? real(v) : v
end

_polylog_negative(s::Integer, z) = s == 0 ? z / (1 - z) :
    sum(_eulerian(-s, k) * z^(k + 1) for k in 0:(-s - 1)) / (1 - z)^(1 - s)

# the Eulerian number ⟨n k⟩, for Li₋ₙ(z) = Σₖ ⟨n k⟩ z^(k+1) / (1 - z)^(n+1)
_eulerian(n::Integer, k::Integer) = sum((-1)^j * binomial(n + 1, j) * (k + 1 - j)^n for j in 0:k)

function _bernoulli_polynomial(n::Integer, t)
    B = _bernoulli_numbers(n)
    return sum(binomial(n, k) * B[k + 1] * t^(n - k) for k in 0:n)
end

# B₀, …, Bₙ, with B₁ = -1/2
function _bernoulli_numbers(n::Integer)
    B = Rational{BigInt}[1]
    for m in 1:n
        push!(B, -sum(binomial(big(m + 1), k) * B[k + 1] for k in 0:(m - 1)) / (m + 1))
    end
    return B
end
