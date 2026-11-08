# Bessel, Hankel and Airy functions. GIAC's BesselJ and its kin take only an integer order and a
# real argument, and have no derivatives, so these are kept as calls GIAC does not interpret,
# named as SpecialFunctions names them: besselj(ν, x), …, airyaiprime(x). Their derivatives are
# taken on the tree, their values come from SpecialFunctions, and an order that is half an odd
# integer is written with elementary functions, as are negative whole orders with positive ones.

const _BESSEL = ("besselj", "bessely", "besseli", "besselk", "hankelh1", "hankelh2")
const _AIRY = ("airyai", "airybi", "airyaiprime", "airybiprime")

_special(f::String, args::Node...) = CallNode("of", Node[SymNode(f), args...])

# the name of a call of one of the functions, or nothing
_special_name(n::Node) = nothing
function _special_name(n::CallNode)
    (n.f == "of" && length(n.args) >= 2 && n.args[1] isa SymNode) || return nothing
    f = n.args[1].name
    f in _BESSEL && length(n.args) == 3 && return f
    f in _AIRY && length(n.args) == 2 && return f
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

function _airy(f::String, x::Node)
    if _inexact(x)
        v = _special_value(f, x.v)
        v === nothing || return NumNode(v)
    end
    return _special(f, x)
end

# the value from SpecialFunctions, real where the function is real at a real argument
function _special_value(f::String, args::Number...)
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
    f = _special_name(n)
    f === nothing && return x in _symbol_names(n) ? nothing : _ZERO
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
    below, above = _bessel(f, _add(ν, _MINUS_ONE), u), _bessel(f, _add(ν, _ONE), u)
    d = _is_zero_node(ν) ? (f == "besseli" ? _bessel(f, _ONE, u) : _neg(_bessel(f, _ONE, u))) :
        f == "besseli" ? _mul(NumNode(1 // 2), _add(below, above)) :
        f == "besselk" ? _mul(NumNode(-1 // 2), _add(below, above)) :
        _mul(NumNode(1 // 2), _add(below, _neg(above)))
    return _mul(du, d)
end

# `evalf` of an expression with the functions in it: each at numbers replaced by its value first
_evalf_special(e::GiacExpr, rest...) = _invoke_cmd_direct(:evalf, (GiacExpr(_special_values(_node(e))), rest...))

_special_values(n::Node) = _replace_nodes(n) do m
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

# ∫ c xᵖ Z_ν(kx) dx, from
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
    if up !== nothing && up > 0 && isodd(up)
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
