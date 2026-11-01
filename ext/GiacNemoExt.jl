# ---------------------------------------------------------------------------
# GiacNemoExt — the Nemo.jl bridge.
#
# Loaded only when `Nemo` is loaded alongside `Giac`. Bidirectional:
#
#   outward  Nemo -> Giac   Giac.to_giac(e::Nemo.RingElem)
#   inward   Giac -> Nemo   Giac.to_nemo(e::GiacExpr, parent::NemoRing)
#
# The inward direction takes a *parent ring*: Nemo elements are parent-typed
# (there is no free "Nemo integer", only `ZZ(42)`, `QQ(42)`, an element of a
# specific polynomial ring, ...), whereas a GiacExpr is untyped/symbolic. The
# caller therefore states the destination:
#
#   to_nemo(giac_eval("x^2+1"), polynomial_ring(ZZ,"x")[1])
#   to_nemo(giac_eval("a^2+3"),  K)            # K = number_field(...)
#   to_nemo(giac_eval("t^2+1"),  F)            # F = finite_field(7,3,"t")
#
# CONTRACT — value-preserving and name-preserving; NOT representation-
# preserving. Compare round trips by value (`==`), never by `string`: Nemo
# polynomials are canonical (expanded, ring-sorted); a Giac factored form
# arrives in Nemo expanded. Variable *names* survive both crossings; term
# order does not.
#
# v1 scope — see docs/src/extensions/nemo.md for the full refusal list and
# limitations. In short:
#   * supported outward: ZZRingElem, QQFieldElem, polynomials in one or
#     several variables over ZZ/QQ and their quotients, FqFieldElem (prime &
#     non-prime), AbsSimpleNumFieldElem, ZZMatrix/QQMatrix.
#   * supported inward parent: ZZ, QQ, a polynomial ring in one or several
#     variables over ZZ/QQ or its fraction field, an AbsSimpleNumField, an
#     FqField, a ZZ/QQ MatSpace; or none, for a ring made from the
#     expression's own variables.
#   * refused (both directions): transcendental functions (sin, cos, exp,
#     ln, sqrt, log10, ...), the constants pi/e, real/complex balls
#     (ArbFieldElem/AcbFieldElem), p-adics (PadicFieldElem). Reals are
#     refused because silent precision loss via a printed decimal is exactly
#     the trap the LibPARI bridge warns about.
# ---------------------------------------------------------------------------

module GiacNemoExt

using Giac
using Giac.GenTypes
using Nemo
using Nemo: ZZ, QQ
import Nemo: PolyRing, MatSpace, PolyRingElem, MatElem
using CxxWrap: StdVector

const OUTWARD_SUPPORTED =
    "ZZRingElem, QQFieldElem, polynomials over ZZ/QQ and their quotients, FqFieldElem, " *
    "AbsSimpleNumFieldElem, ZZMatrix/QQMatrix"
const INWARD_SUPPORTED =
    "parent ring ZZ, QQ, a polynomial ring over ZZ/QQ or its fraction field, an " *
    "AbsSimpleNumField, an FqField, or a ZZ/QQ MatSpace"

# Allowed GIAC operators inside an expression destined for an algebraic
# Nemo ring (polynomial / number field / finite field). Anything else
# (sin, cos, exp, ln, sqrt, log10, ...) is transcendental and refused.
const ALGEBRAIC_OPS = Set(["+", "-", "*", "/", "^", "inv", "neg", "unsigned_numeric"])

# ===========================================================================
# Helpers — BigInt <-> Gen (mirror GiacSymPyExt / GiacSymbolicsExt)
# ===========================================================================

function _bigint_to_gen(n::BigInt)
    if n == 0
        return Giac.GiacCxxBindings.Gen(Int32(0))
    end
    n_sign = Int32(Base.sign(n))
    abs_n = abs(n)
    bit_count = ccall((:__gmpz_sizeinbase, Base.GMP.libgmp), Csize_t,
                      (Ref{BigInt}, Cint), abs_n, 2)
    byte_count = div(bit_count + 7, 8)
    bytes = Vector{UInt8}(undef, byte_count)
    actual_count = Ref{Csize_t}(0)
    ccall((:__gmpz_export, Base.GMP.libgmp), Ptr{Cvoid},
          (Ptr{UInt8}, Ref{Csize_t}, Cint, Csize_t, Cint, Csize_t, Ref{BigInt}),
          bytes, actual_count, 1, 1, 1, 0, abs_n)
    if actual_count[] < byte_count
        resize!(bytes, actual_count[])
    end
    return Giac.GiacCxxBindings.make_zint_from_bytes(StdVector{UInt8}(bytes), n_sign)
end

# Build a Gen symbolic power gen_ident^k (gen^0 => 1, gen^1 => gen_ident).
function _gen_pow(gen_ident, k::Int)
    g = Giac.GiacCxxBindings
    if k == 0
        return g.Gen(Int32(1))
    elseif k == 1
        return gen_ident
    else
        return g.make_symbolic_unevaluated("^",
            StdVector{g.Gen}([gen_ident, g.Gen(Int32(k))]))
    end
end

# Build a symbolic product coeff * term (coeff^1 => coeff, term=1 => coeff).
function _gen_mul(coeff_gen, term_gen)
    g = Giac.GiacCxxBindings
    one_gen = g.Gen(Int32(1))
    if term_gen === one_gen || _is_one(term_gen)
        return coeff_gen
    end
    return g.make_symbolic_unevaluated("*",
        StdVector{g.Gen}([coeff_gen, term_gen]))
end

# Build a symbolic sum of the given Gen terms, dropping additive identities.
function _gen_sum(terms)
    g = Giac.GiacCxxBindings
    nonzero = filter(t -> !_is_zero(t), terms)
    isempty(nonzero) && return g.Gen(Int32(0))
    length(nonzero) == 1 && return nonzero[1]
    result = nonzero[1]
    for i in 2:length(nonzero)
        result = g.make_symbolic_unevaluated("+",
            StdVector{g.Gen}([result, nonzero[i]]))
    end
    return result
end

# Cheap structural zero/one test on a Gen, via Giac.to_julia (handles INT,
# ZINT and FRAC leaves uniformly).
function _gen_to_julia(gen)
    return Giac.to_julia(GiacExpr(Giac._gen_to_ptr(gen)))
end

function _is_zero(gen)
    t = GenTypes.T(Giac.GiacCxxBindings.type(gen))
    (t in (GenTypes.INT, GenTypes.ZINT, GenTypes.FRAC)) || return false
    return _gen_to_julia(gen) == 0
end

function _is_one(gen)
    t = GenTypes.T(Giac.GiacCxxBindings.type(gen))
    (t in (GenTypes.INT, GenTypes.ZINT, GenTypes.FRAC)) || return false
    return _gen_to_julia(gen) == 1
end

# ===========================================================================
# Outward: Nemo -> Giac
# ===========================================================================

"""
    Giac.to_giac(e::Nemo.ZZRingElem)

Convert a Nemo integer to a `GiacExpr` (ZINT for arbitrary precision).
"""
function Giac.to_giac(e::ZZRingElem)
    n = BigInt(e)
    if typemin(Int32) <= n <= typemax(Int32)
        return GiacExpr(Giac._gen_to_ptr(Giac.GiacCxxBindings.Gen(Int32(n))))
    end
    return GiacExpr(Giac._gen_to_ptr(_bigint_to_gen(n)))
end

"""
    Giac.to_giac(e::Nemo.QQFieldElem)

Convert a Nemo rational to a `GiacExpr` (GIAC fraction).
"""
function Giac.to_giac(e::QQFieldElem)
    g = Giac.GiacCxxBindings
    num = _bigint_to_gen(BigInt(numerator(e)))
    den = _bigint_to_gen(BigInt(denominator(e)))
    return GiacExpr(Giac._gen_to_ptr(g.make_fraction(num, den)))
end

"""
    Giac.to_giac(e::Nemo.UnivariatePolyRingElem)

Convert a univariate polynomial over ZZ or QQ to a `GiacExpr`, rebuilding it
as a symbolic sum of `c_k * var^k` terms. The variable name is taken from the
polynomial ring's generator.
"""
function Giac.to_giac(e::PolyRingElem)
    R = parent(e)
    vname = string(gen(R))
    gen_ident = Giac.GiacCxxBindings.make_identifier(vname)
    coeffs = collect(coefficients(e))   # low -> high
    terms = Giac.GiacCxxBindings.Gen[]
    for (k, c) in enumerate(coeffs)
        k0 = k - 1
        c_gen = Giac._ptr_to_gen(to_giac(c))
        term = _gen_pow(gen_ident, k0)
        push!(terms, _gen_mul(c_gen, term))
    end
    return GiacExpr(Giac._gen_to_ptr(_gen_sum(terms)))
end

"""
    Giac.to_giac(e::Nemo.FqFieldElem)

Convert a finite-field element to a `GiacExpr`. For a prime field (degree 1)
this is just the lifted integer; for a non-prime field it is a polynomial in
the field generator symbol.
"""
function Giac.to_giac(e::FqFieldElem)
    F = parent(e)
    d = Int(degree(F))
    vname = string(gen(F))
    gen_ident = Giac.GiacCxxBindings.make_identifier(vname)
    terms = Giac.GiacCxxBindings.Gen[]
    for k in 0:(d - 1)
        c = Nemo._coeff(e, k)            # ZZRingElem
        c_gen = Giac._ptr_to_gen(to_giac(c))
        term = _gen_pow(gen_ident, k)
        push!(terms, _gen_mul(c_gen, term))
    end
    return GiacExpr(Giac._gen_to_ptr(_gen_sum(terms)))
end

"""
    Giac.to_giac(e::Nemo.AbsSimpleNumFieldElem)

Convert a (absolute, simple) number-field element to a `GiacExpr`, rebuilding
it as a polynomial in the field generator symbol. The element is stored
mod its defining polynomial, so the result is already reduced.
"""
function Giac.to_giac(e::AbsSimpleNumFieldElem)
    K = parent(e)
    d = Int(degree(K))
    vname = string(gen(K))
    gen_ident = Giac.GiacCxxBindings.make_identifier(vname)
    terms = Giac.GiacCxxBindings.Gen[]
    for k in 0:(d - 1)
        c = Nemo.coeff(e, k)             # QQFieldElem
        c_gen = Giac._ptr_to_gen(to_giac(c))
        term = _gen_pow(gen_ident, k)
        push!(terms, _gen_mul(c_gen, term))
    end
    return GiacExpr(Giac._gen_to_ptr(_gen_sum(terms)))
end

"""
    Giac.to_giac(M::Nemo.ZZMatrix)
    Giac.to_giac(M::Nemo.QQMatrix)

Convert a Nemo matrix over ZZ or QQ to a `Giac.GiacMatrix`.
"""
function Giac.to_giac(M::MatElem)
    nr, nc = nrows(M), ncols(M)
    rows = Vector{Vector{GiacExpr}}()
    for i in 1:nr
        row = GiacExpr[to_giac(M[i, j]) for j in 1:nc]
        push!(rows, row)
    end
    return Giac.GiacMatrix(rows)
end

# Refusals — reals, balls, p-adics are out of scope in v1.
function Giac.to_giac(e::ArbFieldElem)
    error("to_giac(::$(typeof(e))) is not supported in v1: real balls are " *
          "refused to avoid silent precision loss. Supported: $OUTWARD_SUPPORTED")
end
function Giac.to_giac(e::AcbFieldElem)
    error("to_giac(::$(typeof(e))) is not supported in v1: complex balls are " *
          "refused to avoid silent precision loss. Supported: $OUTWARD_SUPPORTED")
end
function Giac.to_giac(e::PadicFieldElem)
    error("to_giac(::$(typeof(e))) is not supported: p-adic elements have no " *
          "Giac analogue. Supported: $OUTWARD_SUPPORTED")
end

# ===========================================================================
# Inward: Giac -> Nemo  (caller supplies the destination parent ring)
# ===========================================================================

# --- algebraic-guard: walk the Gen tree and refuse anything that is not
#     integer / rational / identifier / an algebraic operator. ---
function _assert_algebraic(gen)
    t = GenTypes.T(Giac.GiacCxxBindings.type(gen))
    if t in (GenTypes.INT, GenTypes.ZINT, GenTypes.FRAC)
        return
    elseif t == GenTypes.IDNT
        name = String(Giac.GiacCxxBindings.idnt_name(gen))
        if name == "pi" || name == "e"
            error("to_nemo: GIAC constant '$name' is transcendental and has " *
                  "no element in an algebraic Nemo ring. Supported: $INWARD_SUPPORTED")
        end
        return
    elseif t == GenTypes.DOUBLE || t == GenTypes.REAL
        error("to_nemo: real values are refused in v1 to avoid silent " *
              "precision loss. Supported: $INWARD_SUPPORTED")
    elseif t == GenTypes.STRNG
        error("to_nemo: GIAC strings have no Nemo analogue. Supported: $INWARD_SUPPORTED")
    elseif t == GenTypes.SYMB
        op = String(Giac.GiacCxxBindings.symb_sommet_name(gen))
        if !(op in ALGEBRAIC_OPS)
            error("to_nemo: GIAC operator '$op' is transcendental / " *
                  "unsupported in an algebraic Nemo ring. Supported: $INWARD_SUPPORTED")
        end
        feuille = Giac.GiacCxxBindings.symb_feuille(gen)
        ft = GenTypes.T(Giac.GiacCxxBindings.type(feuille))
        if ft == GenTypes.VECT
            n = Giac.GiacCxxBindings.vect_size(feuille)
            for i in 1:n
                _assert_algebraic(Giac.GiacCxxBindings.vect_at(feuille, i - 1))
            end
        else
            _assert_algebraic(feuille)
        end
        return
    elseif t == GenTypes.CPLX
        error("to_nemo: GIAC complex values have no algebraic Nemo ring " *
              "target in v1 (use number_field(x^2+1) explicitly). Supported: $INWARD_SUPPORTED")
    else
        error("to_nemo: unsupported GIAC type '$(t)'. Supported: $INWARD_SUPPORTED")
    end
end

# --- integer / rational leaf extraction via Giac.to_julia ---
function _leaf_to_julia(expr::GiacExpr)
    v = Giac.to_julia(expr)
    v isa Integer || v isa Rational ||
        error("to_nemo: expected an integer or rational value, got $(typeof(v)). " *
              "Supported: $INWARD_SUPPORTED")
    return v
end

# --- into ZZ ---
function Giac.to_nemo(expr::GiacExpr, ::ZZRing)
    gen = Giac._ptr_to_gen(expr)
    t = GenTypes.T(Giac.GiacCxxBindings.type(gen))
    if t in (GenTypes.DOUBLE, GenTypes.REAL)
        error("to_nemo: real values are refused in v1. Supported: $INWARD_SUPPORTED")
    elseif t == GenTypes.FRAC
        error("to_nemo: non-integer rational cannot be converted to ZZ. " *
              "Use QQ as the target. Supported: $INWARD_SUPPORTED")
    end
    v = Giac.to_julia(expr)
    v isa Integer ||
        error("to_nemo: expected an integer for ZZ target, got $(typeof(v)). " *
              "Supported: $INWARD_SUPPORTED")
    return ZZ(BigInt(v))
end

# --- into QQ ---
function Giac.to_nemo(expr::GiacExpr, ::QQField)
    gen = Giac._ptr_to_gen(expr)
    t = GenTypes.T(Giac.GiacCxxBindings.type(gen))
    if t in (GenTypes.DOUBLE, GenTypes.REAL)
        error("to_nemo: real values are refused in v1. Supported: $INWARD_SUPPORTED")
    end
    v = Giac.to_julia(expr)
    if v isa Integer
        return QQ(BigInt(v), BigInt(1))
    elseif v isa Rational
        return QQ(BigInt(numerator(v)), BigInt(denominator(v)))
    else
        error("to_nemo: expected an integer or rational for QQ target, got " *
              "$(typeof(v)). Supported: $INWARD_SUPPORTED")
    end
end

# --- into a univariate polynomial ring over ZZ or QQ ---
function Giac.to_nemo(expr::GiacExpr, R::PolyRing)
    gen = Giac._ptr_to_gen(expr)
    _assert_algebraic(gen)
    B = base_ring(R)
    vars = Giac.Commands.lvar(expr)
    nvars = length(vars)
    if nvars == 0
        return R(to_nemo(expr, B))
    elseif nvars == 1
        v = vars[1]
        d = Int(Giac.to_julia(Giac.Commands.degree(expr, v)))
        coeffs = [to_nemo(Giac.Commands.coeff(expr, v, k), B) for k in 0:d]
        return R(coeffs)
    else
        error("to_nemo: a univariate polynomial ring was given, and the expression has " *
              "$nvars variables; use a polynomial ring in several variables")
    end
end

# --- into an AbsSimpleNumField ---
function Giac.to_nemo(expr::GiacExpr, K::AbsSimpleNumField)
    gen = Giac._ptr_to_gen(expr)
    _assert_algebraic(gen)
    vars = Giac.Commands.lvar(expr)
    nvars = length(vars)
    if nvars == 0
        return K(to_nemo(expr, QQ))
    elseif nvars == 1
        v = vars[1]
        d = Int(Giac.to_julia(Giac.Commands.degree(expr, v)))
        coeffs = [to_nemo(Giac.Commands.coeff(expr, v, k), QQ) for k in 0:d]
        PQ = parent(defining_polynomial(K))    # QQ[x]
        return K(PQ(coeffs))
    else
        error("to_nemo: multivariate expressions are not supported into a " *
              "number field (got $nvars free variables). Supported: $INWARD_SUPPORTED")
    end
end

# --- into an FqField (prime or non-prime) ---
function Giac.to_nemo(expr::GiacExpr, F::FqField)
    gen = Giac._ptr_to_gen(expr)
    _assert_algebraic(gen)
    vars = Giac.Commands.lvar(expr)
    nvars = length(vars)
    if nvars == 0
        # constant -> lift to ZZ then into F
        v = Giac.to_julia(expr)
        v isa Integer ||
            error("to_nemo: expected an integer constant for a finite field, " *
                  "got $(typeof(v)). Supported: $INWARD_SUPPORTED")
        return F(ZZ(BigInt(v)))
    elseif nvars == 1
        v = vars[1]
        d = Int(Giac.to_julia(Giac.Commands.degree(expr, v)))
        # Build the element directly as a polynomial in the field generator.
        # (Going through polynomial_ring(F,"x")[1] and F(poly) does NOT map
        # x -> gen(F) for Fq fields — it applies an unrelated coercion that
        # yields wrong results.)
        t = Nemo.gen(F)
        result = Nemo.zero(F)
        for k in 0:d
            c = F(to_nemo(Giac.Commands.coeff(expr, v, k), ZZ))
            result += c * t^k
        end
        return result
    else
        error("to_nemo: multivariate expressions are not supported into a " *
              "finite field (got $nvars free variables). Supported: $INWARD_SUPPORTED")
    end
end

# --- into a ZZ/QQ MatSpace ---
function Giac.to_nemo(expr::GiacExpr, M::MatSpace)
    B = base_ring(M)
    nr = nrows(M)
    nc = ncols(M)
    # expr is a Giac VECT of VECTs (rows). Length-check the structure.
    length(expr) == nr ||
        error("to_nemo: matrix has $(length(expr)) rows, expected $nr.")
    elems = elem_type(B)[]
    for i in 1:nr
        row = expr[i]
        length(row) == nc ||
            error("to_nemo: row $i has $(length(row)) cols, expected $nc.")
        for j in 1:nc
            push!(elems, to_nemo(row[j], B))
        end
    end
    return matrix(B, nr, nc, elems)
end

# ===========================================================================
# Multivariate: polynomial rings and their fraction fields, names preserved
# ===========================================================================
#
# Both directions go through Giac's expression tree, not through printing: a sum of monomials
# becomes a polynomial built term by term, and a polynomial comes back as a tree of the same
# names. Into a polynomial ring, a power must be a whole nonnegative number; into a fraction
# field, any whole power.

import Nemo: MPolyRing, MPolyRingElem, FracField, FracElem

const _Node = Giac.Node

# the generators of a ring by name, as elements of `F` (the ring itself or its fraction field)
function _gens_by_name(R, F)
    return Dict(string(s) => F(g) for (s, g) in zip(symbols(R), gens(R)))
end

_qq(v::Integer) = QQ(BigInt(v))
_qq(v::Rational) = QQ(BigInt(numerator(v)), BigInt(denominator(v)))
_qq(v) = error("to_nemo: the number $v is not an integer or a rational; floating-point and " *
               "complex numbers have no exact element here")

# a number in the polynomial ring R, or in its coefficient ring
_coerce(R, v) = R(_scalar(R, v))
function _scalar(R, v)
    base_ring(R) isa ZZRing || return base_ring(R)(_qq(v))
    v isa Integer || error("to_nemo: $v is not an integer, and the ring is over ZZ")
    return ZZ(BigInt(v))
end

# The value of a tree in `F`: a polynomial ring or a fraction field, with `vars` its generators
_value(n::Giac.NumNode, F, vars) = _constant(F, n.v)
function _value(n::Giac.SymNode, F, vars)
    v = get(vars, n.name, nothing)
    v === nothing && error("to_nemo: $(n.name) is not a variable of the ring (its variables are " *
                           join(sort!(collect(keys(vars))), ", ") * ")")
    return v
end
function _value(n::Giac.AddNode, F, vars)
    F isa MPolyRing && all(t -> _is_monomial(first(t)), n.terms) && return _built(n, F, vars)
    acc = zero(F)
    for (m, c) in n.terms
        acc += _constant(F, c) * _value(m, F, vars)
    end
    return acc
end
function _value(n::Giac.MulNode, F, vars)
    acc = _constant(F, n.coef)
    for (b, e) in n.factors
        k = _whole_power(e, F)
        acc *= k >= 0 ? _value(b, F, vars)^k : inv(_value(b, F, vars))^(-k)
    end
    return acc
end
_value(n::Giac.CallNode, F, vars) =
    error("to_nemo: $(n.f)(…) is not a polynomial or rational function of the ring's variables")
_value(n::Giac.GenNode, F, vars) = error("to_nemo: $(n.key) has no element in $(F)")

_constant(F, v) = F isa FracField ? F(_coerce(base_ring(F), v)) : _coerce(F, v)

function _whole_power(e, F)
    (e isa Giac.NumNode && e.v isa Integer) ||
        error("to_nemo: the power $(Giac._str(e)) is not a whole number")
    e.v < 0 && !(F isa FracField) &&
        error("to_nemo: a negative power has no value in a polynomial ring; use its fraction field")
    return Int(e.v)
end

# a product of variables to whole nonnegative powers, with coefficient 1
_is_monomial(m::_Node) = m === Giac._UNIT || isequal(m, Giac._UNIT) || m isa Giac.SymNode ||
    (m isa Giac.MulNode && isone(m.coef) &&
     all(((b, e),) -> b isa Giac.SymNode && e isa Giac.NumNode && e.v isa Integer && e.v >= 0, m.factors))

# a sum of monomials, built term by term
function _built(n::Giac.AddNode, R, vars)
    index = Dict(string(s) => i for (i, s) in enumerate(symbols(R)))
    ctx = MPolyBuildCtx(R)
    for (m, c) in n.terms
        ev = zeros(Int, length(index))
        for (b, e) in (m isa Giac.MulNode ? m.factors : m isa Giac.SymNode ? [m => Giac._ONE] : Pair{_Node, _Node}[])
            i = get(index, b.name, nothing)
            i === nothing && _value(b, R, vars)       # the error naming the variable
            ev[i] += Int(e.v)
        end
        push_term!(ctx, _scalar(R, c), ev)
    end
    return finish(ctx)
end

"""
    to_nemo(e::GiacExpr, R::MPolyRing)
    to_nemo(e::GiacExpr, F::FracField)

`e` as a polynomial of `R`, or a rational function of the fraction field `F` of a polynomial
ring. A variable of `e` is the generator of the same name. The coefficients are integers or
rationals; functions such as `sin`, and floating-point numbers, are refused.

```julia
using Nemo
@giac_var x y
R, (X, Y) = polynomial_ring(QQ, [:x, :y])
to_nemo((x + y)^2 / 2, R)                     # 1//2*x^2 + x*y + 1//2*y^2
to_nemo(1 / (x - y), fraction_field(R))
```
"""
Giac.to_nemo(e::GiacExpr, R::MPolyRing) = _value(Giac._node(e), R, _gens_by_name(R, R))
Giac.to_nemo(e::GiacExpr, F::FracField) = _value(Giac._node(e), F, _gens_by_name(base_ring(F), F))

"""
    to_nemo(e::GiacExpr; base = QQ)
    to_nemo(es; base = QQ)

`e` in a ring made from its own variables: the polynomial ring over `base` in them, or its
fraction field when `e` has a variable or a sum in a denominator. Several expressions, as a
tuple or vector, go into one ring. The variables are ordered as they were declared.

```julia
@giac_var a b
p, q = to_nemo((a^2 - b^2, a + b))
divexact(p, q)                                # a - b
```
"""
Giac.to_nemo(e::GiacExpr; base = QQ) = only(Giac.to_nemo((e,); base))
function Giac.to_nemo(es::Union{Tuple, AbstractVector}; base = QQ)
    nodes = [Giac._node(e isa GiacExpr ? e : convert(GiacExpr, e)) for e in es]
    names = sort!(collect(union((Giac._symbol_names(n) for n in nodes)...)); by = s -> (Giac._declared_order(s), s))
    R, _ = polynomial_ring(base, Symbol.(names))
    F = any(_has_fraction, nodes) ? fraction_field(R) : R
    vars = _gens_by_name(R, F)
    return map(n -> _value(n, F, vars), nodes)
end

_has_fraction(n::_Node) = false
_has_fraction(n::Giac.AddNode) = any(t -> _has_fraction(first(t)), n.terms)
_has_fraction(n::Giac.MulNode) =
    any(((b, e),) -> (e isa Giac.NumNode && e.v isa Real && e.v < 0) || _has_fraction(b), n.factors)

"""
    to_giac(p::MPolyRingElem)
    to_giac(f::FracElem)

A polynomial of several variables, or a quotient of them, as a `GiacExpr` in the same names.
"""
function Giac.to_giac(p::MPolyRingElem)
    names = [Giac.SymNode(string(s)) for s in symbols(parent(p))]
    terms = Giac.Node[]
    for (c, ev) in zip(coefficients(p), exponent_vectors(p))
        factors = Giac.Node[Giac._pow(names[i], Giac.NumNode(k)) for (i, k) in enumerate(ev) if k > 0]
        push!(terms, Giac._mul(_node_number(c), factors...))
    end
    return GiacExpr(isempty(terms) ? Giac._ZERO : Giac._add(terms...))
end
Giac.to_giac(f::FracElem) = Giac.to_giac(numerator(f)) / Giac.to_giac(denominator(f))

_node_number(c::ZZRingElem) = Giac.NumNode(BigInt(c))
_node_number(c::QQFieldElem) = Giac.NumNode(Rational{BigInt}(c))
_node_number(c) = Giac._node(Giac.to_giac(c))

# ===========================================================================
# positivity: coefficient signs after writing an ordering as positive gaps
# ===========================================================================

_chains(order) = isempty(order) ? Any[] :
    (order isa Tuple && !(first(order) isa Union{Tuple, AbstractVector})) ? Any[order] : collect(Any, order)

function Giac.positivity(e; order = (), positive = (), param = (), strict = true)
    e = convert(GiacExpr, e)
    isempty(param) || (e = substitute(e, param...))
    gaps = Pair{GiacExpr, GiacExpr}[]
    known = Set{String}(string.(positive))         # variables known positive (or ≥ 0 if not strict)
    for chain in _chains(order)
        items = collect(Any, chain)
        start = popfirst!(items)
        acc = convert(GiacExpr, start)
        if start isa Number && iszero(start) && !isempty(items)
            # 0 < v: v itself is the first gap
            v = popfirst!(items)
            push!(known, string(v))
            acc = convert(GiacExpr, v)
        end
        for v in items
            Δ = Giac.giac_identifier("Δ" * string(v))
            push!(known, string(Δ))
            acc = acc + Δ
            push!(gaps, convert(GiacExpr, v) => acc)
        end
    end
    isempty(gaps) || (e = substitute(e, gaps...))
    rational = Giac.Commands.normal(e)
    num, den = Giac.Commands.numer(rational), Giac.Commands.denom(rational)
    N, D = Giac.to_nemo((num, den))
    rn, rd = _sign_report(N, known, strict), _sign_report(D, known, strict)
    return Giac.Positivity(rn.sign * rd.sign, rn.strict && rd.strict, rn, rd, gaps)
end

function _sign_report(P, known, strict)
    vars = [string(s) for (s, g) in zip(symbols(parent(P)), gens(parent(P))) if degree(P, g) > 0]
    free = [Giac.giac_identifier(v) for v in vars if !(v in known) && !_declared_positive(v)]
    cs = collect(coefficients(P))
    pos, neg = count(>(0), cs), count(<(0), cs)
    sign = !isempty(free) || iszero(P) ? 0 : neg == 0 ? 1 : pos == 0 ? -1 : 0
    majority = pos >= neg ? 1 : -1
    against = [Giac.to_giac(c * m) for (c, m) in zip(coefficients(P), monomials(P)) if sign == 0 && Base.sign(c) == -majority]
    # strictly positive variables and a nonzero polynomial give a strict sign; with ≤ chains
    # (strict = false) a constant term is needed
    strict_sign = sign != 0 && (strict ? all(v -> v in known || _declared_positive(v, true), vars) : !iszero(constant_coefficient(P)))
    return Giac.SignReport(sign, strict_sign, Giac.to_giac(P), against, free)
end

# whether a name was declared positive (`@giac_var A::positive`), or with `strict_only = false`
# nonnegative
function _declared_positive(name, strict_only = false)
    tag = lock(() -> get(Giac._DECLARED, name, nothing), Giac._DECLARED_LOCK)
    return tag === :positive || (!strict_only && tag === :nonnegative) ||
           (tag isa Tuple && length(tag) == 2 && tag[1] isa Real && (tag[1] > 0 || !strict_only && tag[1] >= 0))
end

export to_giac, to_nemo

end # module GiacNemoExt