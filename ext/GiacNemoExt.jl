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

function Giac.positivity(e; order = (), positive = (), param = (), strict = true, sos = true)
    e = convert(GiacExpr, e)
    isempty(param) || (e = substitute(e, param...))
    gaps, known = Giac._gap_substitution(order, positive)
    isempty(gaps) || (e = substitute(e, gaps...))
    # expanded and put over a common denominator in Nemo, which is much faster than GIAC's
    # normal on large expressions
    v = only(Giac.to_nemo((e,)))
    N, D = v isa FracElem ? (numerator(v), denominator(v)) : (v, one(v))
    rn, rd = _sign_report(N, known, strict, sos), _sign_report(D, known, strict, sos)
    return Giac.Positivity(rn.sign * rd.sign, rn.strict && rd.strict, rn, rd, gaps)
end

# The sign of P from a sum of squares, when the Clarabel extension is loaded: P, or -P, with
# each variable known to be nonnegative written as a square, is a sum of squares
function _sos_sign(P, known)
    isempty(methods(Giac._sos_certificate)) && return nothing
    R = parent(P)
    Ps = evaluate(P, [(string(s) in known || _declared_positive(string(s))) ? g^2 : g for (s, g) in zip(symbols(R), gens(R))])
    for s in (1, -1)
        c = Giac._sos_certificate(s * Ps)
        c === nothing || return s, c
    end
    return nothing
end

function _sign_report(P, known, strict, sos = true)
    vars = [string(s) for (s, g) in zip(symbols(parent(P)), gens(parent(P))) if degree(P, g) > 0]
    free = [Giac.giac_identifier(v) for v in vars if !(v in known) && !_declared_positive(v)]
    cs = collect(coefficients(P))
    pos, neg = count(>(0), cs), count(<(0), cs)
    sign = !isempty(free) || iszero(P) ? 0 : neg == 0 ? 1 : pos == 0 ? -1 : 0
    certificate = nothing
    if sign == 0 && sos && !iszero(P) && (pos > 0 && neg > 0 || !isempty(free))
        found = _sos_sign(P, known)
        found === nothing || ((sign, certificate) = found)
    end
    majority = pos >= neg ? 1 : -1
    degrees = iszero(P) ? (0:0) : (minimum(sum, exponent_vectors(P)):maximum(sum, exponent_vectors(P)))
    # the terms against the majority sign, largest coefficient first, and the variables in them
    opposing = [(c, e) for (c, e) in zip(coefficients(P), exponent_vectors(P)) if sign == 0 && Base.sign(c) == -majority]
    sort!(opposing; by = t -> -abs(t[1]))
    names = string.(symbols(parent(P)))
    uses = zeros(Int, length(names))
    foreach(((c, e),) -> (uses .+= e .> 0), opposing)
    in_against = sort!([names[i] => uses[i] for i in eachindex(names) if uses[i] > 0]; by = p -> -last(p))
    R = parent(P)
    against = [Giac.to_giac(c * _monomial(R, e)) for (c, e) in opposing]
    # strictly positive variables and a nonzero polynomial give a strict sign; with ≤ chains
    # (strict = false) a constant term is needed
    # a sum of squares may vanish somewhere, so its sign is not strict
    strict_sign = sign != 0 && certificate === nothing &&
                  (strict ? all(v -> v in known || _declared_positive(v, true), vars) : !iszero(constant_coefficient(P)))
    return Giac.SignReport(sign, strict_sign, Giac.to_giac(P), against, free, (pos, neg), degrees, in_against, certificate)
end

_monomial(R, e) = (ctx = MPolyBuildCtx(R); push_term!(ctx, one(base_ring(R)), e); finish(ctx))

# whether a name was declared positive (`@giac_var A::positive`), or with `strict_only = false`
# nonnegative
function _declared_positive(name, strict_only = false)
    tag = Giac._declared_tag(name)
    return tag === :positive || (!strict_only && tag === :nonnegative) ||
           (tag isa Tuple && length(tag) == 2 && tag[1] isa Real && (tag[1] > 0 || !strict_only && tag[1] >= 0))
end

# ===========================================================================
# decompose: target = Σ Mₖ Bₖ with unknown multipliers, by exact coefficient matching
# ===========================================================================

function Giac.decompose(target, blocks; vars = nothing, degree = nothing, param = (), products = 1)
    pairs = [b isa Pair ? (string(first(b)), convert(GiacExpr, last(b))) : (nothing, convert(GiacExpr, b)) for b in blocks]
    factors = [[k] for k in eachindex(pairs)]
    if products > 1
        factors = [f for n in 1:products for f in _multisets(length(pairs), n)]
        label(i) = Giac._block_text(pairs[i]...)
        pairs = [(length(f) == 1 ? pairs[only(f)][1] : join(label.(f), "·"), prod(pairs[i][2] for i in f)) for f in factors]
    end
    exprs = GiacExpr[convert(GiacExpr, target); last.(pairs)]
    isempty(param) || (exprs = [substitute(e, param...) for e in exprs])
    T, Bs = _cleared(Giac.to_nemo(exprs))
    R = parent(T)
    names = Union{String, Nothing}[n for n in first.(pairs)]
    allowed = vars === nothing ? collect(1:nvars(R)) :
              [findfirst(==(Symbol(string(v))), symbols(R)) for v in vars]
    any(isnothing, allowed) && throw(ArgumentError("decompose: a variable in `vars` does not occur in the target or blocks"))
    degrees = degree === nothing ? [total_degree(T) - total_degree(B) for B in Bs] :
              degree isa Integer ? fill(degree, length(Bs)) : collect(degree)
    length(degrees) == length(Bs) || throw(ArgumentError("decompose: one degree per block"))

    # the unknowns: one coefficient per (block, monomial of the multiplier)
    unknowns = Tuple{Int, Vector{Int}}[]
    for (k, d) in enumerate(degrees), ev in _exponents(nvars(R), allowed, d)
        push!(unknowns, (k, ev))
    end
    rows = Dict{Vector{Int}, Int}()
    row(ev) = get!(rows, ev, length(rows) + 1)
    entries = Tuple{Int, Int, QQFieldElem}[]
    for (j, (k, ev)) in enumerate(unknowns)
        for (c, e) in zip(coefficients(Bs[k]), exponent_vectors(Bs[k]))
            push!(entries, (row(e .+ ev), j, QQ(c)))
        end
    end
    rhs = [(row(e), QQ(c)) for (c, e) in zip(coefficients(T), exponent_vectors(T))]
    solved = _solve_sparse(entries, rhs, length(rows), length(unknowns))
    solved === nothing && return nothing
    x, K = solved
    multipliers(v) = [Giac.to_giac(_polynomial(R, [(v[j], unknowns[j][2]) for j in eachindex(unknowns) if unknowns[j][1] == k]))
                      for k in eachindex(Bs)]
    directions = [multipliers(K[:, j]) for j in axes(K, 2)]
    return Giac.Decomposition(exprs[1], exprs[2:end], names, multipliers(x), directions, factors)
end

# The solutions of A x = b, A given by its entries (row, column, value): a particular solution
# and a basis of the kernel, as columns. Rows with a single unknown fix it first, which removes
# most unknowns of a coefficient-matching system (a product monomial that nothing else can
# cancel); the rest is solved densely. `nothing` if there is no solution.
function _solve_sparse(entries, rhs, nrows, nunknowns)
    colentries = [Dict{Int, QQFieldElem}() for _ in 1:nunknowns]
    rowcols = [Set{Int}() for _ in 1:nrows]
    for (i, j, c) in entries
        colentries[j][i] = get(colentries[j], i, QQ(0)) + c
    end
    for j in 1:nunknowns, (i, c) in collect(colentries[j])
        iszero(c) ? delete!(colentries[j], i) : push!(rowcols[i], j)
    end
    b = [QQ(0) for _ in 1:nrows]
    for (i, c) in rhs
        b[i] += c
    end
    value = Dict{Int, QQFieldElem}()
    queue = [i for i in 1:nrows if length(rowcols[i]) == 1]
    while !isempty(queue)
        i = pop!(queue)
        length(rowcols[i]) == 1 || continue
        j = only(rowcols[i])
        v = b[i] // colentries[j][i]
        value[j] = v
        for (r, c) in colentries[j]
            b[r] -= c * v
            delete!(rowcols[r], j)
            length(rowcols[r]) == 1 && push!(queue, r)
        end
    end
    any(i -> isempty(rowcols[i]) && !iszero(b[i]), 1:nrows) && return nothing
    free = [j for j in 1:nunknowns if !haskey(value, j)]
    live = [i for i in 1:nrows if !isempty(rowcols[i])]
    x = [get(value, j, QQ(0)) for j in 1:nunknowns]
    K = Matrix{QQFieldElem}(undef, nunknowns, 0)
    if !isempty(free)
        index = Dict(j => k for (k, j) in enumerate(free))
        A = zero_matrix(QQ, length(live), length(free))
        for (r, i) in enumerate(live), j in rowcols[i]
            A[r, index[j]] = colentries[j][i]
        end
        B = matrix(QQ, length(live), 1, [b[i] for i in live])
        ok, y, N = can_solve_with_solution_and_kernel(A, B; side = :right)
        ok || return nothing
        for (k, j) in enumerate(free)
            x[j] = y[k, 1]
        end
        K = [QQ(0) for _ in 1:nunknowns, _ in 1:ncols(N)]
        for (k, j) in enumerate(free), c in 1:ncols(N)
            K[j, c] = N[k, c]
        end
    end
    return x, K
end

# the multisets of size k of 1:n, as sorted vectors
function _multisets(n, k)
    k == 0 && return [Int[]]
    return [vcat(m, j) for m in _multisets(n, k - 1) for j in (isempty(m) ? 1 : last(m)):n]
end

# polynomials with the denominators cleared: each of T, Bₖ times the lcm of the denominators
function _cleared(values)
    first(values) isa FracElem || return first(values), values[2:end]
    L = reduce(lcm, (denominator(v) for v in values))
    polys = [divexact(numerator(v) * L, denominator(v)) for v in values]
    return first(polys), polys[2:end]
end

# the exponent vectors of total degree ≤ d in the variables `allowed`
function _exponents(n, allowed, d)
    d < 0 && return Vector{Int}[]
    out = Vector{Int}[]
    function walk(i, left, ev)
        if i > length(allowed)
            push!(out, copy(ev))
            return
        end
        for k in 0:left
            ev[allowed[i]] = k
            walk(i + 1, left - k, ev)
        end
        ev[allowed[i]] = 0
    end
    walk(1, d, zeros(Int, n))
    return out
end

function _polynomial(R, terms)
    ctx = MPolyBuildCtx(R)
    for (c, ev) in terms
        iszero(c) || push_term!(ctx, c, ev)
    end
    return finish(ctx)
end

# ===========================================================================
# Algebraic numbers: constants in Nemo's field of algebraic numbers, and back
# ===========================================================================
#
# A constant built from rationals, i, roots, rootof and the trigonometric functions at rational
# multiples of π has an exact value in QQBar, where equality and signs are decided exactly. Back
# in GIAC, a number of degree 2 is written with a square root and one of higher degree as a
# `rootof`. GIAC's `rootof(P, Q)` is P at the root of Q with the largest real part, then the
# largest imaginary part.

import Nemo: QQBarField, QQBarFieldElem

# A constant whose degree could exceed this (by a bound from its structure) is not computed:
# QQBar's arithmetic grows quickly with the degree
const _DEGREE_LIMIT = 512

"""
    to_nemo(e::GiacExpr, QQBar)

The algebraic number `e` in Nemo's field of algebraic numbers (`QQBar`, or
`algebraic_closure(QQ)`). See [`Giac.minimal_polynomial`](@ref) for the constants accepted.
"""
Giac.to_nemo(e::GiacExpr, K::QQBarField) = _algebraic(Giac._node(e), K)

_not_algebraic(n) = error("to_nemo: $(Giac._str(n)) is not an algebraic number")

_algebraic(n::Giac.NumNode, K) = _qqbar(K, n.v)
_algebraic(n::Giac.SymNode, K) = _not_algebraic(n)
_algebraic(n::Giac.GenNode, K) = _not_algebraic(n)
function _algebraic(n::Giac.AddNode, K)
    acc = zero(K)
    for (m, c) in n.terms
        acc += _qqbar(K, c) * _algebraic(m, K)
    end
    return acc
end
function _algebraic(n::Giac.MulNode, K)
    acc = _qqbar(K, n.coef)
    for (b, e) in n.factors
        acc *= _rational_power(_algebraic(b, K), e)
    end
    return acc
end
function _algebraic(n::Giac.CallNode, K)
    f, args = n.f, n.args
    f == "rootof" && return _rootof_value(n, K)
    if f == "surd" && length(args) == 2                  # the real root
        a, k = _algebraic(args[1], K), _whole(args[2])
        isreal(a) || _not_algebraic(n)
        return isodd(k) && is_negative(a) ? -root(-a, k) : root(a, k)
    end
    length(args) == 1 || _not_algebraic(n)
    if f in ("cos", "sin", "tan")
        r = _pi_multiple(args[1])
        r === nothing && _not_algebraic(n)
        q = K(_qq(r))
        return f == "cos" ? cospi(q) : f == "sin" ? sinpi(q) : tanpi(q)
    elseif f == "exp"
        r = _pi_multiple(args[1]; imaginary = true)
        r === nothing && _not_algebraic(n)
        return exp_pi_i(K(_qq(r)))
    end
    v = _algebraic(args[1], K)
    f == "sqrt" && return sqrt(v)
    f == "abs" && return abs(v)
    f in ("re", "real") && return real(v)
    f in ("im", "imag") && return imag(v)
    f == "conj" && return conj(v)
    _not_algebraic(n)
end

_qqbar(K, v::Union{Integer, Rational}) = K(_qq(v))
_qqbar(K, v::Complex) = _qqbar(K, real(v)) + _qqbar(K, imag(v)) * K(im)
_qqbar(K, v) = error("to_nemo: the number $v is not exact, so it has no algebraic value")

function _whole(n)
    (n isa Giac.NumNode && n.v isa Integer && n.v > 0) || error("to_nemo: $(Giac._str(n)) is not a positive whole number")
    return Int(n.v)
end

# a^e for a rational e, on the principal branch: the principal root, to the numerator's power
function _rational_power(a, e)
    (e isa Giac.NumNode && e.v isa Union{Integer, Rational}) ||
        error("to_nemo: the power $(Giac._str(e)) is not a rational number")
    q = Rational{BigInt}(e.v)
    r = isone(denominator(q)) ? a : root(a, Int(denominator(q)))
    k = Int(numerator(q))
    return k >= 0 ? r^k : inv(r)^(-k)
end

# r when n is r·π (or r·i·π with `imaginary`), r rational; nothing otherwise
function _pi_multiple(n; imaginary = false)
    n isa Giac.SymNode && n.name == "pi" && return imaginary ? nothing : 1
    n isa Giac.MulNode && length(n.factors) == 1 || return nothing
    (b, e) = only(n.factors)
    (b isa Giac.SymNode && b.name == "pi" && isequal(e, Giac._ONE)) || return nothing
    c = imaginary ? -im * n.coef : n.coef
    return isreal(c) && real(c) isa Union{Integer, Rational} ? real(c) : nothing
end

# GIAC's rootof(P, Q): P at the root of Q with the largest real part, then imaginary part
function _rootof_value(n::Giac.CallNode, K)
    lists = length(n.args) == 1 ? _list(only(n.args)) : n.args
    length(lists) == 2 || _not_algebraic(n)
    P, Q = (_coefficients(l, n) for l in lists)
    Qx, t = polynomial_ring(QQ, "t"; cached = false)
    γ = _principal(roots(K, sum(c * t^(length(Q) - k) for (k, c) in enumerate(Q))))
    return sum(K(c) * γ^(length(P) - k) for (k, c) in enumerate(P); init = zero(K))
end

_list(g::Giac.GenNode) = [Giac._node(x) for x in g.expr]
_list(n) = error("to_nemo: $(Giac._str(n)) is not a list")
_coefficients(l::Giac.Node, n) = _coefficients(_list(l), n)
_coefficients(l::AbstractVector, n) =
    [x isa Giac.NumNode && x.v isa Union{Integer, Rational} ? _qq(x.v) : _not_algebraic(n) for x in l]

_principal(rs) = reduce((a, b) -> _before(a, b) ? b : a, rs)
# whether b comes after a in GIAC's order: larger real part, then larger imaginary part
function _before(a, b)
    ra, rb = real(a), real(b)
    ra == rb ? imag(a) < imag(b) : ra < rb
end

# A bound on the degree of a constant, from its structure; nothing when it is not one
_degree_bound(n::Giac.NumNode) = n.v isa Complex && !iszero(imag(n.v)) ? 2 : 1
_degree_bound(n::Giac.Node) = nothing
function _degree_bound(n::Giac.AddNode)
    d = 1
    for (m, c) in n.terms
        b = _degree_bound(m)
        b === nothing && return nothing
        d = min(d * b * (c isa Complex && !iszero(imag(c)) ? 2 : 1), typemax(Int) ÷ 1024)
    end
    return d
end
function _degree_bound(n::Giac.MulNode)
    d = n.coef isa Complex && !iszero(imag(n.coef)) ? 2 : 1
    for (b, e) in n.factors
        db = _degree_bound(b)
        (db === nothing || !(e isa Giac.NumNode && e.v isa Union{Integer, Rational})) && return nothing
        d = min(d * db * Int(min(denominator(e.v), 1024)), typemax(Int) ÷ 1024)
    end
    return d
end
function _degree_bound(n::Giac.CallNode)
    if n.f == "rootof"
        lists = length(n.args) == 1 ? _list(only(n.args)) : n.args
        return length(lists) == 2 && lists[2] isa Giac.GenNode ? length(lists[2].expr) - 1 : nothing
    elseif n.f in ("cos", "sin", "tan", "exp") && length(n.args) == 1
        r = _pi_multiple(only(n.args); imaginary = n.f == "exp")
        return r === nothing ? nothing : 2 * Int(min(denominator(r), 1024))
    elseif n.f == "surd" && length(n.args) == 2
        b = _degree_bound(n.args[1])
        k = n.args[2]
        return b === nothing || !(k isa Giac.NumNode && k.v isa Integer) ? nothing : b * Int(min(k.v, 1024))
    elseif n.f in ("sqrt", "abs", "re", "real", "im", "imag", "conj") && length(n.args) == 1
        b = _degree_bound(only(n.args))
        return b === nothing ? nothing : 2b * (n.f == "conj" ? 1 : 2)
    end
    return nothing
end

# the value of a constant node in QQBar, or nothing when it is not an algebraic number of
# moderate degree
function _maybe_algebraic(n::Giac.Node, K = Nemo.QQBar)
    d = _degree_bound(n)
    (d === nothing || d > _DEGREE_LIMIT) && return nothing
    return try
        _algebraic(n, K)
    catch err
        err isa Union{ErrorException, ArgumentError, DomainError, InexactError} || rethrow()
        nothing
    end
end

function Giac._algebraic_sign(n::Giac.Node)
    v = _maybe_algebraic(n)
    (v === nothing || !isreal(v)) && return nothing
    return iszero(v) ? Giac._S_ZERO : is_positive(v) ? Giac._S_POS : Giac._S_NEG
end

"""
    to_giac(a::QQBarFieldElem)

An algebraic number as a `GiacExpr`: a rational, `a + b√d` for a number of degree 2, and otherwise
a `rootof`.
"""
function Giac.to_giac(a::QQBarFieldElem)
    is_rational(a) && return Giac.to_giac(QQFieldElem(a))
    Zx, _ = polynomial_ring(ZZ, "x"; cached = false)
    f = minpoly(Zx, a)
    degree(f) == 2 && return _quadratic(a, f)
    e = _radical(a, f)
    e === nothing || return e
    _principal(conjugates(a)) == a && return _rootof([QQ(0), QQ(1)], f)
    e = _principal_rootof(a, f, Zx)
    e === nothing || return e
    isreal(a) && error("to_giac: no rootof form found for $a")
    return Giac.to_giac(real(a)) + convert(GiacExpr, im) * Giac.to_giac(imag(a))
end

# GIAC's rootof names only the principal root of a polynomial, so another root a is written
# P(β), β = q(a) for a map q under which a is the one conjugate sent to the principal root and
# no two conjugates meet (β generates the same field). The maps tried are -x, ±(x - c)² and
# ±1/(x - c), with c simple rationals near a; P comes from linear algebra in the number field.
# The shortest result is kept; nothing if no map works.
function _principal_rootof(a, f, Zx)
    conj = conjugates(a)
    i = findfirst(==(a), conj)
    za = [ComplexF64(_approx(real(b)), _approx(imag(b))) for b in conj]
    K, α = number_field(change_base_ring(QQ, f), "α"; cached = false)
    best = nothing
    for (k, s, c) in _maps(za, i)
        # (dx - n)^k for c = n/d keeps an algebraic integer integral
        d = Int(denominator(c))
        q(z, c) = k == 0 ? s * inv(z - c) : s * (d * z - d * c)^k
        _numeric_principal(q.(za, Float64(c))) == i || continue
        βs = q.(conj, QQBarFieldElem(c))
        _principal(βs) == βs[i] && allunique(βs) || continue
        β = q(α, K(c))
        P = _coordinates(α, β, degree(f))
        P === nothing && continue
        e = _rootof(P, minpoly(Zx, βs[i]))
        (best === nothing || length(string(e)) < length(string(best))) && (best = e)
    end
    return best
end

# (power, sign, centre) for s(x - c)^k, k = 0 meaning s/(x - c): centres are 0 and simple
# rationals ever closer to a's real part, on both sides
function _maps(za, i)
    a = za[i]
    w = minimum(abs(a - b) for (j, b) in enumerate(za) if j != i)
    cs = QQFieldElem[QQ(0)]
    for d in (w / 2, w / 4, w / 8, w / 32, w / 128)
        lo, hi = real(a) - d, real(a) - d / 4
        push!(cs, _simplest(_rational_below(lo), _rational_above(hi)))
        push!(cs, _simplest(_rational_below(real(a) + d / 4), _rational_above(real(a) + d)))
    end
    unique!(cs)
    return [(k, s, c) for c in cs for k in (2, 0) for s in (-1, 1)] |> m -> vcat([(1, -1, QQ(0))], m)
end

# The index of the root with the largest real part, then imaginary part, in floating point;
# nothing when two are too close to tell
function _numeric_principal(zs)
    order = sortperm(zs; by = z -> (real(z), imag(z)), rev = true)
    length(zs) > 1 && isapprox(zs[order[1]], zs[order[2]]; rtol = 1e-9) && return nothing
    return order[1]
end

# the coordinates of α in powers of β, a generator of the same field: α = Σ P[j] β^(j-1)
function _coordinates(α, β, n)
    M = zero_matrix(QQ, n, n)
    p = one(parent(β))
    for j in 1:n
        for k in 1:n
            M[k, j] = coeff(p, k - 1)
        end
        p *= β
    end
    v = matrix(QQ, n, 1, [coeff(α, k - 1) for k in 1:n])
    ok, x = can_solve_with_solution(M, v; side = :right)
    return ok ? [x[j, 1] for j in 1:n] : nothing
end

# a real algebraic number as a Float64; a rational at most 2^-m below one
_approx(x) = Float64(_below(x, 60))
_below(x, m) = QQFieldElem(floor(x * QQBarFieldElem(ZZ(2)^m))) // ZZ(2)^m
_rational_below(x::Float64) = QQ(floor(BigInt, x * 2^20), ZZ(2)^20)
_rational_above(x::Float64) = QQ(ceil(BigInt, x * 2^20), ZZ(2)^20)

# the rational with the smallest denominator in [lo, hi], then the smallest in size
function _simplest(lo::QQFieldElem, hi::QQFieldElem)
    lo <= 0 <= hi && return QQ(0)
    hi < 0 && return -_simplest(-hi, -lo)
    n = floor(lo)
    (n == lo || n + 1 <= hi) && return QQ(ceil(lo))
    return n + inv(_simplest(inv(hi - n), inv(lo - n)))
end

# A root of a·xⁿ + b as a radical: the real root (b/a)^(1/n), times a root of unity of degree at
# most 2 for the others; nothing for any other polynomial
function _radical(a, f)
    n = degree(f)
    all(k -> iszero(coeff(f, k)), 1:(n - 1)) || return nothing
    r = -QQ(coeff(f, 0)) // QQ(coeff(f, n))
    ρ = root(QQBarFieldElem(abs(r)), n)
    ζ = a // ρ
    degree(ζ) <= 2 || return nothing
    return Giac.to_giac(ζ) * convert(GiacExpr, Rational{BigInt}(abs(r)))^(1 // n)
end

function _quadratic(a, f)
    c2, c1, c0 = (BigInt(coeff(f, k)) for k in 2:-1:0)
    mid, Δ = -c1 // 2c2, c1^2 - 4c2 * c0
    Δ < 0 && return convert(GiacExpr, mid) + (is_positive(imag(a)) ? 1 : -1) * sqrt(convert(GiacExpr, Δ)) / convert(GiacExpr, 2c2)
    s = a > QQBarFieldElem(QQ(mid)) ? 1 : -1
    return convert(GiacExpr, mid) + s * sqrt(convert(GiacExpr, Δ)) / convert(GiacExpr, 2c2)
end

# P(γ) for γ the principal root of the integer polynomial f, as `rootof`; P's coefficients from
# the constant term up. GIAC takes a monic integer polynomial, so γ is scaled by f's leading
# coefficient L (positive, which keeps it principal): Lγ is a root of L^(n-1) f(y/L)
function _rootof(P, f)
    n, L = degree(f), leading_coefficient(f)
    Q = [coeff(f, k) * L^(n - 1 - k) for k in 0:(n - 1)]
    P = [QQ(c) // QQ(L)^(j - 1) for (j, c) in enumerate(P)]
    while length(P) > 1 && iszero(last(P))
        pop!(P)
    end
    q(c) = isone(denominator(c)) ? string(numerator(c)) : string(numerator(c), "/", denominator(c))
    return giac_eval("rootof([" * join(q.(reverse(P)), ",") * "],[1," * join(string.(reverse(Q)), ",") * "])")
end

function Giac.algebraic_roots(p, x)
    p, x = convert(GiacExpr, p), convert(GiacExpr, x)
    others = setdiff(Giac._symbol_names(Giac._node(p)), (string(x),))
    isempty(others) || throw(ArgumentError("algebraic_roots: $p has variables other than $x: " * join(others, ", ")))
    rs = roots(Nemo.QQBar, Giac.to_nemo(p, polynomial_ring(QQ, string(x); cached = false)[1]))
    return GiacExpr[Giac.to_giac(r) for r in sort!(rs; lt = _root_order)]
end

# real roots first, in increasing order; then by real part, then imaginary part
function _root_order(a, b)
    isreal(a) != isreal(b) && return isreal(a)
    real(a) == real(b) ? imag(a) < imag(b) : real(a) < real(b)
end

Giac.minimal_polynomial(e, x) =_polynomial_in(_minpoly(Giac.to_nemo(convert(GiacExpr, e), Nemo.QQBar)), x)

_minpoly(v) = minpoly(polynomial_ring(ZZ, "x"; cached = false)[1], v)
_polynomial_in(f, x) = sum(convert(GiacExpr, BigInt(coeff(f, k))) * convert(GiacExpr, x)^k for k in degree(f):-1:0)

function Giac._rootof_tex(e::GiacExpr)
    v = _maybe_algebraic(Giac._node(e))
    (v === nothing || is_rational(v)) && return nothing
    w = minimum(_approx(abs(v - b)) for b in conjugates(v) if b != v)
    digits = max(3, ceil(Int, log10(4 / w)))
    r, i = round(_approx(real(v)); digits), round(_approx(imag(v)); digits)
    near = iszero(i) ? string(r) : (iszero(r) ? "" : string(r, i < 0 ? " - " : " + ")) * string(abs(i), " i")
    p = Giac.giac_latex(_polynomial_in(_minpoly(v), Giac.giac_identifier("x")))
    return "\\operatorname{root}_{\\approx " * near * "}\\left(" * p * "\\right)"
end

Giac.simplify_algebraic(e) = GiacExpr(_canonical(Giac._node(convert(GiacExpr, e))))

# whether a node can only be a number: its one name, if any, is π
_is_constant(n::Giac.Node) = all(==("pi"), Giac._symbol_names(n))

_canonical(n::Union{Giac.NumNode, Giac.SymNode, Giac.GenNode}) = n
function _canonical(n::Giac.Node)
    if _is_constant(n)
        v = _maybe_algebraic(n)
        if v !== nothing
            c = Giac._node(Giac.to_giac(v))
            (c isa Giac.NumNode || length(Giac._str(c)) <= length(Giac._str(n))) && return c
        end
    end
    return _rebuilt(n)
end
# the constant terms of a sum together, and the constant factors of a product; each part on its
# own when the whole is constant
function _rebuilt(n::Giac.AddNode)
    constant = [(m, c) for (m, c) in n.terms if _is_constant(m)]
    rest = [Giac._mul(Giac.NumNode(c), _canonical(m)) for (m, c) in n.terms if !_is_constant(m)]
    isempty(rest) && return Giac._add((Giac._mul(Giac.NumNode(c), _canonical(m)) for (m, c) in n.terms)...)
    isempty(constant) && return Giac._add(rest...)
    # the constant part where its first term was
    k = count(((m, c),) -> !_is_constant(m), n.terms[1:findfirst(((m, c),) -> _is_constant(m), n.terms)])
    part = _canonical(Giac._add((Giac._mul(Giac.NumNode(c), m) for (m, c) in constant)...))
    return Giac._add(rest[1:k]..., part, rest[(k + 1):end]...)
end
function _rebuilt(n::Giac.MulNode)
    isconst((b, e)) = _is_constant(b) && _is_constant(e)
    rest = [Giac._pow(_canonical(b), _canonical(e)) for (b, e) in n.factors if !isconst((b, e))]
    isempty(rest) && return Giac._mul(Giac.NumNode(n.coef), (Giac._pow(_canonical(b), _canonical(e)) for (b, e) in n.factors)...)
    constant = [Giac._pow(b, e) for (b, e) in n.factors if isconst((b, e))]
    return Giac._mul(_canonical(Giac._mul(Giac.NumNode(n.coef), constant...)), rest...)
end
_rebuilt(n::Giac.CallNode) = Giac._call(n.f, Giac.Node[_canonical(a) for a in n.args])
_rebuilt(n::Giac.Node) = n

# ===========================================================================
# Matrix algebra over rational functions
# ===========================================================================
#
# The determinant, inverse, rank and linear solves of a matrix whose entries are rational
# functions of its variables, in Nemo. A method returns nothing when an entry is anything else
# (a function call, a floating-point number, a root), and GIAC does the work instead.

# the entries as one Nemo ring's elements: the polynomial ring in their variables, or its fraction
# field with `field` or when an entry has a denominator; nothing if an entry is not a rational
# function
function _nemo_entries(es; field = false)
    vals = try
        Giac.to_nemo(GiacExpr[convert(GiacExpr, e) for e in es])
    catch err
        err isa ErrorException || rethrow()
        return nothing
    end
    isempty(vals) && return nothing
    F = parent(first(vals))
    field && !(F isa FracField) && (F = fraction_field(F))
    return F, [F(v) for v in vals]
end

function _nemo_matrix(A::AbstractMatrix; field = false)
    r = _nemo_entries(vec(A); field)
    r === nothing && return nothing
    F, vals = r
    return matrix(F, size(A, 1), size(A, 2), vec(permutedims(reshape(vals, size(A)))))
end

# a rational function with integer coefficients above and below, without a common factor: (b²+2)/(2b)
function _back(x::FracElem)
    N, D = numerator(x), denominator(x)
    is_constant(D) && return Giac.to_giac(N * inv(constant_coefficient(D)))
    cs = vcat(collect(coefficients(N)), collect(coefficients(D)))
    s = reduce(lcm, (denominator(c) for c in cs)) // reduce(gcd, (numerator(c) for c in cs))
    return Giac.to_giac(N * s) / Giac.to_giac(D * s)
end
_back(x) = Giac.to_giac(x)
_back(M::MatElem) = GiacExpr[_back(M[i, j]) for i in 1:nrows(M), j in 1:ncols(M)]

function Giac._exact_matrix(::Val{:det}, A)
    size(A, 1) == size(A, 2) || return nothing
    M = _nemo_matrix(A)
    return M === nothing ? nothing : _back(det(M))
end

function Giac._exact_matrix(::Val{:inv}, A)
    size(A, 1) == size(A, 2) || return nothing
    M = _nemo_matrix(A; field = true)
    (M === nothing || iszero(det(M))) && return nothing          # GIAC reports a singular one
    return _back(inv(M))
end

# the eigenvalues of a matrix of rational numbers, as `algebraic_roots` gives them
function Giac._exact_matrix(::Val{:eigvals}, A)
    size(A, 1) == size(A, 2) || return nothing
    n = size(A, 1)
    entries = [convert(GiacExpr, A[i, j]) for i in 1:n for j in 1:n]
    all(e -> isempty(Giac._symbol_names(Giac._node(e))) && Giac._node(e) isa Giac.NumNode, entries) || return nothing
    values = [Giac._node(e).v for e in entries]
    all(v -> v isa Union{Integer, Rational}, values) || return nothing
    Q = matrix(QQ, n, n, [_qq(v) for v in values])
    λ = [l for (l, m) in eigenvalues_with_multiplicities(Nemo.QQBar, Q) for _ in 1:m]
    return GiacExpr[Giac.to_giac(l) for l in sort!(λ; lt = _root_order)]
end

function Giac._exact_matrix(::Val{:rank}, A)
    M = _nemo_matrix(A; field = true)
    return M === nothing ? nothing : rank(M)
end

function Giac._exact_matrix(::Val{:solve}, A, b)
    size(A, 1) == size(A, 2) || return nothing
    B = b isa AbstractVector ? reshape(b, :, 1) : b
    r = _nemo_entries(vcat(vec(A), vec(B)); field = true)
    r === nothing && return nothing
    F, vals = r
    n, m = size(A, 1), size(B, 2)
    MA = matrix(F, n, n, vec(permutedims(reshape(vals[1:(n * n)], n, n))))
    MB = matrix(F, n, m, vec(permutedims(reshape(vals[(n * n + 1):end], n, m))))
    iszero(det(MA)) && return nothing
    X = _back(Nemo.solve(MA, MB; side = :right))
    return b isa AbstractVector ? vec(X) : X
end

# Equations linear in `unknowns`, with coefficients rational functions of the other variables,
# solved exactly: the value of each unknown by name. Nothing when they are not linear or their
# solution is not unique (GIAC then writes the family).
function Giac._exact_matrix(::Val{:linear}, eqs, unknowns)
    sides(e) = Giac.iscall(e) && Giac.symb_funcname(e) == "=" ? Giac.lhs(e) - Giac.rhs(e) : e
    r = _nemo_entries([sides(convert(GiacExpr, e)) for e in eqs])
    r === nothing && return nothing
    F, vals = r
    R = F isa FracField ? base_ring(F) : F
    names = symbols(R)
    ui = [findfirst(==(Symbol(string(u))), names) for u in unknowns]
    any(isnothing, ui) && return nothing                      # an unknown in no equation
    pk = [k for k in eachindex(names) if !(k in ui)]
    Rp, ps = polynomial_ring(QQ, names[pk]; cached = false)
    Fp = fraction_field(Rp)
    rows, rhs = Vector{Vector{elem_type(Fp)}}(), elem_type(Fp)[]
    for v in vals
        N, D = v isa FracElem ? (numerator(v), denominator(v)) : (v, one(v))
        any(k -> degree(D, k) > 0, ui) && return nothing      # an unknown in a denominator
        row, c0 = [zero(Fp) for _ in ui], zero(Fp)
        for (c, ev) in zip(coefficients(N), exponent_vectors(N))
            sum(ev[ui]) > 1 && return nothing                 # not linear
            mono = Fp(c * prod((ps[j]^ev[k] for (j, k) in enumerate(pk)); init = one(Rp)))
            j = findfirst(k -> ev[k] == 1, ui)
            j === nothing ? (c0 -= mono) : (row[j] += mono)
        end
        push!(rows, row)
        push!(rhs, c0)
    end
    A = matrix(Fp, length(rows), length(ui), reduce(vcat, rows))
    ok, x, K = can_solve_with_solution_and_kernel(A, matrix(Fp, length(rhs), 1, rhs); side = :right)
    ok || throw(ArgumentError("the equations have no solution for $(join(string.(unknowns), ", "))"))
    ncols(K) == 0 || return nothing
    return Dict(string(u) => _back(x[k, 1]) for (k, u) in enumerate(unknowns))
end

export to_giac, to_nemo

end # module GiacNemoExt