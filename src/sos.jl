# Sums of squares: a polynomial written as a weighted sum of squares is nonnegative everywhere,
# and a polynomial Lyapunov function with both conditions written so is a proof of stability.
# A semidefinite program proposes the squares in floating point (GiacClarabelExt, with JuMP and
# Clarabel); they are made rational and checked exactly in Nemo, so a returned certificate does
# not rest on the floating-point solve.

"""
    SOSCertificate

`polynomial = Σₖ weights[k] * squares[k]^2`, exactly, with every weight a positive rational:
a proof that `polynomial ≥ 0` for all real values of its variables. `GiacExpr(c)` is the sum
written out; `expand(GiacExpr(c) - c.polynomial)` is zero.
"""
struct SOSCertificate
    polynomial::GiacExpr
    weights::Vector{Rational{BigInt}}
    squares::Vector{GiacExpr}
end

function GiacExpr(c::SOSCertificate)
    isempty(c.weights) && return convert(GiacExpr, 0)
    return sum(convert(GiacExpr, w) * q^2 for (w, q) in zip(c.weights, c.squares))
end

_rational_text(w) = isone(denominator(w)) ? string(numerator(w)) : string(numerator(w), "/", denominator(w))

function Base.show(io::IO, ::MIME"text/plain", c::SOSCertificate)
    println(io, "Sum of squares, ", length(c.squares), length(c.squares) == 1 ? " square" : " squares", ":")
    print(io, "  ", c.polynomial, " =")
    isempty(c.squares) && print(io, " 0")
    for (i, (w, q)) in enumerate(zip(c.weights, c.squares))
        weight = isone(w) ? "" : _rational_text(w) * "·"
        print(io, i == 1 ? " " : "\n    + ", weight, "(", q, ")²")
    end
end
Base.show(io::IO, c::SOSCertificate) = print(io, "SOSCertificate(", length(c.squares), " squares)")

"""
    sos_decomposition(p; vars = nothing, bits = 48)

`p` as a sum of squares of polynomials with positive rational weights, a
[`Giac.SOSCertificate`](@ref), or `nothing` when none is found. A certificate proves `p ≥ 0`
everywhere. Needs JuMP, Clarabel and Nemo to be loaded.

The squares are found by a semidefinite program over the monomials that can occur in them,
solved in floating point, then rounded to rationals, projected exactly onto the polynomial's
coefficients and factored exactly; the identity is checked in Nemo. Roundings from 8 to `bits`
bits are tried in turn. Not every nonnegative polynomial is a sum of squares (Motzkin's is not),
and one whose squares must vanish together somewhere may not round to an exact certificate.

```julia
using Giac, JuMP, Clarabel, Nemo
@giac_var x y
sos_decomposition(x^2 - 2x*y + 2y^2)
# Sum of squares, 2 squares:
#   x^2-2*x*y+2*y^2 = (x-y)² + (y)²
```
"""
function sos_decomposition end

"""
    LyapunovCertificate

A polynomial Lyapunov function `V` for `ẋ = f(x)` at the origin, from [`lyapunov`](@ref):
`positive` proves `V − ε|x|² ≥ 0` and `decreasing` proves `−V̇ − δ|x|² ≥ 0`, both as
[`Giac.SOSCertificate`](@ref)s. With `δ > 0` the origin is globally asymptotically stable; with
`δ = 0`, stable.
"""
struct LyapunovCertificate
    V::GiacExpr
    Vdot::GiacExpr
    ε::Rational{BigInt}
    δ::Rational{BigInt}
    positive::SOSCertificate
    decreasing::SOSCertificate
end

function Base.show(io::IO, ::MIME"text/plain", c::LyapunovCertificate)
    println(io, "Lyapunov function, ", c.δ > 0 ? "global asymptotic stability" : "stability", ":")
    println(io, "  V  = ", c.V)
    println(io, "  V̇  = ", c.Vdot)
    println(io, "  V − ", _rational_text(c.ε), "|x|² is a sum of ", length(c.positive.squares), " squares")
    print(io, "  −V̇", c.δ > 0 ? string(" − ", _rational_text(c.δ), "|x|²") : "", " is a sum of ",
          length(c.decreasing.squares), " squares")
end
Base.show(io::IO, c::LyapunovCertificate) = print(io, "LyapunovCertificate(V = ", c.V, ")")

"""
    lyapunov(f, x; degree = 2, strict = true, ε = 1//1000)

A polynomial Lyapunov function for `ẋ = f(x)` at the origin, certified exactly, as a
[`Giac.LyapunovCertificate`](@ref), or `nothing` when none of that degree is found. `f` is a
vector of polynomials in the state variables `x`, with `f(0) = 0`. `V` has terms of degree 2 to
`degree` (even), and is found with `V − ε|x|²` and `−V̇ − δ|x|²` both sums of squares, `δ = ε`
when `strict` (asymptotic stability) and `0` otherwise. Needs JuMP, Clarabel and Nemo.

```julia
@giac_var x₁ x₂
c = lyapunov([-x₁ + x₂, -x₁ - x₂^3], [x₁, x₂]; degree = 2)
c.V
```
"""
function lyapunov end
