# Algebraic numbers: constants built from rationals, i, roots, `rootof` and the trigonometric
# functions at rational multiples of π are decided exactly in Nemo's field of algebraic numbers
# (GiacNemoExt). This file has the entry points; the extension adds their methods.

"""
    minimal_polynomial(e, x)

The minimal polynomial of the algebraic number `e` over the rationals, as a polynomial in `x`
with integer coefficients and positive leading coefficient. Needs Nemo to be loaded.

`e` is a constant built from integers, rationals, `i`, `+ - * /`, rational powers, `sqrt`,
`rootof`, `abs`, `re`, `im`, `conj`, and `cos`, `sin`, `tan` of a rational multiple of `π` or
`exp` of `i` times one.

```julia
using Giac, Nemo
@giac_var x
Giac.minimal_polynomial(sqrt(GiacExpr(2)) + sqrt(GiacExpr(3)), x)   # x^4-10*x^2+1
Giac.minimal_polynomial(giac_eval("cos(pi/7)"), x)              # 8*x^3-4*x^2-4*x+1
```
"""
function minimal_polynomial end

"""
    algebraic_roots(p, x)

The roots of the polynomial `p` in `x`, with rational coefficients, as exact algebraic numbers,
each as often as its multiplicity: the real roots in increasing order, then the others by real
part and then imaginary part. Each is written as by [`Giac.simplify_algebraic`](@ref): a rational,
`a + b√d`, a radical, or a `rootof`, which typesets as the root of its minimal polynomial near a
decimal. Needs Nemo to be loaded.

```julia
using Giac, Nemo
@giac_var x
Giac.algebraic_roots(x^3 - 3x + 1, x)   # three real roots, each a rootof
Giac.algebraic_roots(x^5 - x - 1, x)    # one real root and two complex pairs
```
"""
function algebraic_roots end

"""
    simplify_algebraic(e)

`e` with each algebraic constant in it decided exactly and written in a canonical form, where
that is no longer than the constant as written: a rational, `a + b√d` for a number of degree 2,
a radical for a root of `a·xⁿ + b`, or a `rootof`. A constant that is zero becomes 0. Needs Nemo
to be loaded.

```julia
using Giac, Nemo
@giac_var x
Giac.simplify_algebraic(giac_eval("(2+sqrt(5))^(1/3)") * x)      # (1/2+sqrt(5)/2)*x
Giac.simplify_algebraic(giac_eval("cos(pi/5) - sin(pi/10)"))     # 1/2
```
"""
function simplify_algebraic end

# The exact sign of a constant node, as a `_Signs`, or nothing when it is not a real algebraic
# number; the Nemo extension adds the method
function _algebraic_sign end

# A `rootof` written as the root of its minimal polynomial near a decimal, with enough digits to
# tell it from the other roots, or nothing; the Nemo extension adds the method
function _rootof_tex end
