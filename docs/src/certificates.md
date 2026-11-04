# Exact certificates: positivity and decompositions

Giac is where an expression is written, rewritten and typeset. Some questions about it are better
answered by a specialist engine and then checked exactly: whether an expression is positive under
an ordering of its variables, and how to write it as a combination of given pieces. These tools do
the algebra in [Nemo](https://nemocas.github.io/Nemo.jl/) (FLINT), and the choice among solutions
with a linear program in JuMP and HiGHS. Every answer they return has been checked exactly over the
rationals; a floating-point solve only proposes.

```julia
using Giac, Nemo            # positivity, decompose
using JuMP, HiGHS           # certify
```

Polynomials and rational functions in any number of variables cross to Nemo by name; see
[Nemo.jl](extensions/nemo.md).

## Positivity under an ordering

`positivity(e; order, positive, param)` writes each variable of an ordering as the one before it
plus a positive gap, and reads the sign off the coefficients: a polynomial in positive variables
whose coefficients are all positive is positive.

```julia
@giac_var u t₁ t₂ t₃ l″ A A′ A₃
M = (t₃ - t₁) * (A′ + A₃) * (A * (t₂ - u) * (l″ - t₁) + A′ * (l″ - t₂) * (t₁ - u))
positivity(M; order = (u, t₁, t₂, t₃, l″), positive = (A, A′, A₃))
# Positivity: positive
#   numerator: positive; 28 terms (28 positive, 0 negative), degree 5
#   gaps: t₁ = u+Δt₁, t₂ = u+Δt₁+Δt₂, …
```

- `order` is a chain `(u, t₁, t₂)` for `u < t₁ < t₂`, or a vector of chains. A chain may start at
  a number: `(0, A, A₃)` is `0 < A < A₃`, `(1, g)` is `g > 1`.
- `positive` lists variables that are positive and in no chain.
- `param` substitutes first, for relations a chain cannot state: with `t₂² = l′u′`, write
  `param = (l′ => t₂/g, u′ => t₂*g)` and put `g` in a chain `(1, g)`.
- A rational function is put over a common denominator, and numerator and denominator are
  reported separately.

When the signs are mixed, the report says how many terms are against the majority, which variables
they contain and how often, and shows the largest. `Giac.group_terms(r.numerator.against, vars)`
groups them by their part in chosen variables. Mixed signs do not mean the expression changes
sign: `x² − 2xy + y²` is not decided by coefficients, though it is a square.

## Signs

`sign_of(e)` decides the sign of any expression as far as it can: `:positive`, `:negative`,
`:zero`, `:nonnegative`, `:nonpositive`, or `:unknown`. It reads what is known of the variables
from their declarations (`@giac_var k::positive`), assumptions in force (`assuming(k > 0)`) and
the `positive`/`negative` keywords; works through sums, products, powers and functions of known
sign (`exp`, `abs`, `sqrt`, `ln`, ...); and otherwise hands the polynomial skeleton, each
function call a variable of known sign, to `positivity`.

```julia
@giac_var x y k::positive
sign_of(log(1 + k))                     # :positive
sign_of(x^2 - 2x*y + y^2)               # :nonnegative, by a sum of squares
sign_of(sin(x)^2 - 2sin(x) + 1)         # :nonnegative
```

`simplify_signs(e)` uses it to resolve `abs`, `sign`, `max` and `min`:

```julia
@giac_var a::positive b::positive
simplify_signs(abs(a*b + 1))                # a*b+1
simplify_signs(max(a^2 + b^2, 2a*b))        # a^2+b^2
```

## Algebraic numbers

A constant built from rationals, `i`, `+ − × ÷`, rational powers, `sqrt`, `rootof`, `abs`, `re`,
`im`, `conj`, and `cos`, `sin`, `tan` at rational multiples of `π` is an algebraic number. With
Nemo loaded it is computed exactly in Nemo's field of algebraic numbers (Calcium), where equality
and signs are decided without floating point. `sign_of` uses it for constants, and for the
constant part of a sum:

```julia
using Giac, Nemo
@giac_var x
r(n) = sqrt(GiacExpr(n))
sign_of(r(2) + r(3) - r(10))                        # :negative
sign_of(giac_eval("cos(pi/7)") - 9//10)             # :positive
sign_of(x^2 + r(2) - 1)                             # :positive
```

`Giac.minimal_polynomial(e, x)` gives the minimal polynomial over the rationals,
`Giac.algebraic_roots(p, x)` the exact roots of a polynomial, including those that have no
expression in radicals, and `Giac.simplify_algebraic(e)` rewrites the constants in an expression
in a canonical form, which proves equal constants equal:

```julia
Giac.minimal_polynomial(giac_eval("cos(pi/7)"), x)              # 8*x^3-4*x^2-4*x+1
Giac.simplify_algebraic(giac_eval("(2+sqrt(5))^(1/3)") * x)     # (1/2+sqrt(5)/2)*x
Giac.simplify_algebraic(giac_eval("sqrt(2)*sqrt(3) - sqrt(6)")) # 0
Giac.algebraic_roots(x^5 - x - 1, x)                            # one real root, two complex pairs
```

A number of degree 2 is written with a square root, and a root of `a·xⁿ + b` as a radical. Any
other is a `rootof`. GIAC's `rootof` names the root of a polynomial with the largest real part,
so another root `a` is written `P(β)`, with `β = q(a)` for a map `q`, such as `−(x − c)²` or
`1/(x − c)`, that sends `a` alone to that root. In LaTeX a `rootof` is shown as the root of its
own minimal polynomial near a decimal, with enough digits to tell it from the other roots:
``\operatorname{root}_{\approx 0.347}\left(x^{3} - 3 x + 1\right)``.

## Decompositions

`decompose(target, blocks; degree, vars, param, products)` writes `target = Σ Mₖ Bₖ` with
unknown polynomial multipliers `Mₖ`, by matching coefficients exactly. The result holds one
solution and the free directions of the whole family.

```julia
@giac_var p r L
decompose((p*(L - r) + (L - p)*r)^2 - (p - r)^2 * L^2, [:Q => p*(L - p)*r*(L - r)])
# (4) · Q, unique
```

Each multiplier has total degree `degree` in `vars`, by default the target's degree less the
block's. Rows of the coefficient system with a single unknown are solved first, which removes
most unknowns of such systems, and the rest is solved densely over the rationals.

With `products = n`, every product of up to `n` blocks is a block, named by its factors, so the
search finds which products the target is made of.

## Choosing a certified member

`certify(d; order, positive, param, signs)` looks for a member of the family whose multipliers
each have the sign wanted, under the ordering. A linear program over the free directions proposes
weights, which are made rational and checked exactly. With `block_signs`, each multiplier takes
the sign that makes its term agree with the target:

```julia
@giac_var x y
d = decompose(x^3 - x^2*y - x*y^2 + y^3, [:x => x, :y => y, :s => (x - y)^2]; products = 3, degree = 0)
c = certify(d; positive = (x, y), block_signs = (1, 1, 1))
# x³ − x²y − xy² + y³ = x·s + y·s
```

`c.checks` holds the [`positivity`](@ref) report of each signed multiplier.

## Sums of squares

`sos_decomposition(p)` writes a polynomial as a weighted sum of squares, which proves it is
nonnegative everywhere, including where coefficient signs decide nothing. A semidefinite program
(Clarabel, through JuMP) proposes a Gram matrix; it is rounded to rationals, moved exactly onto
the polynomial's coefficients, and factored over the rationals.

```julia
using JuMP, Clarabel, Nemo
@giac_var x y
sos_decomposition(x^2 - 2x*y + 2y^2)
# Sum of squares, 2 squares:
#   x^2-2*x*y+2*y^2 = (x-y)²
#     + (y)²
```

It returns `nothing` when it finds no certificate: the polynomial may take negative values, or
be nonnegative without being a sum of squares (Motzkin's `x⁴y² + x²y⁴ − 3x²y² + 1`).

## Lyapunov functions

`lyapunov(f, x; degree)` looks for a polynomial `V` for `ẋ = f(x)` with `V − ε|x|²` and
`−V̇ − δ|x|²` both sums of squares, and certifies both exactly:

```julia
@giac_var x₁ x₂
L = lyapunov([-x₁ - x₁*x₂^2, -x₂ + x₁^2*x₂], [x₁, x₂])
# Lyapunov function, global asymptotic stability:
#   V  = …
#   V̇  = …
L.positive, L.decreasing        # the two certificates
```

`strict = false` asks only for `V̇ ≤ 0` (stability). A higher `degree` searches more
functions.

## Regions of attraction

When the origin is only locally stable, `region_of_attraction(f, x)` finds a level `c` with the
set `V ≤ c` inside its region of attraction: `−V̇ − ε|x|² − s·(c − V)` is a sum of squares for a
sum-of-squares multiplier `s`, so `V` decreases throughout the set. `c` is found by bisection,
and `V` defaults to the quadratic Lyapunov function of the linearization, `AᵀP + PA = −I` solved
exactly.

```julia
@giac_var x
region_of_attraction([-x + x^3], [x])
# Region of attraction contains V ≤ 244/489 (≈ 0.49897750511247446):
#   V  = x^2/2
#   V̇  = x^4-x^2
#   s  = 2*x^2, a sum of 1 square
#   −V̇ − 1/1000|x|² − s·(c − V) is a sum of 1 square
```

The exact region is `|x| < 1`, `V < 1/2`.

## Stopping a long computation

GIAC checks an interruption flag in its long loops. `Giac.with_timeout(f, seconds)` stops a GIAC
computation in `f` that runs longer, with a `GiacTimeout`; `Giac.interrupt()` stops the one
running now, from another task. Both need Julia started with more than one thread.

## API

```@docs
positivity
Giac.Positivity
Giac.SignReport
Giac.group_terms
sign_of
simplify_signs
Giac.minimal_polynomial
Giac.algebraic_roots
Giac.simplify_algebraic
decompose
Giac.Decomposition
Giac.Certified
Giac.combination
certify
sos_decomposition
Giac.SOSCertificate
lyapunov
Giac.LyapunovCertificate
region_of_attraction
Giac.AttractionCertificate
Giac.with_timeout
Giac.interrupt
```
