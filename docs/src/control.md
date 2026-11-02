# Nonlinear control

`Giac.Control` has the differential-geometric tools for control-affine systems
`ẋ = f(x) + Σ gᵢ(x) uᵢ`. A vector field is a vector of expressions in the state variables.

```julia
using Giac, Giac.Control
```

## Lie derivatives and brackets

```julia
@giac_var θ ω
f = [ω, -sin(θ)]
lie_derivative(θ^2, f, [θ, ω])                  # 2*θ*ω
lie_derivative(θ^2, f, [θ, ω]; order = 2)       # -2*θ*sin(θ)+2*ω^2

@giac_var px py φ
lie_bracket([cos(φ), sin(φ), 0], [0, 0, 1], [px, py, φ])   # [sin(φ), -cos(φ), 0]
```

`ad(f, g, x, k)` is the bracket with `f` taken `k` times.

## Accessibility and involutivity

`accessibility_rank(f, g, x)` is the rank of the distribution spanned by the input fields and
their brackets, generically or at a point (`at = (x => 0, ...)`). Equal to the number of states,
the system is locally accessible (the Lie algebra rank condition). The unicycle has two inputs and
three states, and reaches all three through a bracket:

```julia
gs = [[cos(φ), sin(φ), 0], [0, 0, 1]]
accessibility_rank([0, 0, 0], gs, [px, py, φ])     # 3
involutive(gs, [px, py, φ])                        # false
```

## Relative degree and input-output linearization

`relative_degree(h, f, g, x)` is the least `r` with `L_g L_f^{r-1} h ≠ 0`.
`io_linearization(h, f, g, x)` gives the coordinates `z = (h, L_f h, …, L_f^{r-1} h)` and the
feedback `u = α + β v` with `y⁽ʳ⁾ = v`:

```julia
io_linearization(θ, [ω, -sin(θ)], [0, 1], [θ, ω])
# (r = 2, z = [θ, ω], α = sin(θ), β = 1)
```

## Hurwitz stability

`hurwitz_conditions(p, s)` gives the Hurwitz determinants of a polynomial, or of a matrix's
characteristic polynomial; with a positive leading coefficient, all roots lie in the open left
half-plane exactly when every one is positive. With parameters in the coefficients they describe
the stable region. `hurwitz_stable` decides them with [`sign_of`](@ref), under the declared signs
and assumptions:

```julia
@giac_var s k a q::positive
hurwitz_conditions(s^3 + 2s^2 + k*s + a, s)     # [2, -a+2*k, -a^2+2*a*k]
hurwitz_stable(s^3 + 2s^2 + q*s + q, s)         # :stable
hurwitz_stable(GiacExpr[0 1; -2 -3], s)         # :stable
```

## Equilibria

With AlgebraicSolving loaded, `equilibria(f, x)` finds every real equilibrium of a polynomial
system and the number of eigenvalues of its linearization in the right half-plane. The values of
the Hurwitz determinants at each equilibrium are isolated along with it, so their signs, and the
count, are certain; an equilibrium with an eigenvalue possibly on the imaginary axis is reported
as not decided by the linearization.

```julia
using AlgebraicSolving
@giac_var x y
equilibria([y, x - x^3 - y], [x, y])
# (x ≈ -1.0, y ≈ 0.0): stable
# (x ≈ 0.0, y ≈ 0.0): unstable, 1 eigenvalue in the right half-plane
# (x ≈ 1.0, y ≈ 0.0): stable
```

For stability and regions of attraction, see `lyapunov` and `region_of_attraction` on the
[exact certificates](certificates.md) page; for how the number of equilibria depends on
parameters, `real_root_classification` on [polynomial systems](polynomial_systems.md).

## API

```@docs
Giac.Control
Giac.Control.lie_derivative
Giac.Control.lie_bracket
Giac.Control.ad
Giac.Control.accessibility_rank
Giac.Control.involutive
Giac.Control.relative_degree
Giac.Control.io_linearization
Giac.Control.hurwitz_conditions
Giac.Control.hurwitz_stable
Giac.Control.equilibria
Giac.Control.Equilibrium
```
