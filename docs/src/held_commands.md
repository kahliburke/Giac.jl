# Held Commands

A held command is a command left unevaluated: an expression like any other, written in the
command's notation (∫, d/dx, lim, Σ, ℒ, …). It can be part of sums and products, stand on either
side of an equation, and be substituted into; `release` evaluates it.

```julia
using Giac
using Giac.Commands: hold_cmd, release

@giac_var x t s n z a

h = hold_cmd(:integrate, x^2, x)     # ∫ x² dx
release(h)                           # x³/3
h ~ release(h)                       # ∫ x² dx = x³/3
2 * hold_cmd(:limit, sin(x)/x, x, 0) + 1
substitute(hold_cmd(:sum, a*n, n, 1, 3), a => 2)   # Σ 2n, released: 12
```

A held integral is the same expression [`integral`](@ref) builds, so [`by_parts`](@ref) and
[`integrand`](@ref) work on it too.

## Specialized LaTeX Rendering

The following commands render with standard mathematical notation in LaTeX-capable environments (Jupyter, Pluto):

### Integration

```julia
# Indefinite integral: ∫ f dx
hold_cmd(:integrate, sin(x), x)

# Definite integral: ∫₀¹ f dx
hold_cmd(:integrate, x^2, x, 0, 1)
```

### Differentiation (Leibniz notation)

```julia
# First derivative: d/dx f
hold_cmd(:diff, x^3, x)

# Higher-order: d²/dx² f
hold_cmd(:diff, x^3, x, 2)
```

### Laplace Transform

```julia
# Forward: ℒ{f}(s)
hold_cmd(:laplace, exp(-t), t, s)

# Inverse: ℒ⁻¹{F}(t)
hold_cmd(:invlaplace, 1/(s+1), s, t)
# Also works with :ilaplace
```

### Z-Transform

```julia
# Forward: 𝒵{f}(z)
hold_cmd(:ztrans, n^2, n, z)

# Inverse: 𝒵⁻¹{F}(n)
hold_cmd(:invztrans, z/(z-1), z, n)
```

### Limits

```julia
# Basic limit: lim_{x→0} sin(x)/x
hold_cmd(:limit, sin(x)/x, x, 0)

# Limit at infinity: lim_{x→+∞} 1/x
hold_cmd(:limit, 1/x, x, Inf)

# One-sided limits (direction: 1 for right, -1 for left)
hold_cmd(:limit, sign(x), x, 0, 1)   # lim_{x→0⁺}
hold_cmd(:limit, sign(x), x, 0, -1)  # lim_{x→0⁻}
```

### Sums and Products

```julia
# Finite sum: Σ_{n=1}^{17} 1/n²
hold_cmd(:sum, 1/n^2, n, 1, 17)

# Infinite sum (Basel problem): Σ_{n=1}^{∞} 1/n²
hold_cmd(:sum, 1/n^2, n, 1, Inf)

# Product: Π_{k=1}^{n} k
hold_cmd(:product, k, k, 1, n)
```

### Riemann Sums

```julia
# Renders as: lim_{n→+∞} Σ_{k=0}^{n-1} 1/(n+k)
hold_cmd(:sum_riemann, 1/(n+k), [n, k])
```

## Generic Commands

Commands without specialized rendering use function-call notation:

```julia
hold_cmd(:factor, x^2 - 1)     # Renders as: factor(x² - 1)
hold_cmd(:simplify, sin(x)^2)  # Renders as: simplify(sin²(x))
```

## Release

`release` evaluates every held command in an expression, from the inside out. A held command on
its own gives GIAC's result as GIAC returns it, so `release(hold_cmd(:ifactor, 120))` is `2^3*3*5`.

```julia
release(hold_cmd(:integrate, x^2, x) + x)   # x³/3 + x
```

## Equations with `~`

`~` writes an equation between any two expressions, held or not, without evaluating either side:

```julia
using Giac.Commands: eigenvals

M = GiacMatrix([[1, 2], [3, 4]])
hold_cmd(:eigenvals, M) ~ eigenvals(M)
hold_cmd(:factor, x^2 - 1) ~ hold_cmd(:expand, (x-1)*(x+1))
hold_cmd(:det, M) ~ -2
```

## API Reference

```@docs
Giac.Commands.hold_cmd
Giac.Commands.release
```
