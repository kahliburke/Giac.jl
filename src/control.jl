"""
    Giac.Control

Differential-geometric tools for systems `ẋ = f(x) + Σ gᵢ(x) uᵢ`: Lie derivatives and brackets,
the accessibility rank, involutivity, relative degree and input-output linearization. Vector
fields are vectors of expressions in the state variables `x`.

```julia
using Giac, Giac.Control
@giac_var x₁ x₂ x₃
f = [x₂, -sin(x₁), 0]
g = [0, 1, x₁]
lie_bracket(f, g, [x₁, x₂, x₃])
```
"""
module Control

using ..Giac
using ..Giac: GiacExpr, GiacMatrix, Commands, jacobian, substitute, sign_of

export lie_derivative, lie_bracket, ad, accessibility_rank, involutive, relative_degree, io_linearization
export hurwitz_conditions, hurwitz_stable, equilibria, stable

_field(v) = GiacExpr[convert(GiacExpr, e) for e in v]
_simplified(e::GiacExpr) = Commands.simplify(e)
_is_zero(e::GiacExpr; at = ()) = iszero(_simplified(isempty(at) ? e : substitute(e, at...)))

"""
    lie_derivative(h, f, x; order = 1)

`L_f h = ∇h · f`, the derivative of `h` along the vector field `f`, applied `order` times. For a
vector `h`, each component's.
"""
function lie_derivative(h, f, x; order::Integer = 1)
    h isa AbstractVector && return [lie_derivative(hi, f, x; order) for hi in h]
    r = convert(GiacExpr, h)
    F, X = _field(f), _field(x)
    for _ in 1:order
        r = _simplified(sum(Commands.diff(r, xi) * fi for (xi, fi) in zip(X, F)))
    end
    return r
end

"""
    lie_bracket(f, g, x)

`[f, g] = (∂g/∂x) f − (∂f/∂x) g`, the Lie bracket of two vector fields.
"""
function lie_bracket(f, g, x)
    F, G, X = _field(f), _field(g), _field(x)
    return _simplified.(jacobian(G, X) * F - jacobian(F, X) * G)
end

"""
    ad(f, g, x, k = 1)

`ad_f^k g = [f, [f, … [f, g]]]`, the bracket with `f` taken `k` times (`ad(f, g, x, 0)` is `g`).
"""
function ad(f, g, x, k::Integer = 1)
    r = _field(g)
    for _ in 1:k
        r = lie_bracket(f, r, x)
    end
    return r
end

# a matrix whose columns are the vector fields, with `at` substituted
_columns(fields, at) =
    GiacMatrix(GiacExpr[isempty(at) ? v[i] : substitute(v[i], at...) for i in eachindex(first(fields)), v in fields])
_rank(fields; at = ()) = isempty(fields) ? 0 : Int(Giac.to_julia(Commands.invoke_cmd(:rank, _columns(fields, at))))

_inputs(g) = first(g) isa AbstractVector ? [_field(gi) for gi in g] : [_field(g)]

"""
    accessibility_rank(f, g, x; depth = length(x), at = ())

The rank of the distribution spanned by the input fields `g` (one vector, or a vector of them)
and their brackets with `f` and with each other, up to `depth` brackets deep: generically, or at
the point `at` (pairs `xᵢ => value`). Equal to the number of states, the system is locally
accessible there (the Lie algebra rank condition).
"""
function accessibility_rank(f, g, x; depth::Integer = length(x), at = ())
    F, X = _field(f), _field(x)
    fields = _inputs(g)
    level = fields
    for _ in 1:depth
        _rank(fields; at) == length(X) && break
        next = Vector{GiacExpr}[]
        for v in level, w in vcat([F], fields)
            b = lie_bracket(w, v, X)
            all(_is_zero, b) || push!(next, b)
        end
        isempty(next) && break
        append!(fields, next)
        level = next
    end
    return _rank(fields; at)
end

"""
    involutive(fields, x)

Whether the distribution spanned by `fields` is involutive: every bracket of two of them lies in
their span (generic rank).
"""
function involutive(fields, x)
    D = [_field(v) for v in fields]
    r = _rank(D)
    for i in eachindex(D), j in (i + 1):length(D)
        _rank(vcat(D, [lie_bracket(D[i], D[j], x)])) > r && return false
    end
    return true
end

"""
    relative_degree(h, f, g, x; at = ())

The relative degree of the output `y = h(x)` for `ẋ = f(x) + g(x) u`: the least `r` with
`L_g L_f^{r-1} h ≠ 0`, generically or at the point `at`; `nothing` if there is none up to the
number of states.
"""
function relative_degree(h, f, g, x; at = ())
    Lf = convert(GiacExpr, h)
    for r in 1:length(x)
        _is_zero(lie_derivative(Lf, g, x); at) || return r
        Lf = lie_derivative(Lf, f, x)
    end
    return nothing
end

"""
    io_linearization(h, f, g, x)

Input-output linearization of `y = h(x)`, `ẋ = f(x) + g(x) u`: a named tuple with the relative
degree `r`, the coordinates `z = (h, L_f h, …, L_f^{r-1} h)`, and `α`, `β` such that the
feedback `u = α + β v` makes `y⁽ʳ⁾ = v`:

    α = −L_f^r h / (L_g L_f^{r-1} h),    β = 1 / (L_g L_f^{r-1} h)

```julia
@giac_var x₁ x₂
io_linearization(x₁, [x₂, -sin(x₁)], [0, 1], [x₁, x₂])   # r = 2, α = sin(x₁), β = 1
```
"""
function io_linearization(h, f, g, x)
    r = relative_degree(h, f, g, x)
    r === nothing && throw(ArgumentError("io_linearization: the output has no relative degree"))
    z = [lie_derivative(h, f, x; order = k) for k in 0:(r - 1)]
    k = lie_derivative(last(z), g, x)
    Lr = lie_derivative(last(z), f, x)
    return (r = r, z = z, α = _simplified(-Lr / k), β = _simplified(1 / k))
end

# ---------------------------------------------------------------------------------------------
# Hurwitz stability
# ---------------------------------------------------------------------------------------------

_polynomial(p, s) = p isa AbstractMatrix ? Commands.invoke_cmd(:charpoly, GiacMatrix(GiacExpr.(p)), s) : convert(GiacExpr, p)

"""
    hurwitz_conditions(p, s)

The Hurwitz determinants `Δ₁, …, Δₙ` of the polynomial `p` in `s` (or of the characteristic
polynomial of a square matrix `p`), simplified. With the leading coefficient positive, all roots
of `p` lie in the open left half-plane exactly when every `Δₖ` is positive. The coefficients may
hold parameters, so the conditions describe where the system is stable.

```julia
@giac_var s k a
hurwitz_conditions(s^3 + 2s^2 + k*s + a, s)    # [2, -a+2*k, -a^2+2*a*k]
```
"""
function hurwitz_conditions(p, s)
    P = _polynomial(p, s)
    n = Int(Giac.to_julia(Commands.invoke_cmd(:degree, P, s)))
    n >= 1 || throw(ArgumentError("hurwitz_conditions: the polynomial has no roots"))
    a(k) = 0 <= k <= n ? Commands.invoke_cmd(:coeff, P, s, k) : convert(GiacExpr, 0)
    H = GiacExpr[a(n - 2j + i) for i in 1:n, j in 1:n]
    return [Commands.normal(Giac.LinearAlgebra.det(H[1:k, 1:k])) for k in 1:n]
end

"""
    hurwitz_stable(p, s; positive = (), negative = ())

Whether every root of the polynomial `p` in `s` (or every eigenvalue of a square matrix `p`)
lies in the open left half-plane: `:stable`, `:unstable` (some root does not), or `:unknown`. The
signs of the leading coefficient and of the [`hurwitz_conditions`](@ref) are decided by
[`sign_of`](@ref), under the variables' declarations, the assumptions in force and the
`positive` and `negative` hints.

```julia
@giac_var s k::positive
hurwitz_stable(s^2 + k*s + 1, s)      # :stable
```
"""
function hurwitz_stable(p, s; positive = (), negative = (), sos = true)
    P = _polynomial(p, s)
    n = Int(Giac.to_julia(Commands.invoke_cmd(:degree, P, s)))
    lead = sign_of(Commands.invoke_cmd(:coeff, P, s, n); positive, negative, sos)
    lead in (:positive, :negative) || return :unknown
    lead === :negative && (P = -P)
    signs = [sign_of(Δ; positive, negative, sos) for Δ in hurwitz_conditions(P, s)]
    all(==(:positive), signs) && return :stable
    any(in((:negative, :nonpositive, :zero)), signs) && return :unstable
    return :unknown
end

# ---------------------------------------------------------------------------------------------
# Equilibria
# ---------------------------------------------------------------------------------------------

"""
    Equilibrium

A real equilibrium from [`equilibria`](@ref): `point`, a `Giac.RealSolution` (each state in an
interval guaranteed to contain it), and `unstable`, the number of eigenvalues of the
linearization in the open right half-plane, or `nothing` when one may lie on the imaginary axis
(the linearization does not decide). `stable(e)` is `unstable == 0`.
"""
struct Equilibrium
    point::Giac.RealSolution
    unstable::Union{Int, Nothing}
end

stable(e::Equilibrium) = e.unstable == 0

function Base.show(io::IO, e::Equilibrium)
    print(io, e.point, ": ", e.unstable === nothing ? "not decided by the linearization" :
          e.unstable == 0 ? "stable" : string("unstable, ", e.unstable, e.unstable == 1 ? " eigenvalue" : " eigenvalues",
                                              " in the right half-plane"))
end
Base.show(io::IO, ::MIME"text/plain", e::Equilibrium) = show(io, e)

"""
    equilibria(f, x; precision = 64)

Every real equilibrium of `ẋ = f(x)` (polynomial, rational coefficients), each with its
stability from the linearization: a vector of [`Equilibrium`](@ref)s. The equilibria are
isolated exactly (msolve), and so are the values of the Hurwitz determinants of the Jacobian
there, so the count of eigenvalues in the right half-plane (the sign changes of Routh's first
column) is certain. Needs AlgebraicSolving.

```julia
using Giac, Giac.Control, AlgebraicSolving
@giac_var x y
equilibria([y, x - x^3 - y], [x, y])
# (x ≈ -1.0, y ≈ 0.0): stable
# (x ≈ 0.0, y ≈ 0.0): unstable, 1 eigenvalue in the right half-plane
# (x ≈ 1.0, y ≈ 0.0): stable
```
"""
function equilibria end

end # module
