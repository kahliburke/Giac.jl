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
export hurwitz_conditions, hurwitz_stable, equilibria, stable, center_manifold

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

# ---------------------------------------------------------------------------------------------
# Center manifolds
# ---------------------------------------------------------------------------------------------

"""
    CenterManifold

From [`center_manifold`](@ref): the manifold `y = h(x)` as pairs `yⱼ => hⱼ(x)` up to the order
asked, the `reduced` dynamics `ẋ` on it, and for a one-dimensional center the `stability` of
the origin it implies: `:stable` (asymptotically), `:unstable`, or `:unknown`. When the
coordinates were changed to separate the center and stable directions, `coordinates` gives each
original state in the new ones (`xᵢ => …`), and `h` and `reduced` are in the new coordinates.
"""
struct CenterManifold
    center::Vector{GiacExpr}
    h::Vector{Pair{GiacExpr, GiacExpr}}
    reduced::Vector{GiacExpr}
    order::Int
    stability::Symbol
    coordinates::Vector{Pair{GiacExpr, GiacExpr}}
end

function Base.show(io::IO, ::MIME"text/plain", m::CenterManifold)
    println(io, "Center manifold to order ", m.order, ":")
    isempty(m.coordinates) || println(io, "  in coordinates with ", join((string(x, " = ", e) for (x, e) in m.coordinates), ", "))
    for (y, h) in m.h
        println(io, "  ", y, " = ", h)
    end
    for (x, r) in zip(m.center, m.reduced)
        println(io, "  d", x, "/dt = ", r)
    end
    print(io, "  the origin is ", m.stability === :stable ? "asymptotically stable" :
              m.stability === :unstable ? "unstable" : "not decided at this order")
end
Base.show(io::IO, m::CenterManifold) = print(io, "CenterManifold(order ", m.order, ", ", m.stability, ")")

# the exponent vectors of total degree k in n variables
_exponents(n, k) = n == 1 ? [[k]] : [vcat(i, e) for i in k:-1:0 for e in _exponents(n - 1, k - i)]

_monomial(xs, α) = prod((x^a for (x, a) in zip(xs, α) if a > 0); init = convert(GiacExpr, 1))

# p's terms of total degree k in xs: monomial exponent => coefficient (which may hold parameters)
function _graded_terms(p, xs, k)
    out = Pair{Vector{Int}, GiacExpr}[]
    for α in _exponents(length(xs), k)
        c = Commands.invoke_cmd(:coeff, p, xs, α)
        iszero(c) || push!(out, α => c)
    end
    return out
end

_truncated(p, xs, order) =
    sum((c * _monomial(xs, α) for k in 0:order for (α, c) in _graded_terms(p, xs, k)); init = convert(GiacExpr, 0))

"""
    center_manifold(f, x; center = nothing, order = 3)

The center manifold of `ẋ = f(x)` at the origin, to order `order`, and the dynamics on it. `f`
is polynomial. With `center`, the variables of the center directions, the others are the stable
ones and the linear part must already be in block form, with no linear coupling between the two
groups; the coefficients may then hold parameters. Without it, the coordinates are changed
first: the linear part's characteristic polynomial is factored over the rationals into the
factors with roots on the imaginary axis and the Hurwitz ones, and new variables `ξ` (center) and
`η` (stable) are taken along the kernels of the two products, exactly; `coordinates` in the
result gives `x` in them. The linear part's coefficients must then be numbers and it may have
no eigenvalue in the right half-plane. The manifold `y = h(x)` is found order by order
from the invariance equation, each order a linear system; a [`CenterManifold`](@ref) holds it,
the reduced dynamics, and for one center variable the stability the leading term gives, its sign
decided by [`sign_of`](@ref).

```julia
@giac_var x y
m = center_manifold([x*y, -y + x^2], [x, y]; center = [x])
m.h            # [y => x^2]
m.reduced      # [x^3]: the origin is unstable

center_manifold([-x + y - x^3, x - y], [x, y])  # coupled linear part: x = ξ₁ + η₁, y = ξ₁ - η₁
```
"""
function center_manifold(f, x; center = nothing, order::Integer = 3)
    center === nothing && return _center_manifold_by_blocks(f, x, order)
    F, X = _field(f), _field(x)
    C = _field(center)
    names = string.(X)
    ci = [something(findfirst(==(string(c)), names), 0) for c in C]
    0 in ci && throw(ArgumentError("center_manifold: a center variable is not a state variable"))
    si = setdiff(eachindex(X), ci)
    S = X[si]
    order >= 2 || throw(ArgumentError("center_manifold: order must be at least 2"))
    zero_at = [v => convert(GiacExpr, 0) for v in X]
    A = GiacExpr[_simplified(substitute(Commands.diff(F[i], X[j]), zero_at...)) for i in eachindex(F), j in eachindex(X)]
    all(iszero, A[ci, si]) && all(iszero, A[si, ci]) ||
        throw(ArgumentError("center_manifold: the linear part couples the center and stable variables; " *
                            "change coordinates to block form first"))

    H = Dict{String, GiacExpr}(string(y) => convert(GiacExpr, 0) for y in S)
    on_manifold(e, Hs) = substitute(e, (y => Hs[string(y)] for y in S)...)
    for k in 2:order
        # the unknown coefficients of degree k, and the invariance equation's degree-k terms
        unknowns = GiacExpr[]
        trial = Dict{String, GiacExpr}()
        for (j, y) in enumerate(S)
            terms = GiacExpr[]
            for α in _exponents(length(C), k)
                c = Giac.giac_identifier("cm_$(j)_" * join(α, "_"))
                push!(unknowns, c)
                push!(terms, c * _monomial(C, α))
            end
            trial[string(y)] = H[string(y)] + sum(terms)
        end
        eqs = GiacExpr[]
        for (j, y) in enumerate(S)
            hy = trial[string(y)]
            lhs = sum(Commands.diff(hy, C[i]) * on_manifold(F[ci[i]], trial) for i in eachindex(C))
            residual = Commands.expand(lhs - on_manifold(F[si[j]], trial))
            append!(eqs, last.(_graded_terms(residual, C, k)))
        end
        values = isempty(eqs) ? fill(convert(GiacExpr, 0), length(unknowns)) :
                 collect(Commands.invoke_cmd(:linsolve, eqs, unknowns))
        length(values) == length(unknowns) ||
            throw(ArgumentError("center_manifold: no solution at order $k (is the stable block's linear part invertible?)"))
        solved = [u => v for (u, v) in zip(unknowns, values)]
        for y in S
            H[string(y)] = Commands.expand(substitute(trial[string(y)], solved...))
        end
    end
    reduced = [Commands.expand(_truncated(Commands.expand(on_manifold(F[i], H)), C, order)) for i in ci]
    stability = length(C) == 1 ? _leading_stability(reduced[1], C[1], order) : :unknown
    return CenterManifold(C, [y => H[string(y)] for y in S], reduced, order, stability, Pair{GiacExpr, GiacExpr}[])
end

# The coordinates x = T z that split the linear part into its center and stable blocks: the
# kernels of the products of the characteristic polynomial's factors with roots on the imaginary
# axis and of the Hurwitz ones. Then the reduction in z.
function _center_manifold_by_blocks(f, x, order)
    F, X = _field(f), _field(x)
    n = length(X)
    zero_at = [v => convert(GiacExpr, 0) for v in X]
    A = GiacExpr[_simplified(substitute(Commands.diff(F[i], X[j]), zero_at...)) for i in 1:n, j in 1:n]
    all(a -> Giac.to_julia(a) isa Number, A) ||
        throw(ArgumentError("center_manifold: the linear part has parameters; pass `center` for a system in block form"))
    s = Giac.giac_identifier("cm_s")
    p = Commands.invoke_cmd(:charpoly, GiacMatrix(A), s)
    fl = collect(Commands.invoke_cmd(:factors, p))
    pc, ps = convert(GiacExpr, 1), convert(GiacExpr, 1)
    for k in 1:2:length(fl)
        q, m = fl[k], fl[k + 1]
        Int(Giac.to_julia(Commands.invoke_cmd(:degree, q, s))) == 0 && continue
        if hurwitz_stable(q, s) === :stable
            ps *= q^m
        elseif _roots_on_axis(q, s)
            pc *= q^m
        else
            throw(ArgumentError("center_manifold: the linearization has an eigenvalue in the right half-plane " *
                                "(a root of $q): the origin is unstable"))
        end
    end
    isone(pc) && throw(ArgumentError("center_manifold: no eigenvalue on the imaginary axis; the linearization decides"))
    Vc = _kernel(_matrix_polynomial(pc, A, s))
    Vs = isone(ps) ? Vector{GiacExpr}[] : _kernel(_matrix_polynomial(ps, A, s))
    length(Vc) + length(Vs) == n || throw(ArgumentError("center_manifold: the center and stable directions do not span the state space"))
    # already in block form, up to the order of the variables: keep them
    unit(v) = count(!iszero, v) == 1 && isone(v[findfirst(!iszero, v)])
    all(unit, Vc) && all(unit, Vs) && return center_manifold(F, X; center = [X[findfirst(!iszero, v)] for v in Vc], order)
    T = GiacExpr[v[i] for i in 1:n, v in vcat(Vc, Vs)]
    Tinv = Giac._giac_inv(T)
    ξ = [Giac.giac_identifier("ξ" * _subscript(k)) for k in eachindex(Vc)]
    η = [Giac.giac_identifier("η" * _subscript(k)) for k in eachindex(Vs)]
    Z = vcat(ξ, η)
    xs = [Commands.expand(sum(T[i, j] * Z[j] for j in 1:n)) for i in 1:n]
    G = [Commands.expand(e) for e in Tinv * [substitute(Fi, (X .=> xs)...) for Fi in F]]
    m = center_manifold(G, Z; center = ξ, order)
    return CenterManifold(m.center, m.h, m.reduced, m.order, m.stability, X .=> xs)
end

_subscript(k) = join(Char(0x2080 + d) for d in reverse(digits(k)))

# p(A) for a polynomial p in s
function _matrix_polynomial(p, A, s)
    n = size(A, 1)
    d = Int(Giac.to_julia(Commands.invoke_cmd(:degree, p, s)))
    E = GiacExpr[i == j ? 1 : 0 for i in 1:n, j in 1:n]
    P = zero(E)
    for k in 0:d
        P += Commands.invoke_cmd(:coeff, p, s, k) * E
        E = E * A
    end
    return P
end

# a basis of the kernel of a square matrix, exactly, each vector's first nonzero entry positive
function _kernel(M)
    all(iszero, M) && return [GiacExpr[i == j ? 1 : 0 for i in 1:size(M, 1)] for j in 1:size(M, 2)]
    basis = [collect(GiacExpr, collect(v)) for v in collect(Commands.invoke_cmd(:ker, GiacMatrix(M)))]
    return [Giac.to_julia(v[findfirst(!iszero, v)]) < 0 ? -v : v for v in basis]
end

# whether every root of q lies on the imaginary axis: q(-s) = ±q(s), and numerically no real part
function _roots_on_axis(q, s)
    qm = substitute(q, s => -s)
    (iszero(_simplified(qm - q)) || iszero(_simplified(qm + q))) || return false
    d = Int(Giac.to_julia(Commands.invoke_cmd(:degree, q, s)))
    c = [Float64(Giac.to_julia(Commands.invoke_cmd(:coeff, q, s, k))) for k in 0:d]
    d == 1 && return iszero(c[1])
    companion = zeros(d, d)
    companion[2:end, 1:(end - 1)] .= Giac.LinearAlgebra.I(d - 1)
    companion[:, end] .= -c[1:d] ./ c[end]
    return all(r -> abs(real(r)) <= 1.0e-9 * max(1, abs(r)), Giac.LinearAlgebra.eigvals(companion))
end

# ẋ = a xᵐ + …: the lowest term that is not zero decides, asymptotically stable for odd m and
# a < 0, unstable for a > 0 or even m
function _leading_stability(r, x, order)
    for m in 1:order
        s = sign_of(Commands.invoke_cmd(:coeff, r, x, m))
        s === :zero && continue
        s in (:positive, :negative) || return :unknown
        return isodd(m) && s === :negative ? :stable : :unstable
    end
    return :unknown
end

end # module
