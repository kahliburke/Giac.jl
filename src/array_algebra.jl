# Symbolic linear algebra on ordinary Julia arrays of `GiacExpr`.
#
# A `Matrix{GiacExpr}` already gets products, `adjoint`, slicing, `tr` and broadcasting from
# Julia's generic code, with Giac doing the scalar algebra. The methods here fill the gaps where
# the generic code assumes a numeric element type: adding a `UniformScaling` (`A + I`), and
# `det`/`inv`, whose generic LU factorisation branches on comparisons of entries. They hand off
# to Giac through `GiacMatrix`.

"""
    Matrix(M::GiacMatrix) -> Matrix{GiacExpr}

The entries of a Giac matrix as a Julia matrix, which supports slicing, broadcasting and
Julia's generic linear algebra.
"""
Base.Matrix(M::GiacMatrix) = GiacExpr[M[i, j] for i in 1:size(M, 1), j in 1:size(M, 2)]
Base.Matrix{GiacExpr}(M::GiacMatrix) = Matrix(M)

function _add_scaled_identity(A::AbstractMatrix{GiacExpr}, λ)
    n = LinearAlgebra.checksquare(A)
    B = Matrix{GiacExpr}(A)
    for i in 1:n
        B[i, i] += λ
    end
    return B
end

Base.:+(A::AbstractMatrix{GiacExpr}, J::UniformScaling) = _add_scaled_identity(A, J.λ)
Base.:+(J::UniformScaling, A::AbstractMatrix{GiacExpr}) = A + J
Base.:-(A::AbstractMatrix{GiacExpr}, J::UniformScaling) = _add_scaled_identity(A, -J.λ)
Base.:-(J::UniformScaling, A::AbstractMatrix{GiacExpr}) = _add_scaled_identity(-A, J.λ)

# Scalars times arrays. `GiacExpr` is not a `Number`, so Julia's own methods don't apply.
Base.:*(x::GiacExpr, A::AbstractArray) = map(a -> x * a, A)
Base.:*(A::AbstractArray, x::GiacExpr) = map(a -> a * x, A)
Base.:/(A::AbstractArray, x::GiacExpr) = map(a -> a / x, A)

# In broadcasting an expression is a scalar: `t .* A` scales every entry by `t`.
Base.broadcastable(x::GiacExpr) = Ref(x)

"""
    mul!(C, A, B, α::GiacExpr, β::GiacExpr)

`C = α A B + β C` in place, for symbolic scale factors. LinearAlgebra's methods take `Number`s,
which `GiacExpr` is not, and array packages built on the 5-argument form (BlockArrays, through
ArrayLayouts) pass the element type's `one` and `zero`.
"""
function LinearAlgebra.mul!(C::AbstractVecOrMat{GiacExpr}, A::AbstractVecOrMat, B::AbstractVecOrMat,
                            α::GiacExpr, β::GiacExpr)
    AB = A * B
    # β is usually zero, and C may be uninitialized then
    if isequal(β, zero(GiacExpr))
        C .= isequal(α, one(GiacExpr)) ? AB : α .* AB
    else
        C .= α .* AB .+ β .* C
    end
    return C
end

"""
    s * I

A symbolic multiple of the identity, of whatever size the matrix it meets: `s*I - A`.
(`UniformScaling` only holds a `Number`, which `GiacExpr` is not.)
"""
struct SymbolicScaling
    λ::GiacExpr
end

Base.:*(x::GiacExpr, J::UniformScaling) = SymbolicScaling(x * J.λ)
Base.:*(J::UniformScaling, x::GiacExpr) = SymbolicScaling(J.λ * x)
Base.:-(S::SymbolicScaling) = SymbolicScaling(-S.λ)
Base.:+(A::AbstractMatrix, S::SymbolicScaling) = _add_scaled_identity(Matrix{GiacExpr}(A), S.λ)
Base.:+(S::SymbolicScaling, A::AbstractMatrix) = A + S
Base.:-(A::AbstractMatrix, S::SymbolicScaling) = _add_scaled_identity(Matrix{GiacExpr}(A), -S.λ)
Base.:-(S::SymbolicScaling, A::AbstractMatrix) = _add_scaled_identity(-Matrix{GiacExpr}(A), S.λ)
Base.:*(S::SymbolicScaling, A::AbstractArray) = S.λ * A
Base.:*(A::AbstractArray, S::SymbolicScaling) = A * S.λ
Base.:~(A::AbstractMatrix{GiacExpr}, S::SymbolicScaling) = A ~ _add_scaled_identity(fill(zero(GiacExpr), size(A)), S.λ)

# ---------------------------------------------------------------------------------------------
# Matrix equations: `A ~ B` entry by entry, and `solve` for the free entries of a matrix unknown
# ---------------------------------------------------------------------------------------------

"""
    A ~ B

Entry-by-entry equations between arrays of expressions, with the same shape. `B` may be
another array or a `UniformScaling` such as `-I`.
"""
function Base.:~(A::AbstractArray{GiacExpr}, B::AbstractArray)
    size(A) == size(B) || throw(DimensionMismatch("equation sides have sizes $(size(A)) and $(size(B))"))
    return map(~, A, B)
end
Base.:~(A::AbstractMatrix{GiacExpr}, J::UniformScaling) =
    A ~ _add_scaled_identity(fill(zero(GiacExpr), size(A)), J.λ)
# every entry equal to the same value: `f(x) ~ 0` for equilibria
Base.:~(A::AbstractArray{GiacExpr}, b::Union{Number, GiacExpr}) = map(a -> a ~ b, A)

"""
    solve(eqs::AbstractArray{GiacExpr}, X::AbstractMatrix{GiacExpr}) -> Matrix{GiacExpr}

Solve the equations `eqs` (e.g. `P*A + A'*P ~ -I`) for the free entries of the matrix unknown
`X`, returning `X` with its entries replaced by the solution. Repeated equations (from a
symmetric unknown, say) count once. Throws if there is no solution or more than one.

```julia
@giac_matrix A[2, 2] P[2, 2] symmetric
P★ = solve(P*A + A'*P ~ -I, P)
```
"""
Commands.solve(eqs::AbstractArray{GiacExpr}, X::AbstractMatrix{GiacExpr}) = _solve_for_matrix(eqs, X)
# the generated `solve(::GiacInput, …)` and `solve(::GiacMatrix, …)` also accept these arguments
Commands.solve(eqs::AbstractVector{GiacExpr}, X::AbstractMatrix{GiacExpr}) = _solve_for_matrix(eqs, X)
Commands.solve(eqs::GiacMatrix, X::AbstractMatrix{GiacExpr}) = _solve_for_matrix(eqs, X)

function _solve_for_matrix(eqs, X)
    unknowns = unique(string, filter(is_identifier, vec(X)))
    # the same equation can come out with its terms in a different order: compare normal forms
    equations = unique(e -> string(Commands.normal(e)), vec(eqs))
    solutions = Commands.solve(equations, unknowns)
    n = length(solutions)
    n == 1 || throw(ArgumentError(n == 0 ? "the equations have no solution for $(join(string.(unknowns), ", "))" :
                                           "the equations have $n solutions; solve them as a vector to see all of them"))
    values = Dict(string(u) => solutions[1][k] for (k, u) in enumerate(unknowns))
    return map(x -> get(values, string(x), x), X)
end

# ---------------------------------------------------------------------------------------------
# Vector calculus
# ---------------------------------------------------------------------------------------------

"""
    ∇(f, x::AbstractVector) -> Vector{GiacExpr}

Gradient of the scalar `f` with respect to the variables `x`: `[∂f/∂x₁, …, ∂f/∂xₙ]`.
"""
∇(f, x::AbstractVector) = GiacExpr[Commands.diff(f, xi) for xi in x]

"""
    jacobian(f::AbstractVector, x::AbstractVector) -> Matrix{GiacExpr}

Jacobian `J[i, j] = ∂fᵢ/∂xⱼ`.
"""
jacobian(f::AbstractVector, x::AbstractVector) = GiacExpr[Commands.diff(fi, xj) for fi in f, xj in x]

"""
    hessian(f, x::AbstractVector) -> Matrix{GiacExpr}

Hessian `H[i, j] = ∂²f/∂xᵢ∂xⱼ` of the scalar `f`.
"""
hessian(f, x::AbstractVector) = jacobian(∇(f, x), x)

"""
    taylor(f, x => a, n)
    taylor(f, [x1, x2] => (a1, a2), n)
    taylor(f, x, n)

The Taylor polynomial of `f` about `x = a` (about 0 when no point is given): every term up to
total degree `n` in the variables. `x` may be a vector of variables, with the point a vector or
a tuple (a tuple keeps exact constants, `x => (π, 0)`) or one value for all. For an array `f`,
entry by entry.

```julia
@giac_var x[1:2] u
f = [x2, -sin(x1) + u * cos(x1)]
taylor(f, [x; u] => (π, 0, 0), 1)     # [x2, x1 - π - u]: linearized about the upright position
```
"""
function taylor(f::GiacExpr, point::Pair, n::Integer)
    x, a = point
    vars = x isa GiacExpr ? GiacExpr[x] : collect(GiacExpr, x)
    at = a isa Union{AbstractArray, Tuple} ? collect(Any, a) : fill(a, length(vars))
    length(at) == length(vars) ||
        throw(DimensionMismatch("expanding about a point of $(length(at)) values in $(length(vars)) variables"))
    return invoke_cmd(:series, f, vars, at, n)
end
taylor(f::GiacExpr, x::Union{GiacExpr, AbstractVector{GiacExpr}}, n::Integer) = taylor(f, x => 0, n)
taylor(f::AbstractArray{GiacExpr}, args...) = map(e -> taylor(e, args...), f)

# ---------------------------------------------------------------------------------------------
# Typesetting
# ---------------------------------------------------------------------------------------------

"Giac's own LaTeX for an expression, without delimiters (its plain form if Giac has none)."
function _giac_tex(e::GiacExpr)
    s = string(Commands.invoke_cmd(:latex, e))         # a quoted string
    return length(s) > 2 ? strip(chop(s; head = 1, tail = 1)) : string(e)
end

"""
LaTeX for one expression (without display delimiters), written as a textbook would (see
[`typeset!`](@ref)); Giac's own LaTeX, tidied, if that fails.
"""
function _entry_latex(e)
    g = convert(GiacExpr, e)
    _is_list(g) && return _list_latex(g)
    return try
        _present_latex(g)
    catch
        tidy_latex(_giac_tex(g))
    end
end

# Giac's `list[…]` (vector subtype `_LIST__VECT`), as `solve` returns its solutions. Read from
# the printed form: display runs this for every entry, and the type queries re-evaluate each one.
_is_list(g::GiacExpr) = startswith(string(g), "list[")

# A plain number as Giac prints one: an integer, a fraction, or a decimal, possibly negative.
_is_plain_number(g) = occursin(r"^-?\d+(\.\d+)?(e-?\d+)?(/\d+)?$", string(g))

"A list of solutions as a set; solutions that are points are column vectors."
function _list_latex(g::GiacExpr)
    items = collect(GiacExpr, g)
    show_item(x) = giac_type(x) == VECT ? giac_latex(collect(GiacExpr, x)) : _entry_latex(x)
    return "\\left\\{" * join(map(show_item, items), ",\\; ") * "\\right\\}"
end

"""
    tidy_latex(s) -> String

Clean up Giac's LaTeX for KaTeX/MathJax: Unicode Greek letters become TeX commands (Giac emits
`\\mathrm{τ}`), `★` and primes in names become superscripts, a few spellings are normalised
(`\\frac1{b}` → `\\frac{1}{b}`, `\\mathrm{det}` → `\\det`, `\\mathrm{trace}` → `\\operatorname{tr}`),
and a series remainder `t^{4} \\mathrm{order\\_size}(t)` becomes `O(t^{4})`.
"""
function tidy_latex(s::AbstractString)
    # \mathrm{…} around a name containing Unicode letters or marks: typeset the parts directly
    s = replace(s, r"\\mathrm\{([^{}]*[^\x00-\x7F][^{}]*)\}" => m -> _tex_name(chop(m; head = 8, tail = 1)))
    # the same for names Giac leaves bare (x̄, A₀, kₚ, Kᵀ, θ)
    s = replace(s, _BARE_NAME => _bare_name_tex)
    s = replace(s, r"\\frac(\d)\{" => s"\\frac{\1}{")
    s = replace(s, "\\mathrm{det}" => "\\det", "\\mathrm{trace}" => "\\operatorname{tr}")
    # a series' remainder `t^{4} order_size(t)` is O(t⁴)
    s = replace(s, _ORDER_TERM => s"O\\left(\1\\right)")
    # the trace or determinant of symbolic matrices, written as such (see matrix_symbols.jl)
    return _atom_tex(s)
end

# a name outside a TeX command; those beyond ASCII letters and digits are rewritten
const _BARE_NAME = r"(?<![\\A-Za-z])[\p{L}∂∇][\p{L}\p{M}\p{N}′″★ᵀ⁺⁻]*"

function _bare_name_tex(m::AbstractString)
    isascii(m) && return m
    t = _tex_name(m)
    return occursin(r"\\[a-zA-Z]+$", t) ? t * " " : t
end

const _ORDER_TERM = r"(\\left\([^()]*\\right\)\^\{\d+\}|[A-Za-z]\w*(?:_\{\d+\})?(?:\^\{\d+\})?) \\mathrm\{order\\_size\}\\left\([^()]*\\right\)"

"""
    giac_latex(x) -> String

LaTeX for an expression or an array of expressions (a bracketed matrix; a vector is a column,
an adjoint vector a row).
"""
giac_latex(e) = _entry_latex(e)
giac_latex(A::AbstractVecOrMat) = _matrix_latex(A)

"""
    _matrix_latex(A; rowbreaks = (), colbreaks = ())

A bracketed matrix. `rowbreaks` and `colbreaks` hold the rows and columns after which a partition
line is drawn (a block matrix's block boundaries).
"""
function _matrix_latex(A::AbstractVecOrMat; rowbreaks = (), colbreaks = ())
    M = A isa AbstractVector ? reshape(A, :, 1) : A
    nr, nc = size(M)
    entries = [_entry_latex(M[i, j]) for i in 1:nr, j in 1:nc]
    # matrix cells are set at text size, which shrinks fractions; set them at display size, and
    # pad each row with an invisible strut so the rows separate and the brackets, which TeX
    # draws slightly short of their contents, still enclose them
    tall = any(e -> occursin("\\frac", e), entries)
    tall && (entries = replace.(entries, "\\frac" => "\\dfrac"))
    # a square root over powers raises its row too, though less than a display fraction
    pad = tall ? _ROW_STRUT : any(e -> occursin("\\sqrt", e), entries) ? _ROOT_STRUT : ""
    rows = [((i - 1) in rowbreaks ? "\\hline " : "") * pad * join(entries[i, :], " & ") for i in 1:nr]
    # a column of plain numbers is right-aligned, so digits line up by place value and signs hang
    # to the left; a column with anything symbolic, or a fraction, stays centred
    numeric(j) = all(i -> _is_plain_number(M[i, j]) && !occursin("\\dfrac", entries[i, j]), 1:nr)
    align = [numeric(j) ? 'r' : 'c' for j in 1:nc]
    if all(==('c'), align) && isempty(rowbreaks) && isempty(colbreaks)
        return "\\begin{bmatrix}" * join(rows, " \\\\ ") * "\\end{bmatrix}"
    end
    spec = join(string(align[j], j in colbreaks ? "|" : "") for j in 1:nc)
    # an `array` pads its outer edges by the column separation, which `bmatrix` drops
    return "\\left[\\hspace{-5pt}\\begin{array}{$spec}" * join(rows, " \\\\ ") *
           "\\end{array}\\hspace{-5pt}\\right]"
end

# the height and depth of a display fraction's row, so a row without one is spaced the same
const _ROW_STRUT = "\\rule[-0.6em]{0pt}{1.9em}"
const _ROOT_STRUT = "\\rule[-0.7em]{0pt}{2.2em}"

Base.show(io::IO, ::MIME"text/latex", A::AbstractVecOrMat{GiacExpr}) = print(io, "\$\$", giac_latex(A), "\$\$")

LinearAlgebra.det(A::AbstractMatrix{GiacExpr}) = det(GiacMatrix(A))

# LinearAlgebra's dense `inv(::StridedMatrix)` outranks an `AbstractMatrix{GiacExpr}` method for
# a plain `Matrix`, so both are defined.
_giac_inv(A) = Matrix(inv(GiacMatrix(A)))
Base.inv(A::AbstractMatrix{GiacExpr}) = _giac_inv(A)
Base.inv(A::StridedMatrix{GiacExpr}) = _giac_inv(A)

"""
    A \\ b

Solve `A x = b` for symbolic `A` (via Giac's inverse, entries simplified).
"""
Base.:\(A::AbstractMatrix{GiacExpr}, b::AbstractVecOrMat) = Commands.simplify.(_giac_inv(A) * b)
Base.:\(A::StridedMatrix{GiacExpr}, b::AbstractVecOrMat) = Commands.simplify.(_giac_inv(A) * b)
# ---------------------------------------------------------------------------------------------
# Spectra, rank, null space and the matrix exponential, computed by Giac
# ---------------------------------------------------------------------------------------------

"A Giac matrix value (a list of rows) as a `Matrix{GiacExpr}`."
_rows_to_matrix(M::GiacExpr) = permutedims(reduce(hcat, [collect(GiacExpr, row) for row in M]))

"Eigenvalues `D` and eigenvector matrix `V` (as columns) from Giac's Jordan decomposition."
function _jordan(A::AbstractMatrix{GiacExpr})
    LinearAlgebra.checksquare(A)
    PJ = Commands.jordan(GiacMatrix(A))
    V, J = _rows_to_matrix(PJ[1]), _rows_to_matrix(PJ[2])
    n = size(J, 1)
    diagonal = all(iszero(J[i, j]) for i in 1:n, j in 1:n if i != j)
    return V, J, diagonal
end

"""
    eigvals(A::AbstractMatrix{GiacExpr}) -> Vector{GiacExpr}

Eigenvalues of a symbolic matrix, with repeats for repeated eigenvalues.
"""
LinearAlgebra.eigvals(A::AbstractMatrix{GiacExpr}) = (J = _jordan(A)[2]; [J[i, i] for i in axes(J, 1)])

"""
    eigen(A::AbstractMatrix{GiacExpr}) -> Eigen

Eigenvalues and eigenvectors (columns of `vectors`, matching the order of `values`). Throws if
`A` is not diagonalizable; `Giac.Commands.jordan` gives the Jordan form instead.
"""
function LinearAlgebra.eigen(A::AbstractMatrix{GiacExpr})
    V, J, diagonal = _jordan(A)
    diagonal || throw(ArgumentError("the matrix is not diagonalizable; use Giac.Commands.jordan for its Jordan form"))
    return Eigen([J[i, i] for i in axes(J, 1)], V)
end

LinearAlgebra.eigvecs(A::AbstractMatrix{GiacExpr}) = eigen(A).vectors

# Λ = diag(λ) beside V, eigenvectors as columns: the factors of A = VΛV⁻¹
function Base.show(io::IO, ::MIME"text/latex", E::Eigen{GiacExpr})
    n = length(E.values)
    Λ = GiacExpr[i == j ? E.values[i] : zero(GiacExpr) for i in 1:n, j in 1:n]
    print(io, "\$\$\\Lambda = ", giac_latex(Λ), ", \\quad V = ", giac_latex(E.vectors), "\$\$")
end

LinearAlgebra.rank(A::AbstractMatrix{GiacExpr}) = to_julia(Commands.invoke_cmd(:rank, GiacMatrix(A)))

"""
    nullspace(A::AbstractMatrix{GiacExpr}) -> Matrix{GiacExpr}

A basis of the null space of `A`, as columns (no columns if `A` has full column rank).
"""
function LinearAlgebra.nullspace(A::AbstractMatrix{GiacExpr})
    K = Commands.invoke_cmd(:ker, GiacMatrix(A))
    length(K) == 0 && return Matrix{GiacExpr}(undef, size(A, 2), 0)
    return permutedims(_rows_to_matrix(K))
end

"""
    exp(A::AbstractMatrix{GiacExpr}) -> Matrix{GiacExpr}

The matrix exponential, computed symbolically: `exp(A*t)` is the state-transition matrix of
`ẋ = A x`. Entries come as sums of exponentials (`2e^{-t} - e^{-2t}`); `simplify` would put them
over a common denominator.
"""
_giac_exp(A) = _rows_to_matrix(Commands.invoke_cmd(:exp, GiacMatrix(A)))
Base.exp(A::AbstractMatrix{GiacExpr}) = _giac_exp(A)
Base.exp(A::StridedMatrix{GiacExpr}) = _giac_exp(A)
