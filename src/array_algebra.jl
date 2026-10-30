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

# `I` is `UniformScaling{Bool}`; a `Bool` converts to a Giac boolean, not to 0/1, so the scale
# is made numeric before it is added.
_numeric(λ::Bool) = Int(λ)
_numeric(λ) = λ

function _add_scaled_identity(A::AbstractMatrix{GiacExpr}, λ)
    n = LinearAlgebra.checksquare(A)
    B = Matrix{GiacExpr}(A)
    for i in 1:n
        B[i, i] += λ
    end
    return B
end

Base.:+(A::AbstractMatrix{GiacExpr}, J::UniformScaling) = _add_scaled_identity(A, _numeric(J.λ))
Base.:+(J::UniformScaling, A::AbstractMatrix{GiacExpr}) = A + J
Base.:-(A::AbstractMatrix{GiacExpr}, J::UniformScaling) = _add_scaled_identity(A, -_numeric(J.λ))
Base.:-(J::UniformScaling, A::AbstractMatrix{GiacExpr}) = _add_scaled_identity(-A, _numeric(J.λ))

# Scalars times arrays. `GiacExpr` is not a `Number`, so Julia's own methods don't apply.
Base.:*(x::GiacExpr, A::AbstractArray) = map(a -> x * a, A)
Base.:*(A::AbstractArray, x::GiacExpr) = map(a -> a * x, A)
Base.:/(A::AbstractArray, x::GiacExpr) = map(a -> a / x, A)

"""
    s * I

A symbolic multiple of the identity, of whatever size the matrix it meets: `s*I - A`.
(`UniformScaling` only holds a `Number`, which `GiacExpr` is not.)
"""
struct SymbolicScaling
    λ::GiacExpr
end

Base.:*(x::GiacExpr, J::UniformScaling) = SymbolicScaling(x * _numeric(J.λ))
Base.:*(J::UniformScaling, x::GiacExpr) = SymbolicScaling(_numeric(J.λ) * x)
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
    A ~ _add_scaled_identity(fill(zero(GiacExpr), size(A)), _numeric(J.λ))

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
function Commands.solve(eqs::AbstractArray{GiacExpr}, X::AbstractMatrix{GiacExpr})
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

# ---------------------------------------------------------------------------------------------
# Typesetting
# ---------------------------------------------------------------------------------------------

"LaTeX for one expression, as Giac typesets it (without display delimiters), tidied."
_entry_latex(e) = tidy_latex(strip(sprint(show, MIME("text/latex"), convert(GiacExpr, e)), ['$', '\n', ' ']))

const _GREEK_TEX = Dict(
    'α' => "\\alpha", 'β' => "\\beta", 'γ' => "\\gamma", 'δ' => "\\delta", 'ε' => "\\varepsilon",
    'ζ' => "\\zeta", 'η' => "\\eta", 'θ' => "\\theta", 'ι' => "\\iota", 'κ' => "\\kappa",
    'λ' => "\\lambda", 'μ' => "\\mu", 'ν' => "\\nu", 'ξ' => "\\xi", 'ρ' => "\\rho", 'σ' => "\\sigma",
    'τ' => "\\tau", 'υ' => "\\upsilon", 'φ' => "\\varphi", 'χ' => "\\chi", 'ψ' => "\\psi",
    'ω' => "\\omega", 'Γ' => "\\Gamma", 'Θ' => "\\Theta", 'Λ' => "\\Lambda", 'Ξ' => "\\Xi",
    'Π' => "\\Pi", 'Υ' => "\\Upsilon", 'Φ' => "\\Phi", 'Ψ' => "\\Psi", 'Ω' => "\\Omega")

const _MARK_TEX = Dict('★' => "^{\\star}", '′' => "'", '″' => "''", '̂' => "", '̃' => "")

"""
    tidy_latex(s) -> String

Clean up Giac's LaTeX for KaTeX/MathJax: Unicode Greek letters become TeX commands (Giac emits
`\\mathrm{τ}`), `★` and primes in names become superscripts, and a few spellings are normalised
(`\\frac1{b}` → `\\frac{1}{b}`, `\\mathrm{det}` → `\\det`, `\\mathrm{trace}` → `\\operatorname{tr}`).
"""
function tidy_latex(s::AbstractString)
    # \mathrm{…} around a name containing Unicode letters or marks: typeset the parts directly
    s = replace(s, r"\\mathrm\{([^{}]*[^\x00-\x7F][^{}]*)\}" => m -> _tex_name(chop(m; head = 8, tail = 1)))
    s = replace(s, r"[α-ωΓ-Ω★′″]" => m -> _tex_char(only(m)))
    s = replace(s, r"\\frac(\d)\{" => s"\\frac{\1}{")
    s = replace(s, "\\mathrm{det}" => "\\det", "\\mathrm{trace}" => "\\operatorname{tr}")
    return s
end

_tex_char(c::Char) = haskey(_GREEK_TEX, c) ? _GREEK_TEX[c] * " " :
                     haskey(_MARK_TEX, c) ? _MARK_TEX[c] : string(c)

function _tex_name(name::AbstractString)
    # Greek letters → commands, marks → superscripts, digit runs → subscripts (as Giac does for
    # `x1`), multi-letter ASCII runs → upright text
    parts = String[]
    for run in eachmatch(r"[A-Za-z]+|[0-9]+|.", name)
        r = run.match
        c = first(r)
        push!(parts, haskey(_GREEK_TEX, c) ? _GREEK_TEX[c] :
                     haskey(_MARK_TEX, c) ? _MARK_TEX[c] :
                     isdigit(c) ? "_{$r}" :
                     length(r) > 1 ? "\\mathrm{$r}" : r)
    end
    return join(parts)
end

"""
    giac_latex(x) -> String

LaTeX for an expression or an array of expressions (a bracketed matrix; a vector is a column,
an adjoint vector a row).
"""
giac_latex(e) = _entry_latex(e)
function giac_latex(A::AbstractVecOrMat)
    M = A isa AbstractVector ? reshape(A, :, 1) : A
    rows = [join((_entry_latex(M[i, j]) for j in axes(M, 2)), " & ") for i in axes(M, 1)]
    return "\\begin{bmatrix}" * join(rows, " \\\\ ") * "\\end{bmatrix}"
end

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
