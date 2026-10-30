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
