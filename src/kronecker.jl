# Kronecker products and vectorization: vec(UXV) = (Vᵀ ⊗ U) vec X turns a linear matrix
# equation into a linear system, as AᵀP + PA = -Q becomes (I⊗Aᵀ + Aᵀ⊗I) vec P = -vec Q.

"""
    kron(A::MatExpr, B::MatExpr)

The Kronecker product `A ⊗ B`, of size `(m·p) × (n·q)` for `A` `m × n` and `B` `p × q` (sizes
given by names multiply as names: `n*n`). It is bilinear, `I ⊗ I` is the identity, and the rules
hold as expressions are built: `(A⊗B)(C⊗D) = AC⊗BD`, `(A⊗B)ᵀ = Aᵀ⊗Bᵀ`, `(A⊗B)⁻¹ = A⁻¹⊗B⁻¹`,
`tr(A⊗B) = tr A tr B`, `det(A⊗B) = (det A)ᵐ (det B)ⁿ` for `A` `n × n` and `B` `m × m`.
`substitute` gives the explicit Kronecker product.
"""
function Base.kron(A::MatExpr, B::MatExpr)
    rows, cols = _dim_mul(A.rows, B.rows), _dim_mul(A.cols, B.cols)
    terms = Pair{Word, GiacExpr}[]
    for (wa, ca) in A.terms, (wb, cb) in B.terms
        unit(w, M) = MatExpr(isempty(w) ? M.rows : _rows(first(w)), isempty(w) ? M.cols : _cols(last(w)), [w => one(GiacExpr)])
        sign, f = _kron_factor(unit(wa, A), unit(wb, B))
        push!(terms, (f === nothing ? Factor[] : Factor[f]) => (sign == 1 ? ca * cb : -(ca * cb)))
    end
    return _matexpr(rows, cols, terms)
end
Base.kron(A::MatExpr, J::UniformScaling) = kron(A, _identity_like(J, A.cols))
Base.kron(J::UniformScaling, A::MatExpr) = kron(_identity_like(J, A.rows), A)
_identity_like(J::UniformScaling, n::Dim) = _coefficient(J.λ) * identity_matrix(n)

"""
    vec(X::MatExpr)

The columns of the matrix symbol `X` stacked into one column, written `vec X`: a vector symbol of
size `(m·n) × 1`. Used with [`vectorize`](@ref).
"""
function Base.vec(X::MatExpr)
    s = _matrix_symbol(X)
    s.cols == 1 && return X
    name = Symbol("vec", s.name)
    _TEX_NAMES[string(name)] = "\\operatorname{vec} " * _tex_name(string(s.name))
    return MatExpr(MatrixSymbol(name, _dim_mul(s.rows, s.cols), 1, Symbol[]))
end

"""
    vectorize(E, X) -> (K, R)

An expression `E` linear in the matrix symbol `X`, as the linear system its entries make:
`vec E = K vec X + vec R`, with `K` from `vec(UXV) = (Vᵀ ⊗ U) vec X` and `R` the terms without `X`.

```julia
@matsym A[n, n] P[n, n]::symmetric
K, R = vectorize(A'*P + P*A, P)       # K = I⊗Aᵀ + Aᵀ⊗I, R = 0
substitute(K, A => [0 1; -2 -3])      # the 4×4 matrix, to solve K vec P = -vec Q
```

`X` must appear at most once in each term, and not transposed (unless it is symmetric or skew,
so that its transpose is itself or its negative).
"""
function vectorize(E::MatExpr, X::MatExpr)
    s = _matrix_symbol(X)
    K = zero_matrix(_dim_mul(E.rows, E.cols), _dim_mul(s.rows, s.cols))
    R = zero_matrix(E.rows, E.cols)
    for (w, c) in E.terms
        at = findall(f -> f.base isa MatrixSymbol && f.base == s, w)
        if isempty(at)
            R = R + c * _word_expr(w, E.rows)
            continue
        end
        (length(at) == 1 && w[only(at)].power == 1) ||
            throw(ArgumentError("$(_plain(E, true)) is not linear in $(s.name)"))
        i = only(at)
        w[i].transposed && throw(ArgumentError("vectorize: $(s.name)ᵀ appears in $(_plain(E, true)); only $(s.name) itself is handled"))
        U = _word_expr(w[1:(i - 1)], s.rows)
        V = _word_expr(w[(i + 1):end], s.cols)
        K = K + c * kron(transpose(V), U)
    end
    return K, R
end
