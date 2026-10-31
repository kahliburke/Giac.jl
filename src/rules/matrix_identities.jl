# Identities of symbolic matrices that their canonical form does not apply by itself: the
# Woodbury and Sherman–Morrison formulas, the push-through identity, Sylvester's determinant
# identity, the matrix determinant lemma, Cayley–Hamilton for 2×2 matrices. Each is a rewrite of
# matrix expressions (or of determinants inside scalars), checked on random numeric matrices.

using ..Giac: MatExpr, MatrixSymbol, Factor, _matexpr, _rows, _cols, _has, _factor_key, _word_rank,
              identity_matrix, _ATOMS, @matsym
using LinearAlgebra: LinearAlgebra, I, det

"""
    MatrixIdentity

An identity of symbolic matrices: `id(M)` rewrites `M` (a `MatExpr`, or a scalar with
determinants of matrices in it) at the first place the identity applies, or returns `nothing`.
Listed by [`matrix_identities`](@ref).
"""
struct MatrixIdentity
    name::String
    tex::String
    apply::Function       # a term's word and coefficient → the expression replacing that term, or nothing
    scalar::Bool          # rewrites a determinant (inside a scalar) rather than a matrix expression
    example::Function     # () -> an expression the identity applies to, for checking it
end

Base.show(io::IO, id::MatrixIdentity) = print(io, "MatrixIdentity(\"", id.name, "\")")
Base.show(io::IO, ::MIME"text/latex", id::MatrixIdentity) = print(io, "\$\$", id.tex, "\$\$")
Base.show(io::IO, ::MIME"text/latex", ids::AbstractVector{MatrixIdentity}) =
    print(io, "\$\$\\begin{array}{ll}", join(("\\text{$(i.name)} & " * i.tex for i in ids), " \\\\ "), "\\end{array}\$\$")

# A word as an expression (`rows × cols` for the empty word, the identity).
_word_expr(w, rows, cols) = _matexpr(isempty(w) ? rows : _rows(first(w)), isempty(w) ? cols : _cols(last(w)), [w => one(GiacExpr)])

# Apply the identity at the first term it rewrites.
function (id::MatrixIdentity)(M::MatExpr)
    id.scalar && return _apply_to_coefficients(id, M)
    for (i, (w, c)) in enumerate(M.terms)
        new = id.apply(w, M.rows, M.cols)
        new === nothing && continue
        rest = [M.terms[j] for j in eachindex(M.terms) if j != i]
        return _matexpr(M.rows, M.cols, rest) + c * new
    end
    return nothing
end

# A determinant identity rewrites a determinant (a scalar atom) wherever it appears.
function (id::MatrixIdentity)(e::GiacExpr)
    id.scalar || return nothing
    for v in Giac.free_symbols(e)
        name = string(v)
        haskey(_ATOMS, name) || continue
        kind, payload = _ATOMS[name]
        kind === :det && payload isa MatExpr || continue
        new = id.apply(payload)
        new === nothing && continue
        return Giac.Commands.normal(substitute(e, v => new))
    end
    return nothing
end

function _apply_to_coefficients(id::MatrixIdentity, M::MatExpr)
    for (i, (w, c)) in enumerate(M.terms)
        new = id(c)
        new === nothing && continue
        return Giac.map_coefficients(k -> k === c ? new : k, M)
    end
    return nothing
end

# the opaque inverses `(sum)⁻¹` in a word, with their positions
_inverse_sums(w) = [(k, f.base) for (k, f) in enumerate(w) if f.base isa MatExpr && f.power == -1 && !f.transposed]

# a sum `aI + cW` (or `cW + aI`): `(a, c, W)`, or `nothing`
function _identity_plus(M::MatExpr)
    length(M.terms) == 2 || return nothing
    (w1, c1), (w2, c2) = M.terms
    isempty(w1) && !isempty(w2) && return (c1, c2, w2)
    isempty(w2) && !isempty(w1) && return (c2, c1, w1)
    return nothing
end

# replace the factor at `k` of word `w` by the expression `R`
_splice(w, k, R, rows, cols) = _word_expr(w[1:(k - 1)], rows, R.rows) * R * _word_expr(w[(k + 1):end], R.cols, cols)

# a column vector (`u`), or one transposed (`vᵀ`)
_is_vector(f::Factor) = f.base isa MatrixSymbol && f.power == 1 && f.base.cols == 1

# 1×1 expression as the scalar it is (its trace)
_scalar(M::MatExpr) = LinearAlgebra.tr(M)

const _MATRIX_IDENTITIES = Ref{Any}(nothing)

"""
    matrix_identities()

The identities of symbolic matrices beyond what their canonical form applies: Woodbury and
Sherman–Morrison, the inverse of a sum of inverses, push-through, commuting with a function of
itself, Sylvester's determinant identity, the matrix determinant lemma, and Cayley–Hamilton for
2×2 matrices. Each is checked on random numeric matrices in the test suite ([`verify`](@ref)).
`simplest(M)` searches them; `rewrite(M, ids...)` applies them until none does.
"""
function matrix_identities()
    _MATRIX_IDENTITIES[] === nothing || return _MATRIX_IDENTITIES[]
    ids = MatrixIdentity[
        MatrixIdentity("Woodbury", "(A + UCV)^{-1} \\;\\to\\; A^{-1} - A^{-1}U\\left(C^{-1} + VA^{-1}U\\right)^{-1}VA^{-1}",
            _woodbury, false,
            () -> (@matsym A_[n, n] U_[n, k] C_[k, k] V_[k, n]; inv(A_ + U_ * C_ * V_))),
        MatrixIdentity("Sherman–Morrison", "(A + uv^{\\top})^{-1} \\;\\to\\; A^{-1} - \\dfrac{A^{-1}uv^{\\top}A^{-1}}{1 + v^{\\top}A^{-1}u}",
            _sherman_morrison, false,
            () -> (@matsym A_[n, n] u_[n] v_[n]; inv(A_ + u_ * v_'))),
        MatrixIdentity("inverse of a sum of inverses", "\\left(A^{-1} + B^{-1}\\right)^{-1} \\;\\to\\; A\\,(A + B)^{-1}B",
            _sum_of_inverses, false,
            () -> (@matsym A_[n, n] B_[n, n]; inv(inv(A_) + inv(B_)))),
        MatrixIdentity("push-through", "(I + AB)^{-1}A \\;\\to\\; A\\,(I + BA)^{-1}",
            _push_through, false,
            () -> (@matsym A_[n, m] B_[m, n]; inv(I + A_ * B_) * A_)),
        MatrixIdentity("commuting", "A\\,p(A)^{-1} \\;\\to\\; p(A)^{-1}A",
            _commute, false,
            () -> (@matsym A_[n, n]; A_ * inv(I + 2A_ + A_^2))),
        MatrixIdentity("Cayley–Hamilton (2×2)", "A^{2} \\;\\to\\; \\operatorname{tr}(A)\\,A - \\det(A)\\,I",
            _cayley_hamilton_2, false,
            () -> (@matsym A_[2, 2] B_[2, 2]; A_^2 * B_)),
        MatrixIdentity("Sylvester", "\\det(I + AB) \\;\\to\\; \\det(I + BA)",
            _sylvester, true,
            () -> (@matsym A_[3, 2] B_[2, 3]; det(I + A_ * B_))),
        MatrixIdentity("determinant lemma", "\\det(A + uv^{\\top}) \\;\\to\\; \\det A\\,\\left(1 + v^{\\top}A^{-1}u\\right)",
            _det_lemma, true,
            () -> (@matsym A_[n, n] u_[n] v_[n]; det(A_ + u_ * v_'))),
    ]
    return _MATRIX_IDENTITIES[] = ids
end

# (A + UCV)⁻¹ at its first inverse of such a sum (UCV a product of at least two factors)
function _woodbury(w, rows, cols)
    for (k, M) in _inverse_sums(w)
        length(M.terms) == 2 || continue
        for (a, b) in ((1, 2), (2, 1))
            (wa, ca), (wb, cb) = M.terms[a], M.terms[b]
            (length(wa) == 1 && _rows(wa[1]) == _cols(wa[1]) && length(wb) >= 2) || continue
            A = ca * _word_expr(wa, M.rows, M.cols)
            U = _word_expr(wb[1:1], M.rows, _cols(wb[1]))
            V = _word_expr(wb[end:end], _rows(wb[end]), M.cols)
            C = cb * (length(wb) == 2 ? identity_matrix(_cols(wb[1])) : _word_expr(wb[2:(end - 1)], _cols(wb[1]), _rows(wb[end])))
            # a rank-one term is Sherman–Morrison's
            (_is_vector(wb[1]) && _is_vector(wb[end]) && length(wb) == 2) && continue
            Ai = inv(A)
            R = Ai - Ai * U * inv(inv(C) + V * Ai * U) * V * Ai
            return _splice(w, k, R, rows, cols)
        end
    end
    return nothing
end

# (A + uvᵀ)⁻¹ with vectors u, v
function _sherman_morrison(w, rows, cols)
    for (k, M) in _inverse_sums(w)
        length(M.terms) == 2 || continue
        for (a, b) in ((1, 2), (2, 1))
            (wa, ca), (wb, cb) = M.terms[a], M.terms[b]
            (length(wb) == 2 && _is_vector(wb[1]) && _is_vector(wb[2]) && !wb[1].transposed && wb[2].transposed) || continue
            A = ca * _word_expr(wa, M.rows, M.cols)
            u = cb * _word_expr(wb[1:1], M.rows, 1)
            vt = _word_expr(wb[2:2], 1, M.cols)
            Ai = inv(A)
            R = Ai - (Ai * u * vt * Ai) / (1 + _scalar(vt * Ai * u))
            return _splice(w, k, R, rows, cols)
        end
    end
    return nothing
end

# (aA⁻¹ + bB⁻¹)⁻¹ = A (bA + aB)⁻¹ B
function _sum_of_inverses(w, rows, cols)
    for (k, M) in _inverse_sums(w)
        length(M.terms) == 2 || continue
        (wa, ca), (wb, cb) = M.terms
        (length(wa) == 1 && length(wb) == 1 && wa[1].power == -1 && wb[1].power == -1) || continue
        A = _word_expr(Factor[Factor(wa[1].base, wa[1].transposed, 1)], M.rows, M.cols)
        B = _word_expr(Factor[Factor(wb[1].base, wb[1].transposed, 1)], M.rows, M.cols)
        return _splice(w, k, A * inv(cb * A + ca * B) * B, rows, cols)
    end
    return nothing
end

# (aI + cAB)⁻¹ A = A (aI + cBA)⁻¹, where A is the factor after the inverse
function _push_through(w, rows, cols)
    for (k, M) in _inverse_sums(w)
        k < length(w) || continue
        p = _identity_plus(M)
        p === nothing && continue
        a, c, W = p
        length(W) >= 2 || continue
        _factor_key(first(W)) == _factor_key(w[k + 1]) || continue
        A = W[1:1]; B = W[2:end]
        n = _cols(A[1])
        inner = a * identity_matrix(n) + c * _word_expr(vcat(B, A), n, n)
        R = _word_expr(A, M.rows, n) * inv(inner)
        return _word_expr(w[1:(k - 1)], rows, M.rows) * R * _word_expr(w[(k + 2):end], n, cols)
    end
    return nothing
end

# A p(A)⁻¹ = p(A)⁻¹ A, where p is a polynomial in A: the inverse is moved to the left
function _commute(w, rows, cols)
    for (k, M) in _inverse_sums(w)
        k > 1 || continue
        f = w[k - 1]
        f.base isa MatrixSymbol && !f.transposed || continue
        all(t -> all(g -> g.base == f.base && !g.transposed, first(t)), M.terms) || continue
        swapped = vcat(w[1:(k - 2)], [w[k], f], w[(k + 1):end])
        return _word_expr(swapped, rows, cols)
    end
    return nothing
end

# A² = tr(A)A - det(A)I for a 2×2 A (a power of 2 or more of a 2×2 symbol)
function _cayley_hamilton_2(w, rows, cols)
    for (k, f) in enumerate(w)
        (f.base isa MatrixSymbol && f.base.rows == 2 && f.base.cols == 2 && f.power >= 2) || continue
        A = _word_expr(Factor[Factor(f.base, f.transposed, 1)], 2, 2)
        rest = f.power == 2 ? identity_matrix(2) : _word_expr(Factor[Factor(f.base, f.transposed, f.power - 2)], 2, 2)
        R = (LinearAlgebra.tr(A) * A - LinearAlgebra.det(A) * identity_matrix(2)) * rest
        return _splice(w, k, R, rows, cols)
    end
    return nothing
end

# det(aI + cW) = det(aI + cW′) for W′ a rotation of the product W. A rotation that is 1×1 is a
# scalar, det(1 + cx) = 1 + cx; otherwise the rotation with the smallest (numeric) size, then the
# one that comes first, is used, and none if that is W itself, so applying it again does nothing.
function _sylvester(M::MatExpr)
    p = _identity_plus(M)
    p === nothing && return nothing
    a, c, W = p
    length(W) >= 2 || return nothing
    size_rank(d) = d isa Int ? d : typemax(Int)
    rank(r) = (size_rank(_rows(r[1])), _word_rank(r))
    best = W
    for k in 1:(length(W) - 1)
        r = vcat(W[(k + 1):end], W[1:k])
        _rows(r[1]) == 1 && _cols(r[end]) == 1 && return a + c * _scalar(_word_expr(r, 1, 1))
        rank(r) < rank(best) && (best = r)
    end
    best === W && return nothing
    n = _rows(best[1])
    return det(a * identity_matrix(n) + c * _word_expr(best, n, n))
end

# det(A + cuvᵀ) = det(A)(1 + c vᵀA⁻¹u)
function _det_lemma(M::MatExpr)
    length(M.terms) == 2 || return nothing
    for (a, b) in ((1, 2), (2, 1))
        (wa, ca), (wb, cb) = M.terms[a], M.terms[b]
        (length(wb) == 2 && _is_vector(wb[1]) && _is_vector(wb[2]) && !wb[1].transposed && wb[2].transposed) || continue
        isempty(wa) && continue          # I + uvᵀ is Sylvester's
        A = ca * _word_expr(wa, M.rows, M.cols)
        u = _word_expr(wb[1:1], M.rows, 1)
        vt = _word_expr(wb[2:2], 1, M.cols)
        return LinearAlgebra.det(A) * (1 + cb * _scalar(vt * inv(A) * u))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Checking an identity on random numeric matrices
# ---------------------------------------------------------------------------------------------

"""
    verify(id::MatrixIdentity; samples = 3) -> Bool

Check a matrix identity on random numeric matrices: its example expression and the rewrite of it
are evaluated with random entries for every symbol (symmetric, positive definite, orthogonal or
skew as declared; sizes 2 to 4 for size names) and compared.
"""
function verify(id::MatrixIdentity; samples::Integer = 3)
    e = id.example()
    r = id(e)
    r === nothing && return false
    syms = _matrix_symbols(e)
    for _ in 1:samples
        sizes = Dict{Symbol, Int}()
        for s in syms, d in (s.rows, s.cols)
            d isa Symbol && !haskey(sizes, d) && (sizes[d] = 2 + length(sizes) % 3)
        end
        pairs = [MatExpr(s) => _random_matrix(s, sizes) for s in syms]
        a, b = _numeric(substitute(e, pairs...)), _numeric(substitute(r, pairs...))
        isapprox(a, b; rtol = 1e-8, atol = 1e-8) || return false
    end
    return true
end

_numeric(x::GiacExpr) = Float64(Giac.to_julia(Giac.Commands.evalf(x)))
_numeric(X::AbstractArray) = map(_numeric, X)

function _random_matrix(s::MatrixSymbol, sizes)
    dim(d) = d isa Int ? d : sizes[d]
    m, n = dim(s.rows), dim(s.cols)
    X = randn(m, n)
    _has(s, :posdef) && return X * X' + I
    _has(s, :negdef) && return -(X * X' + I)
    (_has(s, :symmetric) || _has(s, :diagonal)) && return _has(s, :diagonal) ? LinearAlgebra.Diagonal(LinearAlgebra.diag(X)) |> Matrix : (X + X') / 2
    _has(s, :skew) && return X - X'
    _has(s, :orthogonal) && return Matrix(LinearAlgebra.qr(X).Q)
    m == n && return X + 3I          # comfortably invertible
    return X
end

# the matrix symbols an expression (or a scalar with traces and determinants in it) involves
function _matrix_symbols(e, out = MatrixSymbol[])
    add(s::MatrixSymbol) = any(t -> t == s, out) || push!(out, s)
    if e isa MatExpr
        for (w, c) in e.terms
            for f in w
                f.base isa MatrixSymbol ? add(f.base) : _matrix_symbols(f.base, out)
            end
            _matrix_symbols(c, out)
        end
    elseif e isa GiacExpr
        for v in Giac.free_symbols(e)
            haskey(_ATOMS, string(v)) || continue
            kind, payload = _ATOMS[string(v)]
            payload isa MatrixSymbol ? add(payload) :
            payload isa MatExpr ? _matrix_symbols(payload, out) :
            _matrix_symbols(_matexpr(_rows(first(payload)), _cols(last(payload)), [payload => one(GiacExpr)]), out)
        end
    end
    return out
end
