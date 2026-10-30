"""
    GiacBlockArraysExt

Block matrices of expressions with BlockArrays.jl: they typeset with partition lines at the block
boundaries, and products of conforming block matrices work (and stay partitioned).

```julia
using Giac, BlockArrays, LinearAlgebra
@giac_matrix A[2, 2]
M = BlockArray([A I; I A'], [2, 2], [2, 2])
M[Block(1, 2)]
M * M
```
"""
module GiacBlockArraysExt

using Giac: Giac, GiacExpr
using BlockArrays: BlockArrays, AbstractBlockMatrix, AbstractBlockVector, blocklengths

# the last row (or column) of every block but the final one: where a partition line goes
_block_ends(ax) = Set(cumsum(blocklengths(ax))[1:end-1])

Giac.giac_latex(M::AbstractBlockMatrix{GiacExpr}) =
    Giac._matrix_latex(M; rowbreaks = _block_ends(axes(M, 1)), colbreaks = _block_ends(axes(M, 2)))
Giac.giac_latex(v::AbstractBlockVector{GiacExpr}) = Giac._matrix_latex(v; rowbreaks = _block_ends(axes(v, 1)))

# A block product starts from a lazy array of zeros, which ArrayLayouts builds only for `Number`
# element types; an expression has a `zero` too.
BlockArrays.ArrayLayouts.mulzeros(::Type{GiacExpr}, M) = BlockArrays.FillArrays.Zeros{GiacExpr}(axes(M))

end
