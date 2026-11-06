# GiacSparseArraysExt: sparse Jacobian prototypes with the structure of a symbolic Jacobian, and
# their stored values for generated code to write.

module GiacSparseArraysExt

using Giac
using SparseArrays: SparseMatrixCSC, nonzeros, sparse

function Giac._sparse_pattern(mask::AbstractMatrix{Bool})
    I = findall(mask)
    return sparse(getindex.(I, 1), getindex.(I, 2), ones(length(I)), size(mask)...)
end

Giac._stored_values(out::SparseMatrixCSC) = nonzeros(out)

end # module
