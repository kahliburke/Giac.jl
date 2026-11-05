# GiacSparseArraysExt: sparse Jacobian prototypes with the structure of a symbolic Jacobian.

module GiacSparseArraysExt

using Giac
using SparseArrays: sparse

function Giac._sparse_pattern(mask::AbstractMatrix{Bool})
    I = findall(mask)
    return sparse(getindex.(I, 1), getindex.(I, 2), ones(length(I)), size(mask)...)
end

end # module
