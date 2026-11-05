# GiacStaticArraysExt: generated out-of-place code returns static arrays for a static state.

module GiacStaticArraysExt

using Giac
using StaticArrays: StaticArray, SArray, SVector

Giac._array_like(::StaticArray, ::Val{dims}, entries::Tuple) where {dims} = SArray{Tuple{dims...}}(entries)
Giac._static(v::AbstractVector) = SVector{length(v)}(v)

end # module
