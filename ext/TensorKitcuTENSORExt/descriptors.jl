# Block-sparse tensor descriptors
# -------------------------------
# The low-level `CuTensorBSDescriptor` constructor is used deliberately: the `::CuTensorBS`
# one asserts contiguous blocks, whereas a TensorKit subblock is a strided view.

"build a cuTENSOR block-sparse descriptor for the layout of `W` with scalar type `T`"
function bs_descriptor(W::HomSpace, ::Type{T}) where {T}
    s = blocksparsestructure(W)
    N = nmodes(s)
    N <= MAX_BLOCKSPARSE_MODES || throw(
        BlockSparseUnsupported(
            "cuTENSOR's block-sparse backend supports at most $MAX_BLOCKSPARSE_MODES modes, got $N"
        )
    )
    # cuTENSOR wants one permutation sorting every block's strides ascending; `degeneracystructure`
    # makes that the identity. `vec` on a `Matrix` is a zero-copy reshape.
    return cuTENSOR.CuTensorBSDescriptor(
        Int32(N), Int64(nblocks(s)),
        s.numsections, s.extent, vec(s.coordinates), vec(s.strides), T
    )
end

"fill the host array `ptrs` with a device pointer to each non-zero block of `t`, in canonical order"
function block_pointers!(ptrs::Vector{CuPtr{Cvoid}}, t::CuTensorMapAny)
    s = blocksparsestructure(space(t))
    base = convert(CuPtr{Cvoid}, pointer(t.data))
    elsz = sizeof(scalartype(t))
    resize!(ptrs, nblocks(s))
    @inbounds for i in eachindex(ptrs)
        ptrs[i] = base + s.offsets[i] * elsz
    end
    return ptrs
end

block_pointers(t::CuTensorMapAny) = block_pointers!(Vector{CuPtr{Cvoid}}(), t)
