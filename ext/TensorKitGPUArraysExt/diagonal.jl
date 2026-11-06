# Diagonal sources
# ----------------
# An index manipulation of a `DiagonalTensorMap` maps the diagonal of every source subblock onto
# the diagonal of a destination subblock, i.e. every block is a one-dimensional strided copy.

# position of each subblock in the data of a `DiagonalTensorMap` in the same space
_diagonal_offsets(structure) = first(_work_offsets(first(first(s)) for s in structure))

_diagonal_entries(transformer::UniqueTreeTransformer) = transformer.data
# `V ← V` and any rearrangement of its two indices have a single tree for every subblock
_diagonal_entries(transformer::GenericTreeTransformer) =
    ((only(U), only(inds_dst), only(inds_src)) for (U, inds_dst, inds_src) in transformer.data)

function diagonal_device_transformer(A, transformer, dstdiagonal::Bool)
    (; structure_dst, structure_src) = transformer
    offsets_src = _diagonal_offsets(structure_src)
    offsets_dst = dstdiagonal ? _diagonal_offsets(structure_dst) : nothing
    entries = map(_diagonal_entries(transformer)) do (coeff, idst, isrc)
        (sz, strides_dst, offset_dst) = structure_dst[idst]
        n = first(sz)
        s_dst = dstdiagonal ? ((n,), (1,), offsets_dst[idst]) : ((n,), (sum(strides_dst),), offset_dst)
        return (fill(coeff, 1, 1), (s_dst,), (((n,), (1,), offsets_src[isrc]),))
    end
    return device_transformer(A, DeviceTreeTransformer(scalartype(transformer), entries, (1,), sizeof(scalartype(A))))
end

@cached function diagonal_treebraider(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
        conjsrc::Bool, levels::IndexTuple, dstdiagonal::Bool
    )::devicetransformertype(A, sectortype(Vdst), Val(1)) where {TA <: AnyGPUVector}
    host = TensorKit.treebraider(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc, levels)
    return diagonal_device_transformer(A, host, dstdiagonal)
end

@cached function diagonal_treetransposer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
        conjsrc::Bool, dstdiagonal::Bool
    )::devicetransformertype(A, sectortype(Vdst), Val(1)) where {TA <: AnyGPUVector}
    host = TensorKit.treetransposer(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc)
    return diagonal_device_transformer(A, host, dstdiagonal)
end

@cached function diagonal_treetwister(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
        conjsrc::Bool, twistmask::NTuple{N, Bool}, inv::Bool, dstdiagonal::Bool
    )::devicetransformertype(A, sectortype(Vdst), Val(1)) where {TA <: AnyGPUVector, N}
    host = TensorKit.treetwister(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc, twistmask, inv)
    return diagonal_device_transformer(A, host, dstdiagonal)
end

function _add_diagonal!(tdst::GPUDiagonalTensorMap, tsrc, conjsrc, transformer, α, β)
    _launch_tiled!(tdst.data, tsrc.data, conjsrc, transformer, α, β)
    return tdst
end
# the kernel only touches the diagonal. be careful to apply `β` to the entire tensor.
function _add_diagonal!(tdst::GPUTensorMap, tsrc, conjsrc, transformer, α, β)
    scale!(tdst, β)
    _launch_tiled!(tdst.data, tsrc.data, conjsrc, transformer, α, one(scalartype(tdst)))
    return tdst
end

Base.@propagate_inbounds function TensorKit._braid!(
        tdst::Union{GPUTensorMap, GPUDiagonalTensorMap}, tsrc::GPUDiagonalTensorMap,
        p::Index2Tuple, conjsrc::Bool, levels::IndexTuple, α, β, backend, allocator
    )
    Vdst, Vsrc = space(tdst), space(tsrc)
    @boundscheck TensorKit.spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)
    transformer = diagonal_treebraider(
        storagetype(tdst), Vdst, Vsrc, p, conjsrc, levels, tdst isa DiagonalTensorMap
    )
    return _add_diagonal!(tdst, tsrc, conjsrc, transformer, α, β)
end

Base.@propagate_inbounds function TensorKit._transpose!(
        tdst::Union{GPUTensorMap, GPUDiagonalTensorMap}, tsrc::GPUDiagonalTensorMap,
        p::Index2Tuple, conjsrc::Bool, α, β, backend, allocator
    )
    Vdst, Vsrc = space(tdst), space(tsrc)
    @boundscheck TensorKit.spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)
    transformer = diagonal_treetransposer(
        storagetype(tdst), Vdst, Vsrc, p, conjsrc, tdst isa DiagonalTensorMap
    )
    return _add_diagonal!(tdst, tsrc, conjsrc, transformer, α, β)
end

Base.@propagate_inbounds function TensorKit._twist!(
        tdst::Union{GPUTensorMap, GPUDiagonalTensorMap}, tsrc::GPUDiagonalTensorMap,
        p::Index2Tuple, conjsrc::Bool, twistmask::Tuple{Vararg{Bool}}, inv::Bool, α, β, backend, allocator
    )
    Vdst, Vsrc = space(tdst), space(tsrc)
    @boundscheck TensorKit.spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)
    transformer = diagonal_treetwister(
        storagetype(tdst), Vdst, Vsrc, p, conjsrc, twistmask, inv, tdst isa DiagonalTensorMap
    )
    return _add_diagonal!(tdst, tsrc, conjsrc, transformer, α, β)
end
