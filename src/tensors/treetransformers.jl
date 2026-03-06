"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

struct TrivialTreeTransformer <: TreeTransformer end

const _AbelianTransformerData{T, N} = Tuple{T, StridedStructure{N}, StridedStructure{N}}

struct AbelianTreeTransformer{T, N} <: TreeTransformer
    data::Vector{_AbelianTransformerData{T, N}}
end

function AbelianTreeTransformer(transform, p, Vdst, Vsrc)
    t₀ = Base.time()
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    L = length(structure_src.fusiontreelist)
    T = sectorscalartype(sectortype(Vdst))
    N = numind(Vsrc)
    data = Vector{Tuple{T, StridedStructure{N}, StridedStructure{N}}}(undef, L)

    for i in 1:L
        f₁, f₂ = structure_src.fusiontreelist[i]
        (f₃, f₄), coeff = only(transform(f₁, f₂))
        j = structure_dst.fusiontreeindices[(f₃, f₄)]
        stridestructure_dst = structure_dst.fusiontreestructure[j]
        stridestructure_src = structure_src.fusiontreestructure[i]
        data[i] = (coeff, stridestructure_dst, stridestructure_src)
    end

    transformer = AbelianTreeTransformer(data)

    # sort by (approximate) weight to facilitate multi-threading strategies
    # sort!(transformer)

    Δt = Base.time() - t₀

    @debug("Treetransformer for $Vsrc to $Vdst via $p", nblocks = L, Δt)

    return transformer
end

const _GenericTransformerData{T, N} = Tuple{
    Matrix{T},
    Tuple{NTuple{N, Int}, Vector{Tuple{NTuple{N, Int}, Int}}},
    Tuple{NTuple{N, Int}, Vector{Tuple{NTuple{N, Int}, Int}}},
}

struct GenericTreeTransformer{T, N} <: TreeTransformer
    data::Vector{_GenericTransformerData{T, N}}
end

function GenericTreeTransformer(transform, p, Vdst, Vsrc)
    t₀ = Base.time()
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    fusionstructure_dst = structure_dst.fusiontreestructure
    structure_src = fusionblockstructure(Vsrc)
    fusionstructure_src = structure_src.fusiontreestructure
    I = sectortype(Vsrc)

    uncoupleds_src = map(structure_src.fusiontreelist) do (f₁, f₂)
        return TupleTools.vcat(f₁.uncoupled, dual.(f₂.uncoupled))
    end
    uncoupleds_src_unique = unique(uncoupleds_src)

    uncoupleds_dst = map(structure_dst.fusiontreelist) do (f₁, f₂)
        return TupleTools.vcat(f₁.uncoupled, dual.(f₂.uncoupled))
    end

    T = sectorscalartype(I)
    N = numind(Vdst)
    L = length(uncoupleds_src_unique)
    data = Vector{_GenericTransformerData{T, N}}(undef, L)

    # TODO: this can be multithreaded
    for (i, uncoupled) in enumerate(uncoupleds_src_unique)
        inds_src = findall(==(uncoupled), uncoupleds_src)
        fusiontrees_outer_src = structure_src.fusiontreelist[inds_src]

        uncoupled_dst = TupleTools.getindices(uncoupled, (p[1]..., p[2]...))
        inds_dst = findall(==(uncoupled_dst), uncoupleds_dst)

        fusiontrees_outer_dst = structure_dst.fusiontreelist[inds_dst]

        matrix = zeros(sectorscalartype(I), length(inds_dst), length(inds_src))
        for (row, (f₁, f₂)) in enumerate(fusiontrees_outer_src)
            for ((f₃, f₄), coeff) in transform(f₁, f₂)
                col = findfirst(==((f₃, f₄)), fusiontrees_outer_dst)::Int
                matrix[row, col] = coeff
            end
        end

        # size is shared between blocks, so repack:
        # from [(sz, strides, offset), ...] to (sz, [(strides, offset), ...])
        sz_src, newstructs_src = repack_transformer_structure(fusionstructure_src, inds_src)
        sz_dst, newstructs_dst = repack_transformer_structure(fusionstructure_dst, inds_dst)

        @debug(
            "Created recoupling block for uncoupled: $uncoupled",
            sz = size(matrix), sparsity = count(!iszero, matrix) / length(matrix)
        )

        data[i] = (matrix, (sz_dst, newstructs_dst), (sz_src, newstructs_src))
    end

    transformer = GenericTreeTransformer{T, N}(data)

    # sort by (approximate) weight to facilitate multi-threading strategies
    sort!(transformer)

    Δt = Base.time() - t₀

    @debug(
        "TreeTransformer for $Vsrc to $Vdst via $p",
        nblocks = length(data),
        sz_median = size(data[cld(end, 2)][1], 1),
        sz_max = size(data[1][1], 1),
        Δt
    )

    return transformer
end

function repack_transformer_structure(structures, ids)
    sz = structures[first(ids)][1]
    strides_offsets = map(i -> (structures[i][2], structures[i][3]), ids)
    return sz, strides_offsets
end

function buffersize(transformer::GenericTreeTransformer)
    return maximum(transformer.data; init = 0) do (basistransform, structures_dst, _)
        return prod(structures_dst[1]) * size(basistransform, 1)
    end
end

function allocate_buffers(
        tdst::TensorMap, tsrc::TensorMap, transformer::GenericTreeTransformer
    )
    sz = buffersize(transformer)
    return similar(tdst.data, sz), similar(tsrc.data, sz)
end

function treetransformertype(Vdst, Vsrc)
    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer

    T = sectorscalartype(I)
    N = numind(Vdst)
    return FusionStyle(I) == UniqueFusion() ? AbelianTreeTransformer{T, N} : GenericTreeTransformer{T, N}
end

function TreeTransformer(
        transform::Function, p, Vdst::HomSpace{S}, Vsrc::HomSpace{S}
    ) where {S}
    permute(Vsrc, p) == Vdst ||
        throw(SpaceMismatch("Incompatible spaces for permuting"))

    I = sectortype(Vdst)
    I === Trivial && return TrivialTreeTransformer()

    return FusionStyle(I) == UniqueFusion() ?
        AbelianTreeTransformer(transform, p, Vdst, Vsrc) :
        GenericTreeTransformer(transform, p, Vdst, Vsrc)
end

# braid is special because it has levels
function treebraider(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple, levels)
    return fusiontreetransform(f1, f2) = braid(f1, f2, levels..., p...)
end
function treebraider(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple, levels)
    return treebraider(space(tdst), space(tsrc), p, levels)
end
@cached function treebraider(
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, levels
    )::treetransformertype(Vdst, Vsrc)
    fusiontreebraider(f1, f2) = braid(f1, f2, levels..., p...)
    return TreeTransformer(fusiontreebraider, p, Vdst, Vsrc)
end

for (transform, treetransformer) in
    ((:permute, :treepermuter), (:transpose, :treetransposer))
    @eval begin
        function $treetransformer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
            return fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
        end
        function $treetransformer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
            return $treetransformer(space(tdst), space(tsrc), p)
        end
        @cached function $treetransformer(
                Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple
            )::treetransformertype(Vdst, Vsrc)
            fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
            return TreeTransformer(fusiontreetransform, p, Vdst, Vsrc)
        end
    end
end

# default cachestyle is GlobalLRUCache

# Sorting based on cost model
# ---------------------------
function Base.sort!(
        transformer::Union{AbelianTreeTransformer, GenericTreeTransformer};
        by = _transformer_weight, rev::Bool = true
    )
    sort!(transformer.data; by, rev)
    return transformer
end

function _transformer_weight((coeff, struct_dst, struct_src)::_AbelianTransformerData)
    return prod(struct_dst[1])
end

# Cost model for transforming a set of subblocks with fixed uncoupled sectors:
# L x L x length(subblock) where L is the number of subblocks
# this is L input blocks each going to L output blocks of given length
# Note that it might be the case that the permutations are dominant, in which case the
# actual cost model would scale like L x length(subblock)
function _transformer_weight((mat, structs_dst, structs_src)::_GenericTransformerData)
    return length(mat) * prod(structs_dst[1])
end
