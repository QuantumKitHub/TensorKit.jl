"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

# Permutation transformers
# ------------------------
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

function AbelianTreeTransformer(transform, p, Vdst, Vsrc, c)
    t₀ = Base.time()
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)

    T = sectorscalartype(sectortype(Vdst))
    N = numind(Vsrc)
    data = Vector{Tuple{T, StridedStructure{N}, StridedStructure{N}}}()
    transformer = AbelianTreeTransformer(data)

    isnothing(c) && return transformer

    L = length(structure_src.fusiontreelist)
    for i in 1:L
        f₁, f₂ = structure_src.fusiontreelist[i]
        (f₃, f₄), coeff = only(transform(f₁, f₂))
        f₃.coupled == c || continue # TODO this is probably very inefficient
        j = structure_dst.fusiontreeindices[(f₃, f₄)]
        stridestructure_dst = structure_dst.fusiontreestructure[j]
        stridestructure_src = structure_src.fusiontreestructure[i]
        push!(data, (coeff, stridestructure_dst, stridestructure_src))
    end

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


# Contraction transformers
# ------------------------

const _AbelianContractionTransformerData{T, N₁, N₂, N₃} = Tuple{
    NTuple{3, StridedStructure{2}},                  # block data
    AbelianTreeTransformer{T, N₁},                 # permute A
    AbelianTreeTransformer{T, N₂},                  # permute B
    AbelianTreeTransformer{T, N₃},                  # invpermute C
}

struct AbelianContractionTreeTransformer{T, N₁, N₂, N₃} <: TreeTransformer
    data::Vector{_AbelianContractionTransformerData{T, N₁, N₂, N₃}}
end

_trivtuple(::Index2Tuple{N₁, N₂}) where {N₁, N₂} = (ntuple(identity, N₁), ntuple(i -> i + N₁, N₂))

function needsbuffer(treetransformer::AbelianContractionTreeTransformer, i)
    return !isempty(first(treetransformer.data)[i + 1].data)
end

function AbelianContractionTreeTransformer(VC, VA, pA, VB, pB, pAB)
    VA′ = permute(VA, pA)
    VB′ = permute(VB, pB)
    VC′ = compose(VA′, VB′)

    T = sectorscalartype(sectortype(VC))
    bCs = blocksectors(VC′)

    data = Vector{_AbelianContractionTransformerData{T, numind(VA), numind(VB), numind(VC)}}(undef, length(bCs))

    pAnew = _trivtuple(pA)
    pBnew = _trivtuple(pB)
    pABnew = _trivtuple(pAB)

    # sanity check
    ipAB = TO.oindABinC(pAB, pAnew, pBnew)
    @assert VC′ == permute(VC, ipAB)

    treepermuterA(f1, f2) = permute(f1, f2, pA...)
    treepermuterB(f1, f2) = permute(f1, f2, pB...)
    treepermuterAB(f1, f2) = permute(f1, f2, pAB...)

    blockstructuresA = fusionblockstructure(VA′).blockstructure
    blockstructuresB = fusionblockstructure(VB′).blockstructure
    blockstructuresC = fusionblockstructure(VC′).blockstructure


    for (i, c) in enumerate(bCs)
        # how do we transform an individual block of A or B
        # careful, we have to permute INTO C, not from C
        transformA = AbelianTreeTransformer(treepermuterA, pA, VA′, VA, pA == pAnew ? nothing : c)
        transformB = AbelianTreeTransformer(treepermuterB, pB, VB′, VB, pB == pBnew ? nothing : c)
        invtransformC = AbelianTreeTransformer(treepermuterAB, pAB, VC, VC′, pAB == pABnew ? nothing : c)

        # where is the data for the given block
        blockszC, blockrangeC = get(blockstructuresC, c, ((0, 0), 1:0))
        blockstructureC = (blockszC, (1, blockszC[1]), first(blockrangeC) - 1)

        blockszA, blockrangeA = get(blockstructuresA, c, ((blockszC[1], 0), 1:0))
        blockstructureA = (blockszA, (1, blockszA[1]), first(blockrangeA) - 1)

        blockszB, blockrangeB = get(blockstructuresB, c, ((0, blockszC[2]), 1:0))
        blockstructureB = (blockszB, (1, blockszB[1]), first(blockrangeB) - 1)


        data[i] = (
            (blockstructureA, blockstructureB, blockstructureC),
            transformA, transformB, invtransformC,
        )
    end

    return AbelianContractionTreeTransformer(data)
end

function contractiontransformertype(VC, VA, VB)
    T = sectorscalartype(sectortype(VC))
    return AbelianContractionTreeTransformer{T, numind(VA), numind(VB), numind(VC)}
end

@cached function contractiontransformer(
        VC::TensorMapSpace,
        VA::TensorMapSpace, pA::Index2Tuple,
        VB::TensorMapSpace, pB::Index2Tuple,
        pAB::Index2Tuple
    )::contractiontransformertype(VC, VA, VB)
    return AbelianContractionTreeTransformer(VC, VA, pA, VB, pB, pAB)
end
