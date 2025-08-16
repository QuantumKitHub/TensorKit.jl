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
        (f₃, f₄), coeff = only(transform((f₁, f₂)))
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
    T = sectorscalartype(I)
    N = numind(Vdst)
    N₁ = numout(Vsrc)
    N₂ = numin(Vsrc)

    isdual_src = (map(isdual, codomain(Vsrc).spaces), map(isdual, domain(Vsrc).spaces))

    data = Vector{_GenericTransformerData{T, N}}()

    nthreads = get_num_transformer_threads()
    if nthreads > 1
        fusiontreeblocks = Vector{FusionTreeBlock{I, N₁, N₂, fusiontreetype(I, N₁, N₂)}}()
        for cod_uncoupled_src in sectors(codomain(Vsrc)),
                dom_uncoupled_src in sectors(domain(Vsrc))

            fs_src = FusionTreeBlock{I}((cod_uncoupled_src, dom_uncoupled_src), isdual_src)
            trees_src = fusiontrees(fs_src)
            if !isempty(trees_src)
                push!(fusiontreeblocks, fs_src)
            end
        end

        resize!(data, length(fusiontreeblocks))
        counter = Threads.Atomic{Int}(1)
        Threads.@sync for _ in 1:min(nthreads, length(fusiontreeblocks))
            Threads.@spawn begin
                while true
                    local_counter = Threads.atomic_add!(counter, 1)
                    local_counter > nblocks && break
                    fs_src = fusiontreeblocks[local_counter]
                    fs_dst, U = transform(fs_src)
                    matrix = copy(transpose(U)) # TODO: should we avoid this

                    inds_src = map(
                        Base.Fix1(getindex, structure_src.fusiontreeindices), trees_src
                    )
                    trees_dst = fusiontrees(fs_dst)
                    inds_dst = map(
                        Base.Fix1(getindex, structure_dst.fusiontreeindices), trees_dst
                    )

                    # size is shared between blocks, so repack:
                    # from [(sz, strides, offset), ...] to (sz, [(strides, offset), ...])
                    sz_src, newstructs_src = repack_transformer_structure(
                        fusionstructure_src, inds_src
                    )
                    sz_dst, newstructs_dst = repack_transformer_structure(
                        fusionstructure_dst, inds_dst
                    )

                    data1[local_counter] = (
                        matrix, (sz_dst, newstructs_dst),
                        (sz_src, newstructs_src),
                    )
                end
            end
        end
        transformer = GenericTreeTransformer{T, N}(data)
    else
        isdual_src = (map(isdual, codomain(Vsrc).spaces), map(isdual, domain(Vsrc).spaces))
        for cod_uncoupled_src in sectors(codomain(Vsrc)),
                dom_uncoupled_src in sectors(domain(Vsrc))

            fs_src = FusionTreeBlock{I}((cod_uncoupled_src, dom_uncoupled_src), isdual_src)
            trees_src = fusiontrees(fs_src)
            isempty(trees_src) && continue

            fs_dst, U = transform(fs_src)
            matrix = copy(transpose(U)) # TODO: should we avoid this

            inds_src = map(Base.Fix1(getindex, structure_src.fusiontreeindices), trees_src)
            trees_dst = fusiontrees(fs_dst)
            inds_dst = map(Base.Fix1(getindex, structure_dst.fusiontreeindices), trees_dst)

            # size is shared between blocks, so repack:
            # from [(sz, strides, offset), ...] to (sz, [(strides, offset), ...])
            sz_src, newstructs_src = repack_transformer_structure(
                fusionstructure_src, inds_src
            )
            sz_dst, newstructs_dst = repack_transformer_structure(
                fusionstructure_dst, inds_dst
            )

            push!(data, (matrix, (sz_dst, newstructs_dst), (sz_src, newstructs_src)))
        end
        transformer = GenericTreeTransformer{T, N}(data)
    end

    # sort by (approximate) weight to facilitate multi-threading strategies
    sort!(transformer)

    Δt = Base.time() - t₀

    @debug(
        "TreeTransformer for $Vsrc to $Vdst via $p",
        nblocks = length(transformer.data),
        sz_median = size(transformer.data[cld(end, 2)][1], 1),
        sz_max = size(transformer.data[1][1], 1),
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
    return fusiontreetransform(f) = braid(f, p, levels)
end
function treebraider(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple, levels)
    return treebraider(space(tdst), space(tsrc), p, levels)
end
@cached function treebraider(
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, levels
    )::treetransformertype(Vdst, Vsrc)
    fusiontreebraider(f) = braid(f, p, levels)
    return TreeTransformer(fusiontreebraider, p, Vdst, Vsrc)
end

for (transform, treetransformer) in
    ((:permute, :treepermuter), (:transpose, :treetransposer))
    @eval begin
        function $treetransformer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
            return fusiontreetransform(f) = $transform(f, p)
        end
        function $treetransformer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
            return $treetransformer(space(tdst), space(tsrc), p)
        end
        @cached function $treetransformer(
                Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple
            )::treetransformertype(Vdst, Vsrc)
            fusiontreetransform(f) = $transform(f, p)
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
