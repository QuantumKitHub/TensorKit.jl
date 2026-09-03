"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.

The transformers only store how subblocks map onto each other in terms of their positions in
[`StridedSubblocks`](@ref) (the canonical order of `fusiontrees`), together with the recoupling
coefficients, and are therefore independent of the sectortype once constructed. The
transformation is that of `permutedims(op(tsrc), p)` where `p` indexes the legs of `tsrc` itself and
`op` is either `identity` or `conj`; in the latter case the fusion trees that are transformed are
those of `adjoint(space(tsrc))`, which read the subblocks of `tsrc` with the tree pair swapped.
"""
abstract type TreeTransformer end

# (coefficient, destination position, source position)
const AbelianTransformerData{T} = Tuple{T, Int, Int}

"""
    AbelianTreeTransformer{T, N} <: TreeTransformer

Tree transformation for `UniqueFusion` sectors, where every source subblock maps onto a single
destination subblock with a scalar coefficient, stored as `(coeff, idst, isrc)`. The subblock
structures of the destination and source spaces are kept alongside, such that the
[`StridedSubblocks`](@ref) of both tensors can be created without further lookups.
"""
struct AbelianTreeTransformer{T, N} <: TreeTransformer
    data::Vector{AbelianTransformerData{T}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
end

# (recoupling matrix, destination positions, source positions): U[j, i] maps source i onto destination j
const GenericTransformerData{T} = Tuple{Matrix{T}, Vector{Int}, Vector{Int}}

"""
    GenericTreeTransformer{T, N} <: TreeTransformer

Tree transformation for sectors with multiple fusion channels, where the subblocks of a
[`FusionTreeBlock`](@ref) map onto the subblocks of the transformed block through a recoupling
matrix, stored as `(U, inds_dst, inds_src)`. The subblock structures of the destination and
source spaces are kept alongside, such that the [`StridedSubblocks`](@ref) of both tensors can be
created without further lookups.
"""
struct GenericTreeTransformer{T, N} <: TreeTransformer
    data::Vector{GenericTransformerData{T}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
end

# Space and permutation as seen by the tree manipulations: for a conjugated source these are
# those of the adjoint space, whose tree pairs `(f₁, f₂)` address the subblock `(f₂, f₁)` of the source.
function transform_source(Vsrc::HomSpace, p::Index2Tuple, conjsrc::Bool)
    return conjsrc ? (adjoint(Vsrc), adjointtensorindices(Vsrc, p)) : (Vsrc, p)
end
source_tree((f₁, f₂)::FusionTreePair, conjsrc::Bool) = conjsrc ? (f₂, f₁) : (f₁, f₂)

function AbelianTreeTransformer(transform, p, Vdst, Vsrc, conjsrc::Bool)
    t₀ = Base.time()
    Vsrc′, p′ = transform_source(Vsrc, p, conjsrc)
    spacecheck_transform(permute, Vdst, Vsrc′, p′)
    srcindices = fusiontreeindices(Vsrc)
    dstindices = fusiontreeindices(Vdst)
    trees_src = fusiontrees(Vsrc′)
    L = length(trees_src)
    T = sectorscalartype(sectortype(Vdst))
    data = Vector{AbelianTransformerData{T}}(undef, L)

    for (i, f_src) in enumerate(trees_src)
        f_dst, coeff = transform(f_src)
        data[i] = (coeff, dstindices[f_dst], srcindices[source_tree(f_src, conjsrc)])
    end

    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    transformer = AbelianTreeTransformer(data, structure_dst, structure_src)

    Δt = Base.time() - t₀
    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", conjsrc, nblocks = L, Δt)

    return transformer
end

function GenericTreeTransformer(transform, p, Vdst, Vsrc, conjsrc::Bool)
    t₀ = Base.time()
    Vsrc′, p′ = transform_source(Vsrc, p, conjsrc)
    spacecheck_transform(permute, Vdst, Vsrc′, p′)
    srcindices = fusiontreeindices(Vsrc)
    dstindices = fusiontreeindices(Vdst)
    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    T = sectorscalartype(sectortype(Vsrc))

    fblocks = fusionblocks(Vsrc′)
    nblocks = length(fblocks)
    data = Vector{GenericTransformerData{T}}(undef, nblocks)
    weights = Vector{Int}(undef, nblocks)

    nthreads = get_num_manipulation_threads()
    taskforeach(1:nblocks, nthreads) do i
        fs_src = fblocks[i]
        fs_dst, U = transform(fs_src)
        inds_src = map(f -> srcindices[source_tree(f, conjsrc)], fusiontrees(fs_src))
        inds_dst = map(f -> dstindices[f], fusiontrees(fs_dst))
        data[i] = (U, inds_dst, inds_src)
        # cost model: L input blocks each going to L output blocks of a given length
        weights[i] = length(U) * prod(structure_dst[first(inds_dst)][1])

        @debug(
            lazy"Created recoupling block for uncoupled: $(fs_src.uncoupled)",
            sz = size(U), sparsity = count(!iszero, U) / length(U)
        )
    end

    # sort by (approximate) weight to facilitate multi-threading strategies
    Base.permute!(data, sortperm(weights; rev = true))
    transformer = GenericTreeTransformer(data, structure_dst, structure_src)

    Δt = Base.time() - t₀
    @debug(
        lazy"TreeTransformer for $Vsrc to $Vdst via $p", conjsrc,
        nblocks = nblocks,
        sz_median = nblocks > 0 ? size(data[cld(end, 2)][1], 1) : 0,
        sz_max = nblocks > 0 ? size(data[1][1], 1) : 0,
        Δt
    )

    return transformer
end

"""
    buffersize(transformer::TreeTransformer) -> Int

Compute the workspace size required to pack, recouple and unpack the largest multi-tree
block, i.e. `prod(sz_src) * (rows + cols)` where `(rows, cols) = size(U)` is the size of
the recoupling matrix.
"""
buffersize(::AbelianTreeTransformer) = 0
function buffersize(transformer::GenericTreeTransformer)
    structure_src = transformer.structure_src
    return maximum(transformer.data; init = 0) do (U, _, inds_src)
        return length(U) == 1 ? 0 : prod(structure_src[first(inds_src)][1]) * sum(size(U))
    end
end

function treetransformertype(Vdst, Vsrc)
    I = sectortype(Vdst)
    T = sectorscalartype(I)
    N = numind(Vdst)
    return FusionStyle(I) == UniqueFusion() ? AbelianTreeTransformer{T, N} : GenericTreeTransformer{T, N}
end

function TreeTransformer(
        transform::Function, p, Vdst::HomSpace{S}, Vsrc::HomSpace{S}, conjsrc::Bool
    ) where {S}
    I = sectortype(Vdst)
    return FusionStyle(I) == UniqueFusion() ?
        AbelianTreeTransformer(transform, p, Vdst, Vsrc, conjsrc) :
        GenericTreeTransformer(transform, p, Vdst, Vsrc, conjsrc)
end

# braid is special because it has levels
function treebraider(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )
    return treebraider(space(tdst), space(tsrc), p, conjsrc, levels)
end
@cached function treebraider(
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )::treetransformertype(Vdst, Vsrc)
    Vsrc′, p′ = transform_source(Vsrc, p, conjsrc)
    # levels are attached to the legs, so they follow the same relabeling as the permutation
    levels′ = conjsrc ? TupleTools.getindices(levels, adjointtensorindices(Vsrc′, allind(Vsrc′))) : levels
    levels″ = (TupleTools.getindices(levels′, codomainind(Vsrc′)), TupleTools.getindices(levels′, domainind(Vsrc′)))
    fusiontreebraider(f) = braid(f, p′, levels″)
    return TreeTransformer(fusiontreebraider, p, Vdst, Vsrc, conjsrc)
end

function treetransposer(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, conjsrc::Bool)
    return treetransposer(space(tdst), space(tsrc), p, conjsrc)
end
@cached function treetransposer(
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    )::treetransformertype(Vdst, Vsrc)
    Vsrc′, p′ = transform_source(Vsrc, p, conjsrc)
    fusiontreetransform(f) = transpose(f, p′)
    return TreeTransformer(fusiontreetransform, p, Vdst, Vsrc, conjsrc)
end

# default cachestyle is GlobalLRUCache

# For CPU arrays the recoupling matrix can be used as is, also when the scalar types
# do not match, since Strided handles mixed-eltype mul! without the copy that
# Adapt.adapt would make (which additionally dispatches dynamically). Other storage
# types (e.g. GPU arrays) do require the conversion.
# TODO: transformers with dedicated storagetypes
_adapt_recoupling(::Type{<:Array}, U::Matrix) = StridedView(U)
_adapt_recoupling(::Type{A}, U::Matrix) where {A} = Adapt.adapt(A, StridedView(U))
