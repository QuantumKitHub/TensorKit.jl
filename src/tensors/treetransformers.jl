"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

struct TrivialTreeTransformer <: TreeTransformer end

const AbelianTransformerData{T, N} = Tuple{T, StridedStructure{N}, StridedStructure{N}}

struct AbelianTreeTransformer{T, N} <: TreeTransformer
    data::Vector{AbelianTransformerData{T, N}}
end

function AbelianTreeTransformer(transform, p, Vdst, Vsrc)
    t₀ = Base.time()
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    fts_src = subblockstructure(Vsrc)
    fts_dst = subblockstructure(Vdst)
    L = length(fts_src)
    T = sectorscalartype(sectortype(Vdst))
    N = numind(Vsrc)
    data = Vector{Tuple{T, StridedStructure{N}, StridedStructure{N}}}(undef, L)

    @timeit_debug GLOBAL_TIMER "symmetry: tree transform" for (i, (f_src, stridestructure_src)) in enumerate(pairs(fts_src))
        f_dst, coeff = transform(f_src)
        stridestructure_dst = fts_dst[f_dst]
        data[i] = (coeff, stridestructure_dst, stridestructure_src)
    end

    transformer = AbelianTreeTransformer(data)

    # sort by (approximate) weight to facilitate multi-threading strategies
    # sort!(transformer)

    Δt = Base.time() - t₀

    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", nblocks = L, Δt)

    return transformer
end

const GenericTransformerData{T, N} = Tuple{
    Matrix{T},
    Tuple{NTuple{N, Int}, Vector{Tuple{NTuple{N, Int}, Int}}},
    Tuple{NTuple{N, Int}, Vector{Tuple{NTuple{N, Int}, Int}}},
}

struct GenericTreeTransformer{T, N} <: TreeTransformer
    data::Vector{GenericTransformerData{T, N}}
end

"""
    buffersize(transformer::GenericTreeTransformer) -> Int
    buffersize(t::AbstractTensorMap, fblocks) -> Int

Compute the workspace size required to pack, recouple and unpack the largest multi-tree
block, i.e. `prod(sz_src) * (rows + cols)` where `(rows, cols) = size(U)` is the size of
the recoupling matrix. The second form computes this from the fusion-tree blocks of a
tensor instead, using the fact that the recoupling matrix is square.
"""
function buffersize(transformer::GenericTreeTransformer)
    return maximum(transformer.data; init = 0) do (U, _, (sz_src, _))
        return length(U) == 1 ? 0 : prod(sz_src) * sum(size(U))
    end
end
function buffersize(t::AbstractTensorMap, fblocks)
    return maximum(fblocks; init = 0) do src
        n = length(src)
        return n == 1 ? 0 : 2 * n * length(t[first(fusiontrees(src))...])
    end
end

function GenericTreeTransformer(transform, p, Vdst, Vsrc)
    t₀ = Base.time()
    permute(Vsrc, p) == Vdst || throw(SpaceMismatch("Incompatible spaces for permuting."))
    fusionstructure_dst = subblockstructure(Vdst)
    fusionstructure_src = subblockstructure(Vsrc)
    I = sectortype(Vsrc)
    T = sectorscalartype(I)
    N = numind(Vdst)
    N₁ = numout(Vsrc)
    N₂ = numin(Vsrc)

    fblocks = @timeit_debug GLOBAL_TIMER "bookkeeping: fusionblocks" fusionblocks(Vsrc)
    nblocks = length(fblocks)
    data = Vector{GenericTransformerData{T, N}}(undef, nblocks)

    nthreads = get_num_manipulation_threads()
    @timeit_debug GLOBAL_TIMER "symmetry: recoupling matrices" begin
        taskforeach(1:nblocks, nthreads) do i
            fs_src = fblocks[i]
            fs_dst, U = transform(fs_src)
            @timeit_debug GLOBAL_TIMER "bookkeeping: repack" begin
                sz_src, newstructs_src = repack_transformer_structure(fusionstructure_src, fusiontrees(fs_src))
                sz_dst, newstructs_dst = repack_transformer_structure(fusionstructure_dst, fusiontrees(fs_dst))
            end
            data[i] = U, (sz_dst, newstructs_dst), (sz_src, newstructs_src)

            @debug(
                lazy"Created recoupling block for uncoupled: $(fs_src.uncoupled)",
                sz = size(U), sparsity = count(!iszero, U) / length(U)
            )
        end
    end
    transformer = GenericTreeTransformer{T, N}(data)

    # sort by (approximate) weight to facilitate multi-threading strategies
    @timeit_debug GLOBAL_TIMER "bookkeeping: sort" sort!(transformer)

    Δt = Base.time() - t₀

    @debug(
        lazy"TreeTransformer for $Vsrc to $Vdst via $p",
        nblocks = length(transformer.data),
        sz_median = size(transformer.data[cld(end, 2)][1], 1),
        sz_max = size(transformer.data[1][1], 1),
        Δt
    )

    return transformer
end

function repack_transformer_structure(structures::Dictionary, trees)
    sz = structures[first(trees)][1]
    strides_offsets = map(trees) do f
        _, stride, offset = structures[f]
        return stride, offset
    end
    return sz, strides_offsets
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

function treetransposer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
    return fusiontreetransform(f) = transpose(f, p)
end
function treetransposer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
    return treetransposer(space(tdst), space(tsrc), p)
end
@cached function treetransposer(
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple
    )::treetransformertype(Vdst, Vsrc)
    fusiontreetransform(f) = transpose(f, p)
    return TreeTransformer(fusiontreetransform, p, Vdst, Vsrc)
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

# For CPU arrays the recoupling matrix can be used as is, also when the scalar types
# do not match, since Strided handles mixed-eltype mul! without the copy that
# Adapt.adapt would make (which additionally dispatches dynamically). Other storage
# types (e.g. GPU arrays) do require the conversion.
# TODO: transformers with dedicated storagetypes
_adapt_recoupling(::Type{<:Array}, U::Matrix) = StridedView(U)
_adapt_recoupling(::Type{A}, U::Matrix) where {A} = Adapt.adapt(A, StridedView(U))

function _transformer_weight((coeff, struct_dst, struct_src)::AbelianTransformerData)
    return prod(struct_dst[1])
end

# Cost model for transforming a set of subblocks with fixed uncoupled sectors:
# L x L x length(subblock) where L is the number of subblocks
# this is L input blocks each going to L output blocks of given length
# Note that it might be the case that the permutations are dominant, in which case the
# actual cost model would scale like L x length(subblock)
function _transformer_weight((mat, structs_dst, structs_src)::GenericTransformerData)
    return length(mat) * prod(structs_dst[1])
end
