"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.

The transformers only store how subblocks map onto each other, together with the recoupling
coefficients, and are therefore independent of the sectortype once constructed. The
transformation is that of `permutedims(conjsrc ? conj(tsrc) : tsrc, p)` where `p` indexes the legs
of `tsrc` itself; when `conjsrc` is `true` the fusion trees that are transformed are those of
`adjoint(space(tsrc))`, which read the subblocks of `tsrc` with the tree pair swapped.
"""
abstract type TreeTransformer end

# `StridedSubblocks` report their storage as the `StridedView` parent type, which is `Memory` there
@static if isdefined(Core, :Memory)
    const CPUStorage{T} = Union{Array{T}, Memory{T}}
else
    const CPUStorage{T} = Array{T}
end

"""
    recoupling_scalartype(A::Type{<:AbstractVector}, Tₛ::Type{<:Number}) -> Type{<:Number}

Scalar type used to store the recoupling coefficients with sector scalar type `Tₛ` in the
transformers for destination tensors with storagetype `A`. For storage with BLAS scalars, this is
the precision of the storage, where real coefficients are kept real also for complex storage, such
that they can be applied to the real and imaginary parts at once.
"""
function recoupling_scalartype(::Type{A}, ::Type{Tₛ}) where {A, Tₛ <: Number}
    T = scalartype(A)
    T <: BlasFloat || return Tₛ
    return Tₛ <: Real ? real(T) : complex(T)
end

"""
    SubblockLayout

How a transformer locates the subblocks of a tensor. [`StridedLayout`](@ref) addresses the flat
data vector of a `TensorMap` through the `(sizes, strides, offset)` of each subblock, while
[`PositionalLayout`](@ref) indexes the [`subblocks`](@ref) of any other `AbstractTensorMap` by their
position in the canonical order of [`fusiontrees`](@ref).
"""
abstract type SubblockLayout end
struct StridedLayout <: SubblockLayout end
struct PositionalLayout <: SubblockLayout end

subblocklayout(t::AbstractTensorMap) = subblocklayout(typeof(t))
subblocklayout(::Type{<:AbstractTensorMap}) = PositionalLayout()
subblocklayout(::Type{<:TensorMap}) = StridedLayout()

# the object through which the kernels access the subblocks, as located by `subblocklayout`
subblockdata(t::TensorMap) = t.data
subblockdata(t::AbstractTensorMap) = subblocks(t)

"""
    UniqueTreeTransformer{T, VC, VD, VS, VO} <: TreeTransformer

Tree transformation for `UniqueFusion` sectors, where every source subblock maps onto a single
destination subblock with a scalar coefficient. Entry `i` describes the transformation
`dst_i = β * dst_i + α * coeffs[i] * permute(src_i, p)` through the locators `structs_dst[i]` and
`structs_src[i]`, whose type depends on the [`SubblockLayout`](@ref) of either tensor:

- `StridedLayout`: `structs_dst[i]` is the `(sizes, strides, offset)` of the destination subblock,
  and `structs_src[i]` the `(strides, offset)` of the source subblock with its strides already
  permuted by `p`, such that both share the destination sizes.
- `PositionalLayout`: the subblock positions, and `p` is applied when transforming.

The `n + 1` `work_offsets` are the number of destination elements preceding every entry, followed by
their total, such that the work can be distributed per element. All fields are vectors of `isbits`
elements, such that the same structure can be moved to other devices with `Adapt.adapt`.
"""
struct UniqueTreeTransformer{
        T, VC <: AbstractVector{T}, VD <: AbstractVector, VS <: AbstractVector, VO <: AbstractVector{Int},
    } <: TreeTransformer
    coeffs::VC
    structs_dst::VD
    structs_src::VS
    work_offsets::VO
end

Adapt.@adapt_structure UniqueTreeTransformer

function UniqueTreeTransformer(coeffs, structs_dst, structs_src)
    work_offsets = pushfirst!(cumsum(map(s -> prod(first(s)), structs_dst)), 0)
    return UniqueTreeTransformer(coeffs, structs_dst, structs_src, work_offsets)
end

# (recoupling matrix, destination positions, source positions)
const GenericTransformerData{T} = Tuple{Matrix{T}, Vector{Int}, Vector{Int}}

"""
    GenericTreeTransformer{T, N} <: TreeTransformer

Tree transformation for sectors with multiple fusion channels, where the subblocks of a
[`FusionTreeBlock`](@ref) map onto the subblocks of the transformed block through a recoupling
matrix, stored as `(U, inds_dst, inds_src)`, where `U[j, i]` maps source `i` onto destination `j`.
The subblock structures of the destination and source spaces are kept alongside, such that the
[`StridedSubblocks`](@ref) of both tensors can be created without further lookups.
"""
struct GenericTreeTransformer{T, N} <: TreeTransformer
    data::Vector{GenericTransformerData{T}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
end

function UniqueTreeTransformer(
        ::Type{A}, layout_dst::SubblockLayout, layout_src::SubblockLayout, transform, p, Vdst, Vsrc, conjsrc::Bool
    ) where {A}
    t₀ = Base.time()

    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)

    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vdst)))
    # the transformation is a bijection between the trees, so we use the canonical destination order
    coeffs = Vector{T}(undef, length(src_trees))
    inds_src = Vector{Int}(undef, length(src_trees))

    @timeit_debug GLOBAL_TIMER "symmetry: tree transform" for (isrc, (f₁, f₂)) in enumerate(src_trees)
        f_dst, coeff = transform(conjsrc ? (f₂, f₁) : (f₁, f₂))
        _, (_, idst) = gettoken(dst_trees, f_dst)
        coeffs[idst] = convert(T, coeff)
        inds_src[idst] = isrc
    end

    structure_dst = degeneracystructure(Vdst).subblockstructure
    structs_dst = _dst_locators(layout_dst, structure_dst)
    structs_src = _src_locators(layout_src, Vsrc, inds_src, linearize(p))
    work_offsets = pushfirst!(cumsum(map(s -> prod(first(s)), structure_dst)), 0)
    transformer = UniqueTreeTransformer(coeffs, structs_dst, structs_src, work_offsets)

    Δt = Base.time() - t₀
    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", conjsrc, nblocks = length(coeffs), Δt)

    return transformer
end

# the destination locators of a unique transformer are in canonical order
_dst_locators(::StridedLayout, structure::Vector{<:StridedStructure}) = structure
_dst_locators(::PositionalLayout, structure::Vector{<:StridedStructure}) = Base.OneTo(length(structure))

_src_locators(::PositionalLayout, Vsrc, inds_src::Vector{Int}, p) = inds_src
function _src_locators(::StridedLayout, Vsrc, inds_src::Vector{Int}, p::IndexTuple{N}) where {N}
    structure_src = degeneracystructure(Vsrc).subblockstructure
    return map(inds_src) do isrc
        _, strides_src, offset_src = structure_src[isrc]
        return (TupleTools.getindices(strides_src, p), offset_src)
    end::Vector{Tuple{NTuple{N, Int}, Int}}
end

dstlocatortype(::StridedLayout, N) = Vector{StridedStructure{N}}
dstlocatortype(::PositionalLayout, N) = Base.OneTo{Int}
srclocatortype(::StridedLayout, N) = Vector{Tuple{NTuple{N, Int}, Int}}
srclocatortype(::PositionalLayout, N) = Vector{Int}

function GenericTreeTransformer(::Type{A}, transform, p, Vdst, Vsrc, conjsrc::Bool) where {A}
    t₀ = Base.time()
    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)
    # the fusion blocks that are transformed are those of the adjoint space for a conjugated source
    Vsrc′ = conjsrc ? Vsrc' : Vsrc
    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vsrc)))

    fblocks = @timeit_debug GLOBAL_TIMER "bookkeeping: fusionblocks" fusionblocks(Vsrc′)
    nblocks = length(fblocks)
    data = Vector{GenericTransformerData{T}}(undef, nblocks)
    weights = Vector{Int}(undef, nblocks)

    nthreads = get_num_manipulation_threads()
    @timeit_debug GLOBAL_TIMER "symmetry: recoupling matrices" begin
        taskforeach(1:nblocks, nthreads) do i
            fs_src = fblocks[i]
            fs_dst, U = transform(fs_src)
            @timeit_debug GLOBAL_TIMER "bookkeeping: subblock positions" begin
                # the token into the fusion tree `Indices` is the subblock position
                inds_src = map(fusiontrees(fs_src)) do (f₁, f₂)
                    _, (_, isrc) = gettoken(src_trees, conjsrc ? (f₂, f₁) : (f₁, f₂))
                    return isrc
                end
                inds_dst = map(fusiontrees(fs_dst)) do f
                    _, (_, idst) = gettoken(dst_trees, f)
                    return idst
                end
            end
            data[i] = (convert(Matrix{T}, U), inds_dst, inds_src)
            # cost model: L input blocks each going to L output blocks of a given length
            weights[i] = length(U) * prod(structure_dst[first(inds_dst)][1])

            @debug(
                lazy"Created recoupling block for uncoupled: $(fs_src.uncoupled)",
                sz = size(U), sparsity = count(!iszero, U) / length(U)
            )
        end
    end

    # sort by (approximate) weight to facilitate multi-threading strategies
    @timeit_debug GLOBAL_TIMER "bookkeeping: sort" Base.permute!(data, sortperm(weights; rev = true))
    transformer = GenericTreeTransformer{T, numind(Vdst)}(data, structure_dst, structure_src)

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
buffersize(::UniqueTreeTransformer) = 0
function buffersize(transformer::GenericTreeTransformer)
    structure_src = transformer.structure_src
    return maximum(transformer.data; init = 0) do (U, _, inds_src)
        return length(U) == 1 ? 0 : prod(structure_src[first(inds_src)][1]) * sum(size(U))
    end
end

function treetransformertype(::Type{A}, layout_dst, layout_src, Vdst, Vsrc) where {A}
    I = sectortype(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(I))
    N = numind(Vdst)
    FusionStyle(I) == UniqueFusion() || return GenericTreeTransformer{T, N}
    return UniqueTreeTransformer{
        T, Vector{T}, dstlocatortype(layout_dst, N), srclocatortype(layout_src, N), Vector{Int},
    }
end

function TreeTransformer(
        ::Type{A}, layout_dst::SubblockLayout, layout_src::SubblockLayout, transform::Function, p,
        Vdst::HomSpace{S}, Vsrc::HomSpace{S}, conjsrc::Bool
    ) where {A, S}
    I = sectortype(Vdst)
    return FusionStyle(I) == UniqueFusion() ?
        UniqueTreeTransformer(A, layout_dst, layout_src, transform, p, Vdst, Vsrc, conjsrc) :
        GenericTreeTransformer(A, transform, p, Vdst, Vsrc, conjsrc)
end

# braid is special because it has levels
function treebraider(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )
    return treebraider(
        storagetype(tdst), subblocklayout(tdst), subblocklayout(tsrc), space(tdst), space(tsrc), p, conjsrc, levels
    )
end
@cached function treebraider(
        A::Type{TA}, layout_dst::SubblockLayout, layout_src::SubblockLayout,
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )::treetransformertype(A, layout_dst, layout_src, Vdst, Vsrc) where {TA}
    Vsrc′, p′ = conjsrc ? (Vsrc', adjointtensorindices(Vsrc, p)) : (Vsrc, p)
    # levels are attached to the legs, so they follow the same relabeling as the permutation
    levels′ = conjsrc ? TupleTools.getindices(levels, adjointtensorindices(Vsrc′, allind(Vsrc′))) : levels
    levels″ = (TupleTools.getindices(levels′, codomainind(Vsrc′)), TupleTools.getindices(levels′, domainind(Vsrc′)))
    fusiontreebraider(f) = braid(f, p′, levels″)
    return TreeTransformer(A, layout_dst, layout_src, fusiontreebraider, p, Vdst, Vsrc, conjsrc)
end

function treetransposer(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, conjsrc::Bool)
    return treetransposer(
        storagetype(tdst), subblocklayout(tdst), subblocklayout(tsrc), space(tdst), space(tsrc), p, conjsrc
    )
end
@cached function treetransposer(
        A::Type{TA}, layout_dst::SubblockLayout, layout_src::SubblockLayout,
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    )::treetransformertype(A, layout_dst, layout_src, Vdst, Vsrc) where {TA}
    p′ = conjsrc ? adjointtensorindices(Vsrc, p) : p
    fusiontreetransform(f) = transpose(f, p′)
    return TreeTransformer(A, layout_dst, layout_src, fusiontreetransform, p, Vdst, Vsrc, conjsrc)
end

# default cachestyle is GlobalLRUCache
