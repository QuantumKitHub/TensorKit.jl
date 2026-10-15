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
    structure_src = degeneracystructure(Vsrc).subblockstructure
    structs_dst = layout_dst isa StridedLayout ? structure_dst : Base.OneTo(length(structure_dst))
    p′ = linearize(p)
    structs_src = map(isrc -> _src_locator(layout_src, structure_src, isrc, p′), inds_src)
    work_offsets = pushfirst!(cumsum(map(s -> prod(first(s)), structure_dst)), 0)
    transformer = UniqueTreeTransformer(coeffs, structs_dst, structs_src, work_offsets)

    Δt = Base.time() - t₀
    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", conjsrc, nblocks = length(coeffs), Δt)

    return transformer
end

# locators of subblock `i` for the different `SubblockLayout`s
_dst_locator(::StridedLayout, structure, i::Int) = structure[i]
_dst_locator(::PositionalLayout, structure, i::Int) = i
function _src_locator(::StridedLayout, structure, i::Int, p::IndexTuple)
    _, strides, offset = structure[i]
    return (TupleTools.getindices(strides, p), offset)
end
_src_locator(::PositionalLayout, structure, i::Int, p::IndexTuple) = i

dstlocatortype(::StridedLayout, N) = StridedStructure{N}
dstlocatortype(::PositionalLayout, N) = Int
srclocatortype(::StridedLayout, N) = Tuple{NTuple{N, Int}, Int}
srclocatortype(::PositionalLayout, N) = Int

"""
    GenericTreeTransformer{T, U, VC, VD, VS, VO} <: TreeTransformer

Tree transformation for sectors with multiple fusion channels, where the subblocks of a
[`FusionTreeBlock`](@ref) map onto the subblocks of the transformed block through a square
recoupling matrix. Blocks that consist of a single tree are stored as the
[`UniqueTreeTransformer`](@ref) `unique`, while for the others block `b` consists of the entries
`tree_offsets[b] + 1:tree_offsets[b + 1]` of the locators `structs_dst` and `structs_src`, with
the same conventions as for `UniqueTreeTransformer`, and its recoupling matrix `U` is stored in
column-major order as `coeffs[coeff_offsets[b] + 1:coeff_offsets[b + 1]]`, where `U[j, i]` maps
source `i` onto destination `j`. The `work_offsets` count the number of destination elements
preceding every block, followed by their total, and the blocks are sorted by decreasing work.
"""
struct GenericTreeTransformer{
        T, U <: UniqueTreeTransformer{T}, VC <: AbstractVector{T}, VD <: AbstractVector, VS <: AbstractVector,
        VO <: AbstractVector{Int},
    } <: TreeTransformer
    unique::U
    coeffs::VC
    structs_dst::VD
    structs_src::VS
    tree_offsets::VO
    coeff_offsets::VO
    work_offsets::VO
end

Adapt.@adapt_structure GenericTreeTransformer

function GenericTreeTransformer(
        ::Type{A}, layout_dst::SubblockLayout, layout_src::SubblockLayout, transform, p, Vdst, Vsrc, conjsrc::Bool
    ) where {A}
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
    data = Vector{Tuple{Matrix{T}, Vector{Int}, Vector{Int}}}(undef, nblocks)
    weights = Vector{Int}(undef, nblocks)

    nthreads = get_num_manipulation_threads()
    @timeit_debug GLOBAL_TIMER "symmetry: recoupling matrices" begin
        taskforeach(1:nblocks, nthreads) do i
            fs_src = fblocks[i]
            fs_dst, U = transform(fs_src)
            LinearAlgebra.checksquare(U)
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

    N = numind(Vdst)
    p′ = linearize(p)
    unique_coeffs = T[]
    unique_dst = dstlocatortype(layout_dst, N)[]
    unique_src = srclocatortype(layout_src, N)[]
    unique_offsets = [0]
    coeffs = T[]
    structs_dst = dstlocatortype(layout_dst, N)[]
    structs_src = srclocatortype(layout_src, N)[]
    tree_offsets, coeff_offsets, work_offsets = [0], [0], [0]
    for (U, inds_dst, inds_src) in data
        if length(U) == 1
            push!(unique_coeffs, only(U))
            push!(unique_dst, _dst_locator(layout_dst, structure_dst, only(inds_dst)))
            push!(unique_src, _src_locator(layout_src, structure_src, only(inds_src), p′))
            push!(unique_offsets, last(unique_offsets) + prod(first(structure_dst[only(inds_dst)])))
        else
            append!(coeffs, U)
            append!(structs_dst, (_dst_locator(layout_dst, structure_dst, i) for i in inds_dst))
            append!(structs_src, (_src_locator(layout_src, structure_src, i, p′) for i in inds_src))
            push!(tree_offsets, length(structs_dst))
            push!(coeff_offsets, length(coeffs))
            push!(work_offsets, last(work_offsets) + size(U, 1) * prod(structure_dst[first(inds_dst)][1]))
        end
    end
    unique = UniqueTreeTransformer(unique_coeffs, unique_dst, unique_src, unique_offsets)
    transformer = GenericTreeTransformer(
        unique, coeffs, structs_dst, structs_src, tree_offsets, coeff_offsets, work_offsets
    )

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
    buffersize(transformer::GenericTreeTransformer) -> Int

Compute the workspace size required to pack, recouple and unpack the largest multi-tree
block, i.e. twice the number of destination elements of that block.
"""
buffersize(transformer::GenericTreeTransformer) = 2 * maximum(diff(transformer.work_offsets); init = 0)

function treetransformertype(::Type{A}, layout_dst, layout_src, Vdst, Vsrc) where {A}
    I = sectortype(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(I))
    N = numind(Vdst)
    D, S = dstlocatortype(layout_dst, N), srclocatortype(layout_src, N)
    if FusionStyle(I) == UniqueFusion()
        Dᵤ = layout_dst isa StridedLayout ? Vector{D} : Base.OneTo{Int}
        return UniqueTreeTransformer{T, Vector{T}, Dᵤ, Vector{S}, Vector{Int}}
    end
    U = UniqueTreeTransformer{T, Vector{T}, Vector{D}, Vector{S}, Vector{Int}}
    return GenericTreeTransformer{T, U, Vector{T}, Vector{D}, Vector{S}, Vector{Int}}
end

function TreeTransformer(
        ::Type{A}, layout_dst::SubblockLayout, layout_src::SubblockLayout, transform::Function, p,
        Vdst::HomSpace{S}, Vsrc::HomSpace{S}, conjsrc::Bool
    ) where {A, S}
    I = sectortype(Vdst)
    return FusionStyle(I) == UniqueFusion() ?
        UniqueTreeTransformer(A, layout_dst, layout_src, transform, p, Vdst, Vsrc, conjsrc) :
        GenericTreeTransformer(A, layout_dst, layout_src, transform, p, Vdst, Vsrc, conjsrc)
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
