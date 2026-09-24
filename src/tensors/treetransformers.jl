"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.

The transformers only store how subblocks map onto each other in terms of their positions in
[`StridedSubblocks`](@ref) (the canonical order of [`fusiontrees`](@ref)), together with the recoupling
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
    T = eltype(A)
    T <: BlasFloat || return Tₛ
    return Tₛ <: Real ? real(T) : complex(T)
end

# storagetypes are vector types, so the recoupling matrices are stored as reshaped vectors
recoupling_matrixtype(::Type{<:CPUStorage}, ::Type{T}) where {T} = Matrix{T}
Base.@assume_effects :foldable recoupling_matrixtype(::Type{A}, ::Type{T}) where {A, T} =
    Core.Compiler.return_type(reshape, Tuple{similarstoragetype(A, T), Tuple{Int, Int}})

_convert_recoupling(::Type{<:CPUStorage}, ::Type{T}, U::Matrix) where {T} = convert(Matrix{T}, U)
_convert_recoupling(::Type{A}, ::Type{T}, U::Matrix) where {A, T} =
    reshape(Adapt.adapt(similarstoragetype(A, T), vec(convert(Matrix{T}, U))), size(U))

# (coefficient, destination position, source position)
const UniqueTransformerData{T} = Tuple{T, Int, Int}

"""
    UniqueTreeTransformer{T, N} <: TreeTransformer

Tree transformation for `UniqueFusion` sectors, where every source subblock maps onto a single
destination subblock with a scalar coefficient, stored as `(coeff, idst, isrc)`. The subblock
structures of the destination and source spaces are kept alongside, such that the
[`StridedSubblocks`](@ref) of both tensors can be created without further lookups.
"""
struct UniqueTreeTransformer{T, N} <: TreeTransformer
    data::Vector{UniqueTransformerData{T}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
end

"""
    RecouplingBlock{T, M}

Recoupling data of a single [`FusionTreeBlock`](@ref), mapping the subblocks at the source
positions `inds_src` onto those at the destination positions `inds_dst`. Blocks consisting of a
single tree only have a scalar coefficient `coeff`, which is kept on the host, while other blocks
have a recoupling matrix `U::M` in the storage of the destination, where `U[j, i]` maps source `i`
onto destination `j`.
"""
struct RecouplingBlock{T, M <: AbstractMatrix{T}}
    coeff::T
    U::Union{Nothing, M}
    inds_dst::Vector{Int}
    inds_src::Vector{Int}
    # `M` cannot be inferred from `U === nothing`, so the parameters are always specified
    RecouplingBlock{T, M}(coeff, U, inds_dst, inds_src) where {T, M <: AbstractMatrix{T}} =
        new{T, M}(coeff, U, inds_dst, inds_src)
end

"""
    GenericTreeTransformer{T, M, N} <: TreeTransformer

Tree transformation for sectors with multiple fusion channels, where the subblocks of a
[`FusionTreeBlock`](@ref) map onto the subblocks of the transformed block through a recoupling
matrix, stored as a [`RecouplingBlock{T, M}`](@ref RecouplingBlock). The subblock structures of the
destination and source spaces are kept alongside, such that the [`StridedSubblocks`](@ref) of both
tensors can be created without further lookups.
"""
struct GenericTreeTransformer{T, M <: AbstractMatrix{T}, N} <: TreeTransformer
    data::Vector{RecouplingBlock{T, M}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
end

function UniqueTreeTransformer(::Type{A}, transform, p, Vdst, Vsrc, conjsrc::Bool) where {A}
    t₀ = Base.time()

    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)

    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vdst)))
    data = Vector{UniqueTransformerData{T}}(undef, length(src_trees))

    @timeit_debug GLOBAL_TIMER "symmetry: tree transform" for (isrc, (f₁, f₂)) in enumerate(src_trees)
        f_dst, coeff = transform(conjsrc ? (f₂, f₁) : (f₁, f₂))
        _, (_, idst) = gettoken(dst_trees, f_dst)
        data[isrc] = (convert(T, coeff), idst, isrc)
    end

    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    transformer = UniqueTreeTransformer(data, structure_dst, structure_src)

    Δt = Base.time() - t₀
    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", conjsrc, nblocks = length(data), Δt)

    return transformer
end

function GenericTreeTransformer(::Type{A}, transform, p, Vdst, Vsrc, conjsrc::Bool) where {A}
    t₀ = Base.time()
    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)
    # the fusion blocks that are transformed are those of the adjoint space for a conjugated source
    Vsrc′ = conjsrc ? Vsrc' : Vsrc
    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vsrc)))
    M = recoupling_matrixtype(A, T)

    fblocks = @timeit_debug GLOBAL_TIMER "bookkeeping: fusionblocks" fusionblocks(Vsrc′)
    nblocks = length(fblocks)
    data = Vector{RecouplingBlock{T, M}}(undef, nblocks)
    weights = Vector{Int}(undef, nblocks)

    nthreads = get_num_manipulation_threads()
    @timeit_debug GLOBAL_TIMER "symmetry: recoupling matrices" begin
        taskforeach(1:nblocks, nthreads) do i
            fs_src = fblocks[i]
            fs_dst, U₀ = transform(fs_src)
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
            data[i] = length(U₀) == 1 ?
                RecouplingBlock{T, M}(convert(T, only(U₀)), nothing, inds_dst, inds_src) :
                RecouplingBlock{T, M}(zero(T), _convert_recoupling(A, T, U₀), inds_dst, inds_src)
            # cost model: L input blocks each going to L output blocks of a given length
            weights[i] = length(U₀) * prod(structure_dst[first(inds_dst)][1])

            @debug(
                lazy"Created recoupling block for uncoupled: $(fs_src.uncoupled)",
                sz = size(U₀), sparsity = count(!iszero, U₀) / length(U₀)
            )
        end
    end

    # sort by (approximate) weight to facilitate multi-threading strategies
    @timeit_debug GLOBAL_TIMER "bookkeeping: sort" Base.permute!(data, sortperm(weights; rev = true))
    transformer = GenericTreeTransformer{T, M, numind(Vdst)}(data, structure_dst, structure_src)

    Δt = Base.time() - t₀
    @debug(
        lazy"TreeTransformer for $Vsrc to $Vdst via $p", conjsrc,
        nblocks = nblocks,
        sz_median = nblocks > 0 ? length(data[cld(end, 2)].inds_dst) : 0,
        sz_max = nblocks > 0 ? length(data[1].inds_dst) : 0,
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
    return maximum(transformer.data; init = 0) do blk
        isnothing(blk.U) && return 0
        return prod(structure_src[first(blk.inds_src)][1]) * sum(size(blk.U))
    end
end

function treetransformertype(::Type{A}, Vdst, Vsrc) where {A}
    I = sectortype(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(I))
    N = numind(Vdst)
    FusionStyle(I) == UniqueFusion() && return UniqueTreeTransformer{T, N}
    return GenericTreeTransformer{T, recoupling_matrixtype(A, T), N}
end

function TreeTransformer(
        ::Type{A}, transform::Function, p, Vdst::HomSpace{S}, Vsrc::HomSpace{S}, conjsrc::Bool
    ) where {A, S}
    I = sectortype(Vdst)
    return FusionStyle(I) == UniqueFusion() ?
        UniqueTreeTransformer(A, transform, p, Vdst, Vsrc, conjsrc) :
        GenericTreeTransformer(A, transform, p, Vdst, Vsrc, conjsrc)
end

# braid is special because it has levels
function treebraider(
        tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )
    return treebraider(storagetype(tdst), space(tdst), space(tsrc), p, conjsrc, levels)
end
@cached function treebraider(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )::treetransformertype(A, Vdst, Vsrc) where {TA}
    Vsrc′, p′ = conjsrc ? (Vsrc', adjointtensorindices(Vsrc, p)) : (Vsrc, p)
    # levels are attached to the legs, so they follow the same relabeling as the permutation
    levels′ = conjsrc ? TupleTools.getindices(levels, adjointtensorindices(Vsrc′, allind(Vsrc′))) : levels
    levels″ = (TupleTools.getindices(levels′, codomainind(Vsrc′)), TupleTools.getindices(levels′, domainind(Vsrc′)))
    fusiontreebraider(f) = braid(f, p′, levels″)
    return TreeTransformer(A, fusiontreebraider, p, Vdst, Vsrc, conjsrc)
end

function treetransposer(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p::Index2Tuple, conjsrc::Bool)
    return treetransposer(storagetype(tdst), space(tdst), space(tsrc), p, conjsrc)
end
@cached function treetransposer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    )::treetransformertype(A, Vdst, Vsrc) where {TA}
    p′ = conjsrc ? adjointtensorindices(Vsrc, p) : p
    fusiontreetransform(f) = transpose(f, p′)
    return TreeTransformer(A, fusiontreetransform, p, Vdst, Vsrc, conjsrc)
end

# default cachestyle is GlobalLRUCache
