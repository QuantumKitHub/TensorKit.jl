"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.

The transformers only store how subblocks map onto each other in terms of the tokens `(i₁, i₂)` of
their fusion tree pairs in [`fusiontrees`](@ref), which index the [`StridedSubblocks`](@ref), together
with the recoupling coefficients, and are therefore independent of the sectortype once constructed. The
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

# (coefficient, destination token, source token)
const UniqueTransformerData{T} = Tuple{T, SubblockToken, SubblockToken}

"""
    UniqueTreeTransformer{T, N} <: TreeTransformer

Tree transformation for `UniqueFusion` sectors, where every source subblock maps onto a single
destination subblock with a scalar coefficient, stored as `(coeff, idst, isrc)`. The subblock
structures of the destination and source spaces are kept alongside, such that the
[`StridedSubblocks`](@ref) of both tensors can be created without further lookups.
"""
struct UniqueTreeTransformer{T, N} <: TreeTransformer
    data::Vector{UniqueTransformerData{T}}
    structure_dst::SubblockStructure{N}
    structure_src::SubblockStructure{N}
end

# (recoupling matrix, destination tokens, source tokens)
const GenericTransformerData{T} = Tuple{Matrix{T}, Vector{SubblockToken}, Vector{SubblockToken}}

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
    structure_dst::SubblockStructure{N}
    structure_src::SubblockStructure{N}
end

# Both structure arrays are retained by the transformer, even when another cache also
# holds them. Within an entry, identical source/destination arrays are counted only once.
_cache_structure_size(t) = _cache_payload_size(t.structure_src) +
    (t.structure_src === t.structure_dst ? 0 : _cache_payload_size(t.structure_dst))

MemoizationKit.cachesize(t::UniqueTreeTransformer) = sizeof(t) + _cache_payload_size(t.data) + _cache_structure_size(t)

MemoizationKit.cachesize(t::GenericTreeTransformer) = _cache_generic_size(t)

function _cache_generic_size(t)
    bytes = sizeof(t) + sizeof(t.data) + _cache_structure_size(t)
    for (U, inds_dst, inds_src) in t.data
        bytes += _cache_payload_size(U) + sizeof(inds_src) +
            (inds_src === inds_dst ? 0 : sizeof(inds_dst))
    end
    return bytes
end

function UniqueTreeTransformer(::Type{A}, transform, p, Vdst::HomSpace, Vsrc::HomSpace, conjsrc::Bool) where {A}
    t₀ = Base.time()

    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)

    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vdst)))
    data = Vector{UniqueTransformerData{T}}(undef, length(src_trees))

    @timeit_debug GLOBAL_TIMER "symmetry: tree transform" for (k, isrc) in enumerate(Dictionaries.tokens(src_trees))
        f₁, f₂ = gettokenvalue(src_trees, isrc)
        f_dst, coeff = transform(conjsrc ? (f₂, f₁) : (f₁, f₂))
        data[k] = (convert(T, coeff), pairtoken(dst_trees, f_dst), isrc)
    end

    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    transformer = UniqueTreeTransformer(data, structure_dst, structure_src)

    Δt = Base.time() - t₀
    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", conjsrc, nblocks = length(data), Δt)

    return transformer
end

function GenericTreeTransformer(::Type{A}, transform, p, Vdst::HomSpace, Vsrc::HomSpace, conjsrc::Bool) where {A}
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
                # the token of a fusion tree pair also addresses its subblock
                inds_src = map(((f₁, f₂),) -> pairtoken(src_trees, conjsrc ? (f₂, f₁) : (f₁, f₂)), fusiontrees(fs_src))
                inds_dst = map(f -> pairtoken(dst_trees, f), fusiontrees(fs_dst))
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

function treetransformertype(::Type{A}, Vdst, Vsrc) where {A}
    I = sectortype(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(I))
    N = numind(Vdst)
    return FusionStyle(I) == UniqueFusion() ? UniqueTreeTransformer{T, N} : GenericTreeTransformer{T, N}
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

@cached function treetransposer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    )::treetransformertype(A, Vdst, Vsrc) where {TA}
    p′ = conjsrc ? adjointtensorindices(Vsrc, p) : p
    fusiontreetransform(f) = transpose(f, p′)
    return TreeTransformer(A, fusiontreetransform, p, Vdst, Vsrc, conjsrc)
end


"""
    TraceTransformer{T, N₁, N₂}

Planar trace of a tensor with `N₂` indices into a tensor with `N₁` indices, stored as
`(U, inds_dst, inds_src)` where `U[j, i]` is the coefficient with which the trace of source
subblock `inds_src[i]` contributes to destination subblock `inds_dst[j]`. The data is grouped by
destination, such that every destination subblock appears in exactly one entry, where those
without contributions have no source subblocks. The subblock structures of the destination and source
spaces are kept alongside, such that the [`StridedSubblocks`](@ref) of both tensors can be
created without further lookups.
"""
struct TraceTransformer{T, N₁, N₂}
    data::Vector{GenericTransformerData{T}}
    structure_dst::SubblockStructure{N₁}
    structure_src::SubblockStructure{N₂}
end

MemoizationKit.cachesize(t::TraceTransformer) = _cache_generic_size(t)

VectorInterface.scalartype(
    ::Type{<:Union{UniqueTreeTransformer{T}, GenericTreeTransformer{T}, TraceTransformer{T}}}
) where {T} = T

function TraceTransformer(::Type{A}, Vdst::HomSpace, Vsrc::HomSpace, p::Index2Tuple, q::Index2Tuple) where {A}
    t₀ = Base.time()
    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vdst)))

    # destination tokens => (columns of the recoupling matrix, source tokens)
    groups = Dict{Vector{SubblockToken}, Tuple{Vector{Vector{T}}, Vector{SubblockToken}}}()
    function addcolumn!(inds_dst, u, isrc)
        any(!iszero, u) || return nothing
        us, inds_src = get!(() -> (Vector{T}[], SubblockToken[]), groups, inds_dst)
        push!(us, u)
        push!(inds_src, isrc)
        return nothing
    end
    @timeit_debug GLOBAL_TIMER "symmetry: trace coefficients" if FusionStyle(sectortype(Vdst)) isa UniqueFusion
        for isrc in Dictionaries.tokens(src_trees)
            f = gettokenvalue(src_trees, isrc)
            trace_sectors_match(f, q) || continue
            f′, coeff = planar_trace(f, p, q)
            addcolumn!([pairtoken(dst_trees, f′)], [convert(T, coeff)], isrc)
        end
    else
        for fs_src in fusionblocks(Vsrc)
            trace_sectors_match(fs_src, q) || continue
            fs_dst, U = planar_trace(fs_src, p, q)
            inds_dst = map(f -> pairtoken(dst_trees, f), fusiontrees(fs_dst))
            for (i, f) in enumerate(fusiontrees(fs_src))
                addcolumn!(inds_dst, convert(Vector{T}, U[:, i]), pairtoken(src_trees, f))
            end
        end
    end
    data = GenericTransformerData{T}[(reduce(hcat, us), inds_dst, inds_src) for (inds_dst, (us, inds_src)) in groups]
    # destination subblocks without contributions are only scaled
    covered = falses(length(dst_trees))
    for inds_dst in keys(groups), idst in inds_dst
        covered[pairposition(dst_trees, idst)] = true
    end
    for i in findall(!, covered)
        push!(data, (zeros(T, 1, 0), [pairtoken(dst_trees, i)], SubblockToken[]))
    end

    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    # sort by (approximate) weight to facilitate multi-threading strategies
    weights = map(data) do (U, inds_dst, _)
        return length(U) * prod(structure_dst[first(inds_dst)][1])
    end
    Base.permute!(data, sortperm(weights; rev = true))
    transformer = TraceTransformer(data, structure_dst, structure_src)

    Δt = Base.time() - t₀
    @debug("TraceTransformer for $Vsrc to $Vdst via $p and $q", nblocks = length(data), Δt)

    return transformer
end

"""
    buffersize(transformer::TraceTransformer) -> Int

Compute the workspace size required to trace, recouple and unpack the largest block with
multiple destination subblocks, i.e. `prod(sz_dst) * (rows + cols)` where `(rows, cols) = size(U)`.
"""
function buffersize(transformer::TraceTransformer)
    structure_dst = transformer.structure_dst
    return maximum(transformer.data; init = 0) do (U, inds_dst, _)
        return size(U, 1) == 1 ? 0 : prod(structure_dst[first(inds_dst)][1]) * sum(size(U))
    end
end

function tracetransformertype(::Type{A}, Vdst, Vsrc) where {A}
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vdst)))
    return TraceTransformer{T, numind(Vdst), numind(Vsrc)}
end

@cached function treetracer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, q::Index2Tuple
    )::tracetransformertype(A, Vdst, Vsrc) where {TA}
    return TraceTransformer(A, Vdst, Vsrc, p, q)
end

# a twist maps every tree pair onto itself, so this is a `UniqueTreeTransformer` for all sectors,
# where `p` only relabels the legs of a conjugated source
@cached function treetwister(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool,
        twistmask::NTuple{N, Bool}, inv::Bool
    )::UniqueTreeTransformer{recoupling_scalartype(A, sectorscalartype(sectortype(Vdst))), N} where {TA, N}
    # the twist mask is attached to the legs, so it follows the same relabeling as the levels in `treebraider`
    Vsrc′ = conjsrc ? Vsrc' : Vsrc
    twistmask′ = conjsrc ? TupleTools.getindices(twistmask, adjointtensorindices(Vsrc′, allind(Vsrc′))) : twistmask
    fusiontreetwister(f) = twist(f, twistmask′; inv)
    return UniqueTreeTransformer(A, fusiontreetwister, p, Vdst, Vsrc, conjsrc)
end

"""
    device_treebraider(A, Vdst, Vsrc, p, conjsrc, levels)
    device_treetransposer(A, Vdst, Vsrc, p, conjsrc)
    device_treetracer(A, Vdst, Vsrc, p, q)
    device_treetwister(A, Vdst, Vsrc, p, conjsrc, twistmask, inv)

Tree transformers for device storage types `A`, to which `treebraider`, `treetransposer`,
`treetracer` and `treetwister` forward. Their methods are added by package extensions. Being
separate functions, the device transformers have their own caches, which can be configured
independently of the host ones.
"""
function device_treebraider end
function device_treetransposer end
function device_treetracer end
function device_treetwister end
@doc (@doc device_treebraider) device_treetransposer
@doc (@doc device_treebraider) device_treetracer
@doc (@doc device_treebraider) device_treetwister

# Keep TensorKit's profiling categories for the phases instrumented by MemoizationKit.
const TensorKitCachedFunction = Union{
    typeof(fusiontreebraid), typeof(fusiontreetranspose),
    typeof(treebraider), typeof(treetransposer), typeof(treetracer), typeof(treetwister),
    typeof(device_treebraider), typeof(device_treetransposer), typeof(device_treetracer), typeof(device_treetwister),
    typeof(sectorstructure), typeof(degeneracystructure),
}
MemoizationKit.instrument_label(f::TensorKitCachedFunction, ::Val{:lookup}) = "bookkeeping: cache $(nameof(f))"
MemoizationKit.instrument_label(f::TensorKitCachedFunction, ::Val{:compute}) = "symmetry: compute $(nameof(f))"
