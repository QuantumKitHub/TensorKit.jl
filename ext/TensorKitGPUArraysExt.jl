module TensorKitGPUArraysExt

using GPUArrays
using GPUArrays: @allowscalar
using GPUArrays.KernelAbstractions: @kernel, @index, @localmem, @synchronize, @Const, get_backend
using Adapt
using TensorKit.TupleTools
using Strided: StridedViews
using MatrixAlgebraKit, Adapt
using TensorKit
using TensorKit.TensorOperations: linearize, DefaultAllocator
const TO = TensorKit.TensorOperations
using TensorKit.Factorizations
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
using TensorKit: UniqueTreeTransformer, GenericTreeTransformer, TraceTransformer, @cached, TensorMapSpace, Index2Tuple, IndexTuple
using TensorKit: SubblockIterator
import TensorKit: randisometry, rand, randn, fill_braidingsubblock!, add_transform_kernel!, trace_transform_kernel!

function TensorKit.fill_braidingsubblock!(data::TD, val) where {T, TD <: Union{<:AnyGPUMatrix{T}, <:StridedViews.StridedView{T, 4, <:AnyGPUArray{T}}}}
    @kernel function fill_subblock_kernel!(subblock, val)
        idx = @index(Global, Cartesian)
        idx_val = idx[1] == idx[4] && idx[2] == idx[3] ? val : zero(val)
        @inbounds subblock[idx] = idx_val
    end
    kernel = fill_subblock_kernel!(get_backend(data))
    kernel(data, val; ndrange = size(data))
    return data
end

const GPUSectorVector{T, I} = TensorKit.SectorVector{T, I, <:AnyGPUVector{T}}

function MatrixAlgebraKit.findtruncated(
        values::GPUSectorVector, strategy::MatrixAlgebraKit.TruncationByOrder
    )
    I = sectortype(values)

    dims = similar(values, Base.promote_op(dim, I))
    for (c, v) in pairs(dims)
        fill!(v, dim(c))
    end

    isempty(parent(values)) && return similar(values, Bool)

    perm = sortperm(parent(values); strategy.by, strategy.rev)
    cumulative_dim = cumsum(Base.permute!(parent(dims), perm))

    result = similar(values, Bool)
    parent(result)[perm] .= cumulative_dim .<= strategy.howmany
    return result
end

function MatrixAlgebraKit.findtruncated(
        values::GPUSectorVector, strategy::MatrixAlgebraKit.TruncationByError
    )
    (isfinite(strategy.p) && strategy.p > 0) ||
        throw(ArgumentError(lazy"p-norm with p = $(strategy.p) is currently not supported."))
    ϵᵖmax = max(strategy.atol^strategy.p, strategy.rtol^strategy.p * norm(values, strategy.p))
    ϵᵖ = similar(values, typeof(ϵᵖmax))

    # dimensions are all 1 so no need to account for weight
    if FusionStyle(sectortype(values)) isa UniqueFusion
        parent(ϵᵖ) .= abs.(parent(values)) .^ strategy.p
    else
        for (c, v) in pairs(values)
            v′ = ϵᵖ[c]
            v′ .= abs.(v) .^ strategy.p .* dim(c)
        end
    end

    isempty(parent(values)) && return similar(values, Bool)

    perm = sortperm(parent(values); by = abs, rev = false)
    cumulative_err = cumsum(Base.permute!(parent(ϵᵖ), perm))

    result = similar(values, Bool)
    parent(result)[perm] .= cumulative_err .> ϵᵖmax
    return result
end

function MatrixAlgebraKit.findtruncated_svd(values::GPUSectorVector, strategy::S) where {S <: MatrixAlgebraKit.TruncationStrategy}
    # returning a GPUSectorVector wrecks things in truncate_{co}domain
    # because of scalar indexing
    return Adapt.adapt(Vector, MatrixAlgebraKit.findtruncated(values, strategy))
end

for strat in (:(MatrixAlgebraKit.TruncationByOrder), :(MatrixAlgebraKit.TruncationByError), :(MatrixAlgebraKit.TruncationIntersection), :(MatrixAlgebraKit.TruncationMultiplets), :(TensorKit.Factorizations.TruncationSpace))
    @eval function MatrixAlgebraKit.findtruncated_svd(values::GPUSectorVector, strategy::$strat)
        # returning a GPUSectorVector wrecks things in truncate_{co}domain
        # because of scalar indexing
        return Adapt.adapt(Vector, MatrixAlgebraKit.findtruncated(values, strategy))
    end
end

function MatrixAlgebraKit.findtruncated_svd(values::GPUSectorVector, strategy::MatrixAlgebraKit.TruncationByValue)
    atol = TensorKit.Factorizations.rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    return SectorDict(c => Adapt.adapt(Vector, MatrixAlgebraKit.findtruncated_svd(d, strategy′)) for (c, d) in pairs(values))
end

function MatrixAlgebraKit.truncation_error!(values::GPUSectorVector, ind::AbstractVector{Bool})
    for (c, ind_c) in pairs(ind)
        sector_vals = values[c]
        @. sector_vals *= !ind_c
    end
    return norm(values)
end

# project_symmetric! doesn't yet work for GPU types, so do this on the host, then copy
function TensorKit.project_symmetric_and_check(::Type{T}, ::Type{A}, data::AbstractArray, V::TensorMapSpace; tol = sqrt(eps(real(float(eltype(data)))))) where {T, A <: AnyGPUVector{T}}
    h_t = TensorKit.TensorMapWithStorage{T, Vector{T}}(undef, V)
    h_t = TensorKit.project_symmetric!(h_t, Array(data))
    # verify result
    isapprox(Array(reshape(data, dims(h_t))), convert(Array, h_t); atol = tol) ||
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))
    return TensorKit.TensorMapWithStorage{T, A}(A(h_t.data), V)
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::TensorMap{T, S, 0, 0, <:AnyGPUArray}) where {T, S}
    inds = findall(!iszero, t.data)
    return isempty(inds) ? zero(scalartype(t)) : @allowscalar @inbounds t.data[only(inds)]
end

# Device-side tree transformers
# -----------------------------
# The positions and recoupling coefficients of a `TreeTransformer` are resolved into
# sizes/strides/offsets and packed into flat device vectors.

const TreeStructure{N} = Tuple{NTuple{N, Int}, Int}

# strides of a dense array of shape `sz`
_dense_strides(size::Dims) = Base.front((1, cumprod(size)...))

function _work_offsets(work)
    offsets = cumsum(work)
    pushfirst!(offsets, 0)
    total = pop!(offsets)
    return offsets, total
end

"""
    TileShape{N}

A tile of `2^tshift` entries of each of `tmask + 1` consecutive destination trees, with power-of-two
extents in the index order of the destination. Tile entries are numbered both in destination
order and in source order, with the tree as the slowest index, and the coordinates of an entry `e`
are `(e >> shift[n]) & mask[n]` with the corresponding `shift`.
"""
struct TileShape{N}
    dshift::NTuple{N, Int32}
    sshift::NTuple{N, Int32}
    mask::NTuple{N, Int32}
    tshift::Int32
    tmask::Int32
    padshift::Int32 # shared memory has one padding entry per `2^padshift` entries
end

function TileShape(sz::NTuple{N, Int}, ntrees::Int, p::NTuple{N, Int}) where {N}
    lg = map(trailing_zeros, sz)
    dshift = ntuple(n -> sum(lg[1:(n - 1)]; init = 0), Val(N))
    # destination index `n` is source index `p[n]`
    q = invperm(p)
    spos = ntuple(k -> sum(lg[q[i]] for i in 1:(k - 1); init = 0), Val(N))
    sshift = ntuple(n -> spos[p[n]], Val(N))
    # at least one padding entry per two entries bounds the shared memory size
    padshift = N == 0 ? 1 : max(lg[1], 1)
    return TileShape{N}(
        Int32.(dshift), Int32.(sshift), Int32.(sz .- 1), Int32(sum(lg; init = 0)),
        Int32(ntrees - 1), Int32(padshift)
    )
end

"""
    TransformerBlock{N}

Descriptor for a block of a `DeviceTreeTransformer`, indexing into its flat
`coeffs`/`structs_dst`/`structs_src` vectors. A unique (one-tree) block has `rows = cols = 1`.
"""
struct TransformerBlock{N}
    sz::NTuple{N, Int32} # destination subblock size
    rows::Int32
    cols::Int32
    u_offset::Int32 # location of this block's `U` in the flattened `coeffs` vector
    dst_offset::Int32
    src_offset::Int32
    tile::TileShape{N}
end

"""
    TransformerTile{N}

Work item of a single workgroup: a tile of the destination trees `tree:(tree + tile.tmask)`
(0-based) of `block`.
"""
struct TransformerTile{N}
    block::Int32
    tree::Int32
    origin::NTuple{N, Int32} # 0-based coordinates of the first tile entry
end

struct DeviceTreeTransformer{N, VB <: AbstractVector{TransformerBlock{N}}, VT <: AbstractVector{TransformerTile{N}}, VC <: AbstractVector{<:Number}, VS <: AbstractVector{TreeStructure{N}}}
    blocks::VB
    tiles::VT
    coeffs::VC # every `U`, concatenated in column-major order
    structs_dst::VS
    structs_src::VS # with source strides permuted to the destination order
    len::Int # number of entries of every tile
end

# the type of the `DeviceTreeTransformer`s for the space `V` on storage `A`, for type-stable caches
devicetransformertype(A, ::TensorMapSpace{S, N₁, N₂}) where {S, N₁, N₂} =
    devicetransformertype(A, sectortype(S), Val(N₁ + N₂))
function devicetransformertype(::Type{A}, ::Type{I}, ::Val{N}) where {A, I, N}
    T = TensorKit.recoupling_scalartype(A, sectorscalartype(I))
    vec(::Type{X}) where {X} = Base.promote_op(similar, A, Type{X}, Dims{1})
    return DeviceTreeTransformer{N, vec(TransformerBlock{N}), vec(TransformerTile{N}), vec(T), vec(TreeStructure{N})}
end

Adapt.adapt_structure(to, t::DeviceTreeTransformer) = DeviceTreeTransformer(
    adapt(to, t.blocks), adapt(to, t.tiles), adapt(to, t.coeffs), adapt(to, t.structs_dst),
    adapt(to, t.structs_src), t.len
)

# Every block gets its own tile shape, whose extents (including the number of trees) are doubled
# greedily while the estimated cost decreases: 32-byte sectors read (contiguous along `q₁`, shared
# by all trees of the tile) and written (along index 1), one extra per contiguous run, the
# coefficients, and a fixed overhead per tile that includes its threads. The tile length is the
# one that minimizes the total cost.
const TILE_LENGTHS = (32, 64, 128, 256)
const MAX_GROUPSIZE = 128

function _tile_cost(tile, ntrees, q₁, (sz, rows, cols), len, elsize)
    sectors(n) = cld(n * elsize, 32) + 1
    ext = min.(tile, sz)
    n = prod(ext)
    reads = length(sz) == 0 ? sectors(1) : sectors(ext[q₁]) * (n ÷ ext[q₁])
    writes = length(sz) == 0 ? sectors(1) : sectors(ext[1]) * (n ÷ ext[1])
    t = min(ntrees, rows)
    ntiles = prod(cld.(sz, tile)) * cld(rows, ntrees)
    return ntiles * (64 + len / 32 + t * writes + cols * (reads + sectors(t)))
end

function _choose_tile(q₁, block, len, elsize, ::Val{N}) where {N}
    sz, rows, _ = block
    tile, ntrees = ntuple(_ -> 1, Val(N)), 1
    cost = _tile_cost(tile, ntrees, q₁, block, len, elsize)
    while prod(tile) * ntrees < len
        best = (cost, tile, ntrees)
        for d in 1:N
            tile[d] < sz[d] || continue
            tile′ = Base.setindex(tile, 2 * tile[d], d)
            c = _tile_cost(tile′, ntrees, q₁, block, len, elsize)
            c < first(best) && (best = (c, tile′, ntrees))
        end
        if ntrees < rows
            c = _tile_cost(tile, 2 * ntrees, q₁, block, len, elsize)
            c < first(best) && (best = (c, tile, 2 * ntrees))
        end
        first(best) < cost || break
        cost, tile, ntrees = best
    end
    return cost, tile, ntrees
end

# `entries` iterates `(U, structures_dst, structures_src)`, with `structures_src` in source order
function DeviceTreeTransformer(::Type{T}, entries, p::NTuple{N, Int}, elsize::Int) where {T, N}
    layouts = Tuple{NTuple{N, Int}, Int, Int, Int, Int, Int}[]
    coeffs = T[]
    structs_dst = TreeStructure{N}[]
    structs_src = TreeStructure{N}[]
    for (U, structures_dst, structures_src) in entries
        # all trees in a block share the same subblock size
        sz = first(first(structures_dst))
        push!(layouts, (sz, size(U)..., length(coeffs), length(structs_dst), length(structs_src)))
        append!(coeffs, U)
        for (_, strides, offset) in structures_dst
            push!(structs_dst, (strides, offset))
        end
        for (_, strides, offset) in structures_src
            push!(structs_src, (TupleTools.getindices(strides, p), offset))
        end
    end

    # tile shapes only depend on the size and the shape of `U` of a block
    counts = Dict{Tuple{NTuple{N, Int}, Int, Int}, Int}()
    for layout in layouts
        key = layout[1:3]
        counts[key] = get(counts, key, 0) + 1
    end
    q₁ = N == 0 ? 0 : findfirst(==(1), p)
    totalcost(len) = sum(((key, n),) -> n * first(_choose_tile(q₁, key, len, elsize, Val(N))), counts; init = 0.0)
    len = argmin(totalcost, TILE_LENGTHS)
    shapes = Dict(key => Base.tail(_choose_tile(q₁, key, len, elsize, Val(N))) for key in keys(counts))

    blocks = TransformerBlock{N}[]
    tiles = TransformerTile{N}[]
    for (b, (sz, rows, cols, u_offset, dst_offset, src_offset)) in enumerate(layouts)
        tile, ntrees = shapes[(sz, rows, cols)]
        push!(blocks, TransformerBlock{N}(sz, rows, cols, u_offset, dst_offset, src_offset, TileShape(tile, ntrees, p)))
        for i in 0:ntrees:(rows - 1), I in CartesianIndices(cld.(sz, tile))
            push!(tiles, TransformerTile{N}(b, i, (Tuple(I) .- 1) .* tile))
        end
    end
    # the work of a tile scales with the number of source trees, so schedule the heaviest first
    sort!(tiles; by = t -> blocks[t.block].cols, rev = true)
    return DeviceTreeTransformer(blocks, tiles, coeffs, structs_dst, structs_src, len)
end

function _tree_entries(transformer::UniqueTreeTransformer)
    (; structure_dst, structure_src) = transformer
    return (
        (fill(coeff, 1, 1), (structure_dst[idst],), (structure_src[isrc],))
            for (coeff, idst, isrc) in transformer.data
    )
end
function _tree_entries(transformer::GenericTreeTransformer)
    (; structure_dst, structure_src) = transformer
    return (
        (U, view(structure_dst, inds_dst), view(structure_src, inds_src))
            for (U, inds_dst, inds_src) in transformer.data
    )
end

"""
    TraceTransformerBlock{N}

Descriptor for a destination block of a `TraceTransformer`, indexing into the flat
`coeffs`/`structs_dst`/`sources` vectors of a `DeviceTraceTransformer`.
"""
struct TraceTransformerBlock{N}
    sz::NTuple{N, Int}
    dense_strides::NTuple{N, Int}
    rows::Int
    cols::Int
    u_offset::Int
    dst_offset::Int
    src_offset::Int
end

"""
    TraceSourceBlock{N, M}

Descriptor for the partial traces of a source subblock of a `TraceTransformer`, with `N` remaining
legs (permuted to the destination order) and `M` traced pairs, whose entries are found through a
single stride per pair. The partials are stored chunk-major, as `nchunks` dense arrays of shape `sz`.
"""
struct TraceSourceBlock{N, M}
    sz::NTuple{N, Int}
    dense_strides::NTuple{N, Int}
    permuted_strides::NTuple{N, Int}
    offset::Int
    sz_traced::NTuple{M, Int}
    dense_strides_traced::NTuple{M, Int}
    strides_traced::NTuple{M, Int} # strides[q₁] .+ strides[q₂]
    nremaining::Int
    ntraced::Int
    chunksize::Int
    nchunks::Int
end

struct DeviceTraceTransformer{VB <: AbstractVector{<:TraceTransformerBlock}, VO <: AbstractVector{Int}, VC <: AbstractVector{<:Number}, VD <: AbstractVector{<:Tuple{<:Tuple{Vararg{Int}}, Int}}, VS <: AbstractVector{<:TraceSourceBlock}}
    blocks::VB
    work_offsets::VO
    nwork::Int
    coeffs::VC # every `U`, concatenated in column-major order
    structs_dst::VD
    sources::VS
    partial_offsets::VO # also the work offsets of the first stage
    npartials::Int
end

# traces with fewer entries are not split, larger ones are split into at most `√ntraced` chunks,
# such that no thread in either stage sums more than `max(256, √ntraced)` entries
const MIN_TRACE_CHUNKSIZE = 256
_trace_chunksize(ntraced::Int) = max(MIN_TRACE_CHUNKSIZE, isqrt(ntraced))

function DeviceTraceTransformer(transformer::TraceTransformer{T, N₁, N₂}, p, (q₁, q₂)) where {T, N₁, N₂}
    lp = linearize(p)
    M = length(q₁)
    blocks = TraceTransformerBlock{N₁}[]
    coeffs = T[]
    structs_dst = TreeStructure{N₁}[]
    sources = TraceSourceBlock{N₁, M}[]

    (; structure_dst, structure_src) = transformer
    for (U, inds_dst, inds_src) in transformer.data
        # all destination trees in a block share the same subblock size
        size_dst = first(structure_dst[first(inds_dst)])
        dense_strides_dst = _dense_strides(size_dst)
        push!(
            blocks, TraceTransformerBlock{N₁}(
                size_dst, dense_strides_dst, size(U, 1), size(U, 2),
                length(coeffs), length(structs_dst), length(sources)
            )
        )
        append!(coeffs, U)
        for idst in inds_dst
            _, strides_dst, offset_dst = structure_dst[idst]
            push!(structs_dst, (strides_dst, offset_dst))
        end
        for isrc in inds_src
            size_src, strides_src, offset_src = structure_src[isrc]
            size_traced = TupleTools.getindices(size_src, q₁)
            strides_traced = TupleTools.getindices(strides_src, q₁) .+ TupleTools.getindices(strides_src, q₂)
            ntraced = prod(size_traced)
            chunksize = _trace_chunksize(ntraced)
            push!(
                sources, TraceSourceBlock{N₁, M}(
                    size_dst, dense_strides_dst, TupleTools.getindices(strides_src, lp), offset_src,
                    size_traced, _dense_strides(size_traced), strides_traced,
                    prod(size_dst), ntraced, chunksize, cld(ntraced, chunksize)
                )
            )
        end
    end

    work_offsets, nwork = _work_offsets(blk.rows * prod(blk.sz) for blk in blocks)
    partial_offsets, npartials = _work_offsets(src.nchunks * src.nremaining for src in sources)
    return DeviceTraceTransformer(
        blocks, work_offsets, nwork, coeffs, structs_dst, sources, partial_offsets, npartials
    )
end

# Cache the device transformer directly, constructing the cached host version only on a miss.
@cached function TensorKit.treebraider(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
        conjsrc::Bool, levels::IndexTuple
    )::devicetransformertype(A, Vdst) where {TA <: AnyGPUVector}
    host = TensorKit.treebraider(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc, levels)
    return device_transformer(A, host, linearize(p))
end

@cached function TensorKit.treetransposer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    )::devicetransformertype(A, Vdst) where {TA <: AnyGPUVector}
    host = TensorKit.treetransposer(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc)
    return device_transformer(A, host, linearize(p))
end

@cached function TensorKit.treetracer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, q::Index2Tuple
    ) where {TA <: AnyGPUVector}
    host = TensorKit.treetracer(Vector{scalartype(A)}, Vdst, Vsrc, p, q)
    return device_transformer(A, host, p, q)
end

@cached function TensorKit.treetwister(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
        conjsrc::Bool, twistmask::NTuple{N, Bool}, inv::Bool
    )::devicetransformertype(A, Vdst) where {TA <: AnyGPUVector, N}
    host = TensorKit.treetwister(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc, twistmask, inv)
    return device_transformer(A, host, linearize(p))
end

const DEFAULT_DEVICECACHE_SIZE = 1000

function __init__()
    # extensions can't modify GLOBAL_CACHES during precompilation, so register explicitly
    append!(
        TensorKit.GLOBAL_CACHES, (
            Symbol("TensorKitGPUArraysExt.GLOBAL_TREEBRAIDER_CACHE") => GLOBAL_TREEBRAIDER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_TREETRANSPOSER_CACHE") => GLOBAL_TREETRANSPOSER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_DIAGONAL_TREEBRAIDER_CACHE") => GLOBAL_DIAGONAL_TREEBRAIDER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_DIAGONAL_TREETRANSPOSER_CACHE") => GLOBAL_DIAGONAL_TREETRANSPOSER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_DIAGONAL_TREETWISTER_CACHE") => GLOBAL_DIAGONAL_TREETWISTER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_TREETRACER_CACHE") => GLOBAL_TREETRACER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_TREETWISTER_CACHE") => GLOBAL_TREETWISTER_CACHE,
        )
    )

    # device-caches should probably have smaller sizes:
    resize!(GLOBAL_TREEBRAIDER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_TREETRANSPOSER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_DIAGONAL_TREEBRAIDER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_DIAGONAL_TREETRANSPOSER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_DIAGONAL_TREETWISTER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_TREETRACER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_TREETWISTER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)

    return nothing
end

# Allocations must outlive any GPUArrays allocation-cache scope around the operation.
function device_transformer(A, transformer::Union{UniqueTreeTransformer, GenericTreeTransformer}, p)
    entries = _tree_entries(transformer)
    packed = DeviceTreeTransformer(scalartype(transformer), entries, Tuple(p), sizeof(scalartype(A)))
    return device_transformer(A, packed)
end

function device_transformer(A, packed::DeviceTreeTransformer)
    return GPUArrays.@uncached begin
        proto = A(undef, 0)
        # Metadata and coefficients have different element types; preserve each on upload.
        upload(x) = copy!(similar(proto, eltype(x), size(x)), x)
        DeviceTreeTransformer(
            upload(packed.blocks), upload(packed.tiles), upload(packed.coeffs),
            upload(packed.structs_dst), upload(packed.structs_src), packed.len
        )
    end
end

function device_transformer(A, transformer::TraceTransformer, p, q)
    packed = DeviceTraceTransformer(transformer, p, q)
    return GPUArrays.@uncached begin
        proto = A(undef, 0)
        upload(x) = copy!(similar(proto, eltype(x), size(x)), x)
        DeviceTraceTransformer(
            upload(packed.blocks), upload(packed.work_offsets), packed.nwork,
            upload(packed.coeffs), upload(packed.structs_dst), upload(packed.sources),
            upload(packed.partial_offsets), packed.npartials
        )
    end
end

# largest `i` with `offsets[i] <= w`. This corresponds to the
# block which  this kernel thread will work on. Since this is
# used inside a GPU kernel, searchsortedlast/searchsortedfirst
# won't work.
@inline function _searchblock(offsets, w)
    lo, hi = 1, length(offsets)
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        if @inbounds offsets[mid] <= w
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

# Cartesian coordinates of the `w`-th (0-based) entry of a dense subblock of shape `sz`.
@inline function _coordinates(w, size::NTuple{N, Int}, dense_strides::NTuple{N, Int}) where {N}
    return ntuple(n -> (w ÷ dense_strides[n]) % size[n], Val(N))
end

# finds the overall linear index in the output and input arrays corresponding to the **sublock**
# coordinates currently being worked on
@inline function _linear_index(coords::NTuple{N, Integer}, strides::NTuple{N, Int}, offset) where {N}
    # an explicit loop, since `sum` of an empty tuple (e.g. a full trace) does not compile for GPUs
    i = offset + 1
    for n in 1:N
        i += coords[n] * strides[n]
    end
    return i
end

# Tiled index manipulations
# -------------------------
# One workgroup of `G` threads per tile of `L` entries: the tile is gathered (recoupled) in source
# order into shared memory, then written out in destination order. The `L ÷ G` entries of a
# thread are unrolled through tuples, such that all its loads are in flight at once.
@inline _tilecoords(e, shift, mask, origin) = map((s, m, o) -> ((e >> s) & m) + o, shift, mask, origin)
@inline _isinside(coords, sz) = all(map(<, coords, sz))
@inline function _tileindex(coords, origin, shift)
    e = Int32(0)
    for n in 1:length(coords)
        e += (coords[n] - origin[n]) << shift[n]
    end
    return e
end
@inline _shmindex(e, padshift) = e + (e >> padshift) + Int32(1)

@inline _gather(src, op, coords, inside, (strides, offset)) =
    inside ? op(@inbounds src[_linear_index(coords, strides, offset)]) : zero(eltype(src))

# threads beyond the entries of a tile (smaller than the tile length) have `_treeindex > tmask`
@inline _treeindex(e, tile) = e >> tile.tshift

# destination indices of the entries of this thread, and `β` times their current values
@inline function _dstplan(dst, structs_dst, blk, job, es, β, ::Val{B}) where {B}
    tile = blk.tile
    coords = map(e -> _tilecoords(e, tile.dshift, tile.mask, job.origin), es)
    trees = map(e -> job.tree + _treeindex(e, tile), es)
    inside = map((e, c, i) -> _treeindex(e, tile) <= tile.tmask && i < blk.rows && _isinside(c, blk.sz), es, coords, trees)
    inds = map((c, i, b) -> _dstindex(structs_dst, blk, c, i, b), coords, trees, inside)
    olds = map(inside, inds) do b, i
        return B && b ? β * @inbounds(dst[i]) : zero(eltype(dst))
    end
    return inside, inds, olds
end

@inline function _dstindex(structs_dst, blk, coords, i, inside)
    inside || return 0
    strides, offset = @inbounds structs_dst[blk.dst_offset + i + Int32(1)]
    return _linear_index(coords, strides, offset)
end

@inline _coeff(coeffs, blk, i, j, inside) =
    inside ? @inbounds(coeffs[blk.u_offset + i + (j - Int32(1)) * blk.rows + Int32(1)]) : zero(eltype(coeffs))
@inline _accumulate(acc, src, op, coords, inside, struct_src, coeffs, blk, i, j) =
    acc + _coeff(coeffs, blk, i, j, inside) * _gather(src, op, coords, inside, struct_src)

@kernel unsafe_indices = true function tiled_transform_kernel!(
        dst, @Const(src), op, transformer, α, β, ::Val{L}, ::Val{G}, hasβ::Val{B}
    ) where {L, G, B}
    (; blocks, tiles, coeffs, structs_dst, structs_src) = transformer
    shm = @localmem eltype(dst) (L + L ÷ 2,)
    t = Int32(@index(Local, Linear) - 1)
    job = @inbounds tiles[@index(Group, Linear)]
    blk = @inbounds blocks[job.block]
    tile = blk.tile
    es = ntuple(r -> t + Int32((r - 1) * G), Val(L ÷ G))
    # with `β != 0`, the destination is read early so that this overlaps with the gather
    B && (plan = _dstplan(dst, structs_dst, blk, job, es, β, hasβ))

    # dst_i = β * dst_i + α * Σ_j U[i, j] * permute(op(src_j), p), gathered in source order;
    # all trees of the tile read the same source entries
    coords = map(e -> _tilecoords(e, tile.sshift, tile.mask, job.origin), es)
    local_trees = map(e -> _treeindex(e, tile), es)
    inside = map((c, r) -> r <= tile.tmask && job.tree + r < blk.rows && _isinside(c, blk.sz), coords, local_trees)
    acc = map(_ -> zero(eltype(dst)), es)
    for j in Int32(1):blk.cols
        struct_src = @inbounds structs_src[blk.src_offset + j]
        acc = map(@inline((a, c, r, b) -> _accumulate(a, src, op, c, b, struct_src, coeffs, blk, job.tree + r, j)), acc, coords, local_trees, inside)
    end
    map(acc, coords, local_trees, inside) do a, c, r, b
        e = _tileindex(c, job.origin, tile.dshift) + (r << tile.tshift)
        b && @inbounds shm[_shmindex(e, tile.padshift)] = a
        nothing
    end
    @synchronize

    # write out in destination order
    B || (plan = _dstplan(dst, structs_dst, blk, job, es, β, hasβ))
    map(es, plan...) do e, b, id, old
        b && @inbounds(dst[id] = α * shm[_shmindex(e, tile.padshift)] + old)
        nothing
    end
end

# Planar traces run in two kernels, such that the work per thread stays bounded even when a few
# destination elements receive a huge number of traced entries (e.g. a full trace):
# 1. `partial_trace_kernel!` splits the traced entries of every source subblock into `nchunks`
#    chunks of `chunksize`, with one thread summing one chunk for one remaining entry.
# 2. `trace_recouple_kernel!` has one thread per destination element, which only combines the
#    `nchunks` partials of each of the `cols` sources of its block with the coefficients in `U`.
# Since `chunksize` and `nchunks` are both about `√ntraced`, no thread in either kernel sums more
# than `max(256, √ntraced)` entries per source, and each traced entry is read only once rather
# than once per destination tree. A full trace of `n` entries uses `√n` threads in the first
# kernel and a single thread per destination tree in the second.

# `partials[w + 1]` is the sum of chunk `k` of the traced entries of source `j` at its remaining
# entry `e`; consecutive threads share `k` and have consecutive `e`.
@kernel function partial_trace_kernel!(partials, data_src, sources, partial_offsets, npartials)
    w = @index(Global, Linear) - 1
    if w < npartials
        j = _searchblock(partial_offsets, w)
        src = @inbounds sources[j]
        k, e = divrem(w - (@inbounds partial_offsets[j]), src.nremaining)
        coords = _coordinates(e, src.sz, src.dense_strides)
        i_src = _linear_index(coords, src.permuted_strides, src.offset)

        acc = zero(eltype(partials))
        start = k * src.chunksize
        @inbounds for t in start:(min(start + src.chunksize, src.ntraced) - 1)
            coords_traced = _coordinates(t, src.sz_traced, src.dense_strides_traced)
            acc += data_src[i_src + _linear_index(coords_traced, src.strides_traced, -1)]
        end
        @inbounds partials[w + 1] = acc
    end
end

# The traced entries were already summed by `partial_trace_kernel!`, so each thread only adds up
# the partials of the sources of its destination tree.
@kernel function trace_recouple_kernel!(
        data_dst, partials, blocks, work_offsets, coeffs, structs_dst, sources, partial_offsets,
        α, β, nwork
    )
    w = @index(Global, Linear) - 1
    if w < nwork
        b = _searchblock(work_offsets, w)
        blk = @inbounds blocks[b]
        local_w = w - (@inbounds work_offsets[b])
        blocksize = prod(blk.sz)
        i, e = divrem(local_w, blocksize)  # 0-based destination tree and entry
        coords = _coordinates(e, blk.sz, blk.dense_strides)

        st_dst, offs_dst = @inbounds structs_dst[blk.dst_offset + i + 1]
        i_dst = _linear_index(coords, st_dst, offs_dst)

        acc = zero(promote_type(eltype(partials), eltype(coeffs)))
        @inbounds for j in 1:blk.cols
            coeff = coeffs[blk.u_offset + 1 + i + (j - 1) * blk.rows]
            iszero(coeff) && continue
            js = blk.src_offset + j
            offset = partial_offsets[js] + e + 1
            tr = zero(eltype(partials))
            for k in 0:(sources[js].nchunks - 1)
                tr += partials[offset + k * blocksize]
            end
            acc += coeff * tr
        end
        # destinations without contributions are overwritten for `β == 0`
        @inbounds data_dst[i_dst] = iszero(β) ? α * acc : α * acc + β * data_dst[i_dst]
    end
end

# the tile lengths are branched on statically, as constructing a `Val` at runtime costs a dynamic dispatch
function _launch_tiled!(dst, src, conjsrc::Bool, transformer::DeviceTreeTransformer, α, β)
    isempty(transformer.tiles) && return nothing
    op = conjsrc ? conj : identity
    return iszero(β) ? _launch_tiled!(dst, src, op, transformer, α, β, Val(false)) :
        _launch_tiled!(dst, src, op, transformer, α, β, Val(true))
end
function _launch_tiled!(dst, src, op, transformer, α, β, hasβ::Val)
    L = transformer.len
    L == 32 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(32), hasβ)
    L == 64 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(64), hasβ)
    L == 128 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(128), hasβ)
    L == 256 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(256), hasβ)
    error("unexpected tile length $L")
end
function _launch_tiled!(dst, src, op, transformer, α, β, ::Val{L}, hasβ::Val) where {L}
    G = min(L, MAX_GROUPSIZE)
    kernel = tiled_transform_kernel!(get_backend(dst), G)
    kernel(dst, src, op, transformer, α, β, Val(L), Val(G), hasβ; ndrange = length(transformer.tiles) * G)
    return nothing
end

# we don't want/need the `StridedSubblocks` here because that information is already in the transformer
TensorKit._transform_subblocks(
    tdst::TensorMap, tsrc::TensorMap,
    transformer::Union{DeviceTreeTransformer, DeviceTraceTransformer}
) = (tdst.data, tsrc.data)

# Diagonal sources
# ----------------
# A `DiagonalTensorMap` only stores the diagonal of each of its subblocks, and an index
# manipulation maps the entry `(i, i)` of a source subblock onto the entry `(i, i)` of a destination
# subblock. Every block is therefore a one-dimensional strided copy, which the tiled kernel
# handles without needing a dense copy of the source or destination.

const GPUDiagonalTensorMap = DiagonalTensorMap{<:Any, <:Any, <:AnyGPUVector}
const GPUTensorMap = TensorMap{<:Any, <:Any, <:Any, <:Any, <:AnyGPUVector}

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

function TensorKit.add_transform_kernel!(
        dst::AnyGPUVector, src::AnyGPUVector, p, conjsrc::Bool,
        device::DeviceTreeTransformer, α, β, backend, allocator, ntasks::Int
    )
    _launch_tiled!(dst, src, conjsrc, device, α, β)
    return nothing
end

function TensorKit.trace_transform_kernel!(
        dst::AnyGPUVector, src::AnyGPUVector, p::Index2Tuple, q::Index2Tuple,
        device::DeviceTraceTransformer, α, β, backend, allocator, ntasks::Int
    )
    device.nwork == 0 && return nothing
    kernel_backend = get_backend(dst)
    cp = TO.allocator_checkpoint!(allocator)
    partials = TO.tensoralloc(similarstoragetype(src), device.npartials, Val(true), allocator)
    if device.npartials > 0
        partial_trace_kernel!(kernel_backend)(
            partials, src, device.sources, device.partial_offsets, device.npartials;
            ndrange = device.npartials
        )
    end
    trace_recouple_kernel!(kernel_backend)(
        dst, partials, device.blocks, device.work_offsets, device.coeffs, device.structs_dst,
        device.sources, device.partial_offsets, α, β, device.nwork; ndrange = device.nwork
    )
    TO.tensorfree!(partials, allocator)
    TO.allocator_reset!(allocator, cp)
    return nothing
end

# other tensor types are first converted to `TensorMap`s: tracing subblock by subblock with
# `tensortrace!` is not an option, since cuTENSOR requires aligned data for the subblock views
function TensorKit.trace_transform_kernel!(
        dst::SubblockIterator, src::SubblockIterator, p::Index2Tuple, q::Index2Tuple,
        device::DeviceTraceTransformer, α, β, backend, allocator, ntasks::Int
    )
    _tensormap(t) = t isa TensorMap ? t : copy!(similar(t, scalartype(t), space(t)), t)
    tdst, tsrc = _tensormap(dst.t), _tensormap(src.t)
    trace_transform_kernel!(tdst.data, tsrc.data, p, q, device, α, β, backend, allocator, ntasks)
    tdst === dst.t || copy!(dst.t, tdst)
    return nothing
end

end
