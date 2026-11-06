# Tiled index manipulations
# -------------------------
# One workgroup of `G` threads per tile of `L` entries: the tile is gathered (recoupled) in source
# order into shared memory, then written out in destination order. The `L ÷ G` entries of a
# thread are unrolled through tuples, such that all its loads are in flight at once.

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
end

"""
    TransformerTile{N}

Work item of a single workgroup: a tile of the destination tree `tree` (0-based) of `block`.
"""
struct TransformerTile{N}
    block::Int32
    tree::Int32
    origin::NTuple{N, Int32} # 0-based coordinates of the first tile entry
end

"""
    TileShape{N}

A tile with power-of-two extents `2 .^ shift`, in the index order of the destination.
Tile entries are numbered both in destination order and in source order, and the coordinates
of an entry `e` are `(e >> shift[n]) & mask[n]` with the corresponding `shift`.
"""
struct TileShape{N}
    dshift::NTuple{N, Int32}
    sshift::NTuple{N, Int32}
    mask::NTuple{N, Int32}
    len::Int32
    padshift::Int32 # shared memory has one padding entry per `2^padshift` entries
end

function TileShape(sz::NTuple{N, Int}, p::NTuple{N, Int}) where {N}
    lg = map(trailing_zeros, sz)
    dshift = ntuple(n -> sum(lg[1:(n - 1)]; init = 0), Val(N))
    # destination index `n` is source index `p[n]`
    q = invperm(p)
    spos = ntuple(k -> sum(lg[q[i]] for i in 1:(k - 1); init = 0), Val(N))
    sshift = ntuple(n -> spos[p[n]], Val(N))
    padshift = N == 0 ? 0 : lg[1]
    return TileShape{N}(Int32.(dshift), Int32.(sshift), Int32.(sz .- 1), Int32(prod(sz)), Int32(padshift))
end

struct DeviceTreeTransformer{N, VB <: AbstractVector{TransformerBlock{N}}, VT <: AbstractVector{TransformerTile{N}}, VC <: AbstractVector{<:Number}, VS <: AbstractVector{TreeStructure{N}}}
    blocks::VB
    tiles::VT
    coeffs::VC # every `U`, concatenated in column-major order
    structs_dst::VS
    structs_src::VS # with source strides permuted to the destination order
    tile::TileShape{N}
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
    adapt(to, t.structs_src), t.tile
)

# Tile extents are doubled greedily while the estimated cost decreases: 32-byte sectors read
# (contiguous along `q₁`) and written (along index 1), one extra per contiguous run, a fixed
# overhead per tile and the idle threads of partially filled tiles.
const MAX_TILE_LENGTH = 256
const MIN_TILE_LENGTH = 32 # tiles should at least fill a warp
const MAX_GROUPSIZE = 128

function _tile_cost(tile, q₁, sizes, elsize)
    sectors(len) = cld(len * elsize, 32) + 1
    cost = 0.0
    for sz in sizes
        ext = min.(tile, sz)
        n = prod(ext)
        reads = sectors(ext[q₁]) * (n ÷ ext[q₁])
        writes = sectors(ext[1]) * (n ÷ ext[1])
        cost += prod(cld.(sz, tile)) * (64 + reads + writes + prod(tile) / 32)
    end
    return cost
end

function _choose_tile(q₁, sizes, elsize, ::Val{N}) where {N}
    N == 0 && return ()
    # at least two entries along the first index bounds the shared memory padding
    tile = ntuple(n -> n == 1 ? 2 : 1, Val(N))
    cost = _tile_cost(tile, q₁, sizes, elsize)
    while prod(tile) < MAX_TILE_LENGTH
        candidates = ntuple(d -> Base.setindex(tile, 2 * tile[d], d), Val(N))
        costs = map(t -> _tile_cost(t, q₁, sizes, elsize), candidates)
        newcost, d = findmin(costs)
        newcost < cost || prod(tile) < MIN_TILE_LENGTH || break
        tile, cost = candidates[d], newcost
    end
    return tile
end

# `entries` iterates `(U, structures_dst, structures_src)`, with `structures_src` in source order
function DeviceTreeTransformer(::Type{T}, entries, p::NTuple{N, Int}, elsize::Int) where {T, N}
    blocks = TransformerBlock{N}[]
    coeffs = T[]
    structs_dst = TreeStructure{N}[]
    structs_src = TreeStructure{N}[]
    sizes = NTuple{N, Int}[] # one per source tile that is read
    for (U, structures_dst, structures_src) in entries
        # all trees in a block share the same subblock size
        sz = first(first(structures_dst))
        push!(blocks, TransformerBlock{N}(sz, size(U)..., length(coeffs), length(structs_dst), length(structs_src)))
        append!(coeffs, U)
        for (_, strides, offset) in structures_dst
            push!(structs_dst, (strides, offset))
        end
        for (_, strides, offset) in structures_src
            push!(structs_src, (TupleTools.getindices(strides, p), offset))
        end
        append!(sizes, Iterators.repeated(sz, length(U)))
    end
    q₁ = N == 0 ? 0 : findfirst(==(1), p)
    tile = _choose_tile(q₁, sizes, elsize, Val(N))
    tiles = TransformerTile{N}[]
    for (b, blk) in enumerate(blocks), i in 0:(blk.rows - 1), I in CartesianIndices(cld.(blk.sz, tile))
        push!(tiles, TransformerTile{N}(b, i, (Tuple(I) .- 1) .* tile))
    end
    # the work of a tile scales with the number of source trees, so schedule the heaviest first
    sort!(tiles; by = t -> blocks[t.block].cols, rev = true)
    return DeviceTreeTransformer(blocks, tiles, coeffs, structs_dst, structs_src, TileShape(tile, p))
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

function device_transformer(A, transformer::Union{UniqueTreeTransformer, GenericTreeTransformer}, p)
    entries = _tree_entries(transformer)
    packed = DeviceTreeTransformer(scalartype(transformer), entries, Tuple(p), sizeof(scalartype(A)))
    return device_transformer(A, packed)
end

function device_transformer(A, packed::DeviceTreeTransformer)
    return GPUArrays.@uncached begin
        upload = _uploader(A)
        DeviceTreeTransformer(
            upload(packed.blocks), upload(packed.tiles), upload(packed.coeffs),
            upload(packed.structs_dst), upload(packed.structs_src), packed.tile
        )
    end
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

@cached function TensorKit.treetwister(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple,
        conjsrc::Bool, twistmask::NTuple{N, Bool}, inv::Bool
    )::devicetransformertype(A, Vdst) where {TA <: AnyGPUVector, N}
    host = TensorKit.treetwister(Vector{scalartype(A)}, Vdst, Vsrc, p, conjsrc, twistmask, inv)
    return device_transformer(A, host, linearize(p))
end

# we don't want/need the `StridedSubblocks` here because that information is already in the transformer
TensorKit._transform_subblocks(tdst::TensorMap, tsrc::TensorMap, ::DeviceTreeTransformer) = (tdst.data, tsrc.data)

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

# destination indices of the entries of this thread, and `β` times their current values
@inline function _dstplan(dst, (strides, offset), es, tile, origin, sz, β, ::Val{B}) where {B}
    coords = map(e -> _tilecoords(e, tile.dshift, tile.mask, origin), es)
    inside = map(c -> _isinside(c, sz), coords)
    inds = map(c -> _linear_index(c, strides, offset), coords)
    olds = map(inside, inds) do b, i
        return B && b ? β * @inbounds(dst[i]) : zero(eltype(dst))
    end
    return inside, inds, olds
end

@kernel unsafe_indices = true function tiled_transform_kernel!(
        dst, @Const(src), op, transformer, α, β, ::Val{L}, ::Val{G}, hasβ::Val{B}
    ) where {L, G, B}
    (; blocks, tiles, coeffs, structs_dst, structs_src, tile) = transformer
    shm = @localmem eltype(dst) (L + L ÷ 2,)
    t = Int32(@index(Local, Linear) - 1)
    job = @inbounds tiles[@index(Group, Linear)]
    blk = @inbounds blocks[job.block]
    i = job.tree
    es = ntuple(r -> t + Int32((r - 1) * G), Val(L ÷ G))
    struct_dst = @inbounds structs_dst[blk.dst_offset + i + Int32(1)]
    # with `β != 0`, the destination is read early so that this overlaps with the gather
    B && (plan = _dstplan(dst, struct_dst, es, tile, job.origin, blk.sz, β, hasβ))

    # dst_i = β * dst_i + α * Σ_j U[i, j] * permute(op(src_j), p), gathered in source order
    coords = map(e -> _tilecoords(e, tile.sshift, tile.mask, job.origin), es)
    inside = map(c -> _isinside(c, blk.sz), coords)
    acc = map(_ -> zero(eltype(dst)), es)
    for j in Int32(1):blk.cols
        coeff = @inbounds coeffs[blk.u_offset + i + (j - Int32(1)) * blk.rows + Int32(1)]
        struct_src = @inbounds structs_src[blk.src_offset + j]
        acc = map((a, c, b) -> a + coeff * _gather(src, op, c, b, struct_src), acc, coords, inside)
    end
    map(acc, coords) do a, c
        @inbounds shm[_shmindex(_tileindex(c, job.origin, tile.dshift), tile.padshift)] = a
        nothing
    end
    @synchronize

    # write out in destination order
    B || (plan = _dstplan(dst, struct_dst, es, tile, job.origin, blk.sz, β, hasβ))
    map(es, plan...) do e, b, id, old
        b && @inbounds(dst[id] = α * shm[_shmindex(e, tile.padshift)] + old)
        nothing
    end
end

# all tile lengths are powers of two in `MIN_TILE_LENGTH:MAX_TILE_LENGTH`, or 1 without indices,
# and are branched on statically, as constructing a `Val` at runtime costs a dynamic dispatch
function _launch_tiled!(dst, src, conjsrc::Bool, transformer::DeviceTreeTransformer, α, β)
    isempty(transformer.tiles) && return nothing
    op = conjsrc ? conj : identity
    return iszero(β) ? _launch_tiled!(dst, src, op, transformer, α, β, Val(false)) :
        _launch_tiled!(dst, src, op, transformer, α, β, Val(true))
end
function _launch_tiled!(dst, src, op, transformer, α, β, hasβ::Val)
    L = transformer.tile.len
    L == 32 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(32), hasβ)
    L == 64 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(64), hasβ)
    L == 128 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(128), hasβ)
    L == 256 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(256), hasβ)
    L == 1 && return _launch_tiled!(dst, src, op, transformer, α, β, Val(1), hasβ)
    error("unexpected tile length $L")
end
function _launch_tiled!(dst, src, op, transformer, α, β, ::Val{L}, hasβ::Val) where {L}
    G = min(L, MAX_GROUPSIZE)
    kernel = tiled_transform_kernel!(get_backend(dst), G)
    kernel(dst, src, op, transformer, α, β, Val(L), Val(G), hasβ; ndrange = length(transformer.tiles) * G)
    return nothing
end

function TensorKit.add_transform_kernel!(
        dst::AnyGPUVector, src::AnyGPUVector, p, conjsrc::Bool,
        device::DeviceTreeTransformer, α, β, backend, allocator, ntasks::Int
    )
    _launch_tiled!(dst, src, conjsrc, device, α, β)
    return nothing
end
