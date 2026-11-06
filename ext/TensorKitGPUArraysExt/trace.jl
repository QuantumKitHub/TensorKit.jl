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

function device_transformer(A, transformer::TraceTransformer, p, q)
    packed = DeviceTraceTransformer(transformer, p, q)
    return GPUArrays.@uncached begin
        upload = _uploader(A)
        DeviceTraceTransformer(
            upload(packed.blocks), upload(packed.work_offsets), packed.nwork,
            upload(packed.coeffs), upload(packed.structs_dst), upload(packed.sources),
            upload(packed.partial_offsets), packed.npartials
        )
    end
end

@cached function TensorKit.treetracer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, q::Index2Tuple
    ) where {TA <: AnyGPUVector}
    host = TensorKit.treetracer(Vector{scalartype(A)}, Vdst, Vsrc, p, q)
    return device_transformer(A, host, p, q)
end

TensorKit._transform_subblocks(tdst::TensorMap, tsrc::TensorMap, ::DeviceTraceTransformer) = (tdst.data, tsrc.data)

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
