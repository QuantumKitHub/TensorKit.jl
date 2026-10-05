module TensorKitGPUArraysExt

using GPUArrays
using GPUArrays: @allowscalar
using GPUArrays.KernelAbstractions: @kernel, @index, get_backend
using Adapt
using TensorKit: LRU
using TensorKit.TupleTools
using Strided: StridedViews
using MatrixAlgebraKit, Adapt
using TensorKit
using TensorKit.TensorOperations: linearize, DefaultAllocator
using TensorKit.Factorizations
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
using TensorKit: StridedSubblocks, UniqueTreeTransformer, GenericTreeTransformer
import TensorKit: randisometry, rand, randn, fill_braidingsubblock!, add_transform_kernel!

function TensorKit.fill_braidingsubblock!(data::TD, val) where {T, TD <: Union{<:AnyGPUMatrix{T}, <:StridedViews.StridedView{T, 4, <:AnyGPUArray{T}}}}
    # COV_EXCL_START
    # kernels are not reachable by coverage
    @kernel function fill_subblock_kernel!(subblock, val)
        idx = @index(Global, Cartesian)
        idx_val = idx[1] == idx[4] && idx[2] == idx[3] ? val : zero(val)
        @inbounds subblock[idx] = idx_val
    end
    # COV_EXCL_STOP
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

for strat in (:(MatrixAlgebraKit.TruncationByOrder), :(MatrixAlgebraKit.TruncationByError), :(MatrixAlgebraKit.TruncationIntersection), :(TensorKit.Factorizations.TruncationSpace))
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
# A `TreeTransformer` stores the mapping between subblock positions (plus recoupling
# coefficients) and the `StridedStructure`s of the source and destination spaces.
# We resolve the positions into sizes/strides/offsets and pack all the information on the
# CPU side into dense vectors of numbers, plus some accounting information so we know how
# to unpack in the kernel. Also, we can permute the source strides "in advance" on the CPU side.
# We also precompute the strides of the subblock each kernel index will work on,
# so that the GPU thread can recover the Cartesian coordinates it will need for input/ouput,
# and a running `work_offsets` count of destination elements, so that a kernel can run one
# thread per output element and recover which input that element belongs with.
# Some possible TODO here:
# - Try cuTILE as this is a classic tile programming problem
# - Use shared memory to coalesce the reads
# - Use a 2D grid for the Generic case

const TreeStructure{N} = Tuple{NTuple{N, Int}, Int}

"""
    UniqueTransformerBlock{T, N}

`isbits` descriptor for a subblock that is a *single* scaled permutation:
an entry of a `UniqueTreeTransformer` or a unique (one-tree) block of a
`GenericTreeTransformer`.
"""
struct UniqueTransformerBlock{T, N}
    coeff::T
    sz::NTuple{N, Int}
    dense_strides::NTuple{N, Int}
    strides_dst::NTuple{N, Int}
    offsets_dst::Int
    permuted_strides_src::NTuple{N, Int}  # source strides, permuted by `p`
    offsets_src::Int
end

# device-side simple struct that GPU kernels can use
struct DeviceUniqueTreeTransformer{VB <: AbstractVector{<:UniqueTransformerBlock}, VO <: AbstractVector{Int}}
    blocks::VB
    work_offsets::VO
    nwork::Int
end

"""
    GenericTransformerBlock{N}

Descriptor for a recoupling block of a `GenericTreeTransformer`, indexing into the
flat `coeffs`/`structs_dst`/`structs_src` vectors of a `DeviceGenericTreeTransformer`.
"""
struct GenericTransformerBlock{N}
    sz::NTuple{N, Int}
    densestrides::NTuple{N, Int}
    rows::Int
    cols::Int
    u_offset::Int # location in the flattened U vector to find this block's U
    dst_offset::Int
    src_offset::Int
end

# force all the type signatures here to make sure doing something wrong fails
# before the kernel launch. Kernel error dumps are awful and hard to interpret.
struct DeviceGenericTreeTransformer{VO <: AbstractVector{Int}, DA <: DeviceUniqueTreeTransformer{<:Any, VO}, VB <: AbstractVector{<:GenericTransformerBlock}, VC <: AbstractVector{<:Number}, VS <: AbstractVector{<:Tuple{<:Tuple{Vararg{Int}}, Int}}}
    unique_blocks::DA  # length(U) = 1 blocks, can be handled by unique kernel
    blocks::VB
    work_offsets::VO
    nwork::Int
    coeffs::VC  # every `U`, concatenated in column-major order
    structs_dst::VS
    structs_src::VS
end

# strides of a dense array of shape `sz`
_dense_strides(size::Dims) = (1, Base.front(cumprod(size))...)

# `permute(Vsrc, p) == Vdst` is enforced when the transformer is built, so the permuted
# source shape always matches `sz_dst` and the two views share Cartesian inds.
function _unique_block(
        coeff::T, (size_dst, strides_dst, offsets_dst), (_, strides_src, offsets_src), p
    ) where {T}
    return UniqueTransformerBlock{T, length(size_dst)}(
        coeff, size_dst, _dense_strides(size_dst), strides_dst, offsets_dst,
        TupleTools.getindices(strides_src, p), offsets_src
    )
end

function _work_offsets(work)
    offsets = cumsum(work)
    pushfirst!(offsets, 0)
    total = pop!(offsets)
    return offsets, total
end

function DeviceUniqueTreeTransformer(transformer::UniqueTreeTransformer{T, N}, p) where {T, N}
    (; structure_dst, structure_src) = transformer
    blocks = UniqueTransformerBlock{T, N}[
        _unique_block(coeff, structure_dst[idst], structure_src[isrc], p)
            for (coeff, idst, isrc) in transformer.data
    ]
    work_offsets, nwork = _work_offsets(prod(blk.sz) for blk in blocks)
    return DeviceUniqueTreeTransformer(blocks, work_offsets, nwork)
end

function DeviceGenericTreeTransformer(
        transformer::GenericTreeTransformer{T, N}, p
    ) where {T, N}
    unique_blocks = UniqueTransformerBlock{T, N}[]
    blocks = GenericTransformerBlock{N}[]
    coeffs = T[]
    structs_dst = TreeStructure{N}[]
    structs_src = TreeStructure{N}[]

    (; structure_dst, structure_src) = transformer
    for (U, inds_dst, inds_src) in transformer.data
        if length(U) == 1 # same as the unique (Abelian) case
            push!(
                unique_blocks, _unique_block(
                    only(U), structure_dst[only(inds_dst)], structure_src[only(inds_src)], p
                )
            )
        else
            # all trees in a block share the same subblock size
            size_dst = first(structure_dst[first(inds_dst)])
            push!(
                blocks, GenericTransformerBlock{N}(
                    size_dst, _dense_strides(size_dst), size(U, 1), size(U, 2),
                    length(coeffs), length(structs_dst), length(structs_src)
                )
            )
            append!(coeffs, U)
            for idst in inds_dst
                _, strides_dst, offset_dst = structure_dst[idst]
                push!(structs_dst, (strides_dst, offset_dst))
            end
            for isrc in inds_src
                _, strides_src, offset_src = structure_src[isrc]
                push!(structs_src, (TupleTools.getindices(strides_src, p), offset_src))
            end
        end
    end

    unique_offsets, unique_nwork = _work_offsets(prod(blk.sz) for blk in unique_blocks)
    work_offsets, nwork = _work_offsets(blk.rows * prod(blk.sz) for blk in blocks)
    return DeviceGenericTreeTransformer(
        DeviceUniqueTreeTransformer(unique_blocks, unique_offsets, unique_nwork),
        blocks, work_offsets, nwork, coeffs, structs_dst, structs_src
    )
end

"""
    StorageAdaptor(proto)

`Adapt` adaptor moving arrays onto the same device and array type as `proto`, preserving
their element type. For `proto::CuVector{Float64}` and `array::Vector{Int}`,
the call `adapt(typeof(proto), array)` would force-convert the element type `Int`
to `Float64`, while `adapt(StoreAdaptor(proto), array)` does not.
"""
struct StorageAdaptor{A <: AbstractArray}
    proto::A
end
function Adapt.adapt_storage(a::StorageAdaptor, x::AbstractArray)
    dst = similar(a.proto, eltype(x), size(x))
    isempty(x) && return dst
    return copy!(dst, x)
end

function Adapt.adapt_structure(to, t::DeviceUniqueTreeTransformer)
    return DeviceUniqueTreeTransformer(
        Adapt.adapt(to, t.blocks), Adapt.adapt(to, t.work_offsets), t.nwork
    )
end

function Adapt.adapt_structure(to, t::DeviceGenericTreeTransformer)
    return DeviceGenericTreeTransformer(
        Adapt.adapt(to, t.unique_blocks), Adapt.adapt(to, t.blocks),
        Adapt.adapt(to, t.work_offsets), t.nwork, Adapt.adapt(to, t.coeffs),
        Adapt.adapt(to, t.structs_dst), Adapt.adapt(to, t.structs_src)
    )
end

# Copying a transformer to GPU is more expensive than running it, so we cache the device
# copy in a global LRU cache, registered in `TensorKit.GLOBAL_CACHES` so that
# `empty_globalcaches!` also frees the device memory. The key is:
# - transformer which is an immutable struct. It's hashed and compared by the identity of
#   its fields, which avoids walking every recoupling matrix on every lookup.
#   Holding it in the key also keeps it alive, so its `objectid` cannot be
#   reused by a different transformer while the entry is cached.
# - the storage type
# - `p`, which is baked into the permuted source strides.
# TODO: should this live in the main package?
const DEVICE_TRANSFORMER_CACHE = LRU{Any, Any}(; maxsize = TensorKit.DEFAULT_GLOBALCACHE_SIZE[])

function __init__()
    push!(TensorKit.GLOBAL_CACHES, :DEVICE_TRANSFORMER_CACHE => DEVICE_TRANSFORMER_CACHE)
    return nothing
end

# We have this complicated setup because a naive `adapt` doesn't work.
# Rather we copy everything to GPU-native arrays and have kernels that can work
# with that.
function device_transformer(proto::AbstractArray, transformer, p)
    key = (transformer, typeof(proto), p)
    return get!(DEVICE_TRANSFORMER_CACHE, key) do
        # be careful about the lifetime of these, since they live in a global cache and
        # thus persist beyond the call
        GPUArrays.@uncached Adapt.adapt(
            StorageAdaptor(proto), _device_transformer(transformer, p)
        )
    end
end

_device_transformer(t::UniqueTreeTransformer, p) = DeviceUniqueTreeTransformer(t, p)
_device_transformer(t::GenericTreeTransformer, p) = DeviceGenericTreeTransformer(t, p)

# COV_EXCL_START
# kernels are not reachable by coverage

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
@inline function _linear_index(coords::NTuple{N, Int}, strides::NTuple{N, Int}, offset) where {N}
    return offset + sum(ntuple(n -> coords[n] * strides[n], Val(N))) + 1
end

# One thread per destination element in `data_dst`. `op` is `identity` or `conj`, and is
# applied to the source data only (not to the coefficients).
@kernel function unique_batched_permute_kernel!(
        data_dst, data_src, op, blocks, work_offsets, α, β, nwork, ::Val{N}
    ) where {N}
    w = @index(Global, Linear) - 1
    if w < nwork
        b = _searchblock(work_offsets, w)
        blk = @inbounds blocks[b]
        coords = _coordinates(w - (@inbounds work_offsets[b]), blk.sz, blk.dense_strides)
        i_dst = _linear_index(coords, blk.strides_dst, blk.offsets_dst)
        i_src = _linear_index(coords, blk.permuted_strides_src, blk.offsets_src)
        @inbounds data_dst[i_dst] = α * blk.coeff * op(data_src[i_src]) + β * data_dst[i_dst]
    end
end

# COV_EXCL_STOP

# One thread per destination element in `data_dst`. This makes much better use of the
# GPU "massive parallelism" as compared to the one-thread-per-subtransformer approach.
# It also more evenly divides the work among threads so the work profile is less
# jagged. Unlike the CPU implementation, there is no extract → recouple → insert process:
# BLAS is not generally reachable from inside a kernel, and fusing the recoupling into
# the strided gather lets us remove the buffer entirely.
# TODO: what about symmetries like SU(3), where the column by column approach is not
# optimal?
@kernel function generic_batched_permute_kernel!(
        data_dst, data_src, op, blocks, work_offsets, coeffs, structs_dst, structs_src,
        α, β, nwork, ::Val{N}
    ) where {N}
    w = @index(Global, Linear) - 1
    if w < nwork
        # bookkeeping to figure out where to read from and write to
        b = _searchblock(work_offsets, w)
        blk = @inbounds blocks[b]
        local_w = w - (@inbounds work_offsets[b])
        blocksize = prod(blk.sz)
        i = local_w ÷ blocksize  # 0-based destination tree
        coords = _coordinates(local_w % blocksize, blk.sz, blk.densestrides)

        st_dst, offs_dst = @inbounds structs_dst[blk.dst_offset + i + 1]
        i_dst = _linear_index(coords, st_dst, offs_dst)

        # dst_i = β * dst_i + α * Σ_j U[i, j] * permute(src_j, p): each output tree is a
        # linear combination of the input trees weighted by the recoupling coefficients.
        # The permutation of src_j was already done by permuting its strides before the
        # kernel launched.
        acc = zero(promote_type(eltype(data_src), eltype(coeffs)))
        @inbounds for j in 1:blk.cols
            # TODO is there a more efficient way to do this read?
            coeff = coeffs[blk.u_offset + 1 + i + (j - 1) * blk.rows]
            iszero(coeff) && continue
            pst_src, offs_src = structs_src[blk.src_offset + j]
            acc += coeff * op(data_src[_linear_index(coords, pst_src, offs_src)])
        end
        @inbounds data_dst[i_dst] = α * acc + β * data_dst[i_dst]
    end
end

function _launch_unique!(data_dst, data_src, op, transformer, α, β, ::Val{N}) where {N}
    nwork = transformer.nwork
    nwork == 0 && return nothing
    unique_batched_permute_kernel!(get_backend(data_dst))(
        data_dst, data_src, op, transformer.blocks, transformer.work_offsets, α, β, nwork,
        Val(N); ndrange = nwork
    )
    return nothing
end

function _launch_generic!(data_dst, data_src, op, transformer, α, β, ::Val{N}) where {N}
    nwork = transformer.nwork
    nwork == 0 && return nothing
    generic_batched_permute_kernel!(get_backend(data_dst))(
        data_dst, data_src, op, transformer.blocks, transformer.work_offsets,
        transformer.coeffs, transformer.structs_dst, transformer.structs_src, α, β, nwork,
        Val(N); ndrange = nwork
    )
    return nothing
end

const GPUStridedSubblocks = StridedSubblocks{<:AnyGPUArray}

function TensorKit.add_transform_kernel!(
        dst::GPUStridedSubblocks, src::GPUStridedSubblocks, p, conjsrc::Bool,
        transformer::UniqueTreeTransformer{T, N}, α, β, backend, allocator, ntasks::Int
    ) where {T, N}
    # GPU-side object to hold the treetransformer information
    device = device_transformer(dst.data, transformer, linearize(p))
    op = conjsrc ? conj : identity
    _launch_unique!(dst.data, src.data, op, device, α, β, Val(N))
    return nothing
end

function TensorKit.add_transform_kernel!(
        dst::GPUStridedSubblocks, src::GPUStridedSubblocks, p, conjsrc::Bool,
        transformer::GenericTreeTransformer{T, N}, α, β, backend, allocator, ntasks::Int
    ) where {T, N}
    # GPU-side object to hold the treetransformer information
    device = device_transformer(dst.data, transformer, linearize(p))
    op = conjsrc ? conj : identity
    # one-tree blocks are a scaled permutation, which the unique kernel already handles; the
    # two kernels touch disjoint subblocks so the launch order does not matter
    _launch_unique!(dst.data, src.data, op, device.unique_blocks, α, β, Val(N))
    _launch_generic!(dst.data, src.data, op, device, α, β, Val(N))
    return nothing
end

end
