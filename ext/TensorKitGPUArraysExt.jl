module TensorKitGPUArraysExt

using GPUArrays
using GPUArrays: @allowscalar
using GPUArrays.KernelAbstractions: @kernel, @index, get_backend
using Adapt
using Strided: StridedViews
using MatrixAlgebraKit, Adapt
using TensorKit
using TensorKit.TensorOperations: linearize, DefaultAllocator
using TensorKit.Factorizations
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
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
# `AbelianTransformerData` is `isbits`, but `GenericTransformerData` isn't.
# `Matrix{T}` and two ragged `Vector`s, so we can't just call `adapt` on it.
# But we can work around this but packing all the information on the CPU side
# into dense vectors of numbers, plus some accounting information so we know how
# to unpack in the generic kernel. Also, we can permute the source strides "in
# advance" on the CPU side. We also precompute the strides of the subblock each
# kernel index will work on, so that the GPU thread can recover the Cartesian
# coordinates it will need for input/ouput, and a running `work_offsets` count
# of destination elements, so that a kernel can run one thread per output
# element and recover which subblock that element belongs with.

# Some possible TODO here:
# - Try cuTILE as this is a classic tile programming problem
# - Use shared memory to coalesce the reads
# - Use a 2D grid for the Generic case

const TreeStructure{N} = Tuple{NTuple{N, Int}, Int}

"""
    AbelianTransformerBlock{T, N}

Isbits descriptor for a subblock that is a single scaled permutation: either an entry of
an `AbelianTreeTransformer`, or a degenerate (one-tree) block of a
`GenericTreeTransformer`.
"""
struct AbelianTransformerBlock{T, N}
    coeff::T
    sz::NTuple{N, Int}
    densestrides::NTuple{N, Int}
    st_dst::NTuple{N, Int}
    offs_dst::Int
    pst_src::NTuple{N, Int}  # source strides, permuted by `p`
    offs_src::Int
end

"""
    GenericTransformerBlock{N}

Descriptor for a recoupling block of a `GenericTreeTransformer`, indexing into the
flat `coeffs`/`structs_dst`/`structs_src` vectors of a `DeviceGenericTreeTransformer`.
** All offsets are 0-based since it makes the arithmetic easier. **
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

struct DeviceAbelianTreeTransformer{VB <: AbstractVector{<:AbelianTransformerBlock}, VO <: AbstractVector{Int}}
    blocks::VB
    work_offsets::VO
    nwork::Int
end

# force all the type signatures here to make sure doing something wrong fails
# before the kernel launch. Kernel error dumps are awful and hard to interpret.
struct DeviceGenericTreeTransformer{VO <: AbstractVector{Int}, DA <: DeviceAbelianTreeTransformer{<:Any, VO}, VB <: AbstractVector{<:GenericTransformerBlock}, VC <: AbstractVector{<:Number}, VS <: AbstractVector{<:Tuple{<:Tuple{Vararg{Int}}, Int}}}
    degenerate::DA  # length(U) = 1 blocks, can be handled by Abelian kernel
    blocks::VB
    work_offsets::VO
    nwork::Int
    coeffs::VC  # every `U`, concatenated in column-major order
    structs_dst::VS
    structs_src::VS
end

# strides of a dense array of shape `sz`
_densestrides(sz::NTuple{N, Int}) where {N} = ntuple(n -> prod(sz[1:(n - 1)]; init = 1), Val(N))
_permutestrides(st::NTuple{N, Int}, p) where {N} = ntuple(n -> st[p[n]], Val(N))

# `permute(Vsrc, p) == Vdst` is enforced when the transformer is built, so the permuted
# source shape always matches `sz_dst` and the two views share Cartesian inds.
function _abelian_block(
        coeff::T, (sz_dst, st_dst, offs_dst), (_, st_src, offs_src), p
    ) where {T}
    return AbelianTransformerBlock{T, length(sz_dst)}(
        coeff, sz_dst, _densestrides(sz_dst), st_dst, offs_dst,
        _permutestrides(st_src, p), offs_src
    )
end

function _work_offsets(work)
    offsets = cumsum(work)
    pushfirst!(offsets, 0)
    total = pop!(offsets)
    return offsets, total
end

function DeviceAbelianTreeTransformer(
        transformer::TensorKit.AbelianTreeTransformer{T, N}, p
    ) where {T, N}
    blocks = AbelianTransformerBlock{T, N}[_abelian_block(entry..., p) for entry in transformer.data]
    work_offsets, nwork = _work_offsets(prod(blk.sz) for blk in blocks)
    return DeviceAbelianTreeTransformer(blocks, work_offsets, nwork)
end

function DeviceGenericTreeTransformer(
        transformer::TensorKit.GenericTreeTransformer{T, N}, p
    ) where {T, N}
    degenerate = AbelianTransformerBlock{T, N}[]
    blocks = GenericTransformerBlock{N}[]
    coeffs = T[]
    structs_dst = TreeStructure{N}[]
    structs_src = TreeStructure{N}[]

    for (U, (sz_dst, sts_dst), (sz_src, sts_src)) in transformer.data
        if length(U) == 1 # same as the Abelian case
            push!(
                degenerate, _abelian_block(
                    only(U), (sz_dst, only(sts_dst)...), (sz_src, only(sts_src)...), p
                )
            )
        else
            push!(
                blocks, GenericTransformerBlock{N}(
                    sz_dst, _densestrides(sz_dst), size(U, 1), size(U, 2),
                    length(coeffs), length(structs_dst), length(structs_src)
                )
            )
            append!(coeffs, U)
            append!(structs_dst, sts_dst)
            for (st_src, offs_src) in sts_src
                push!(structs_src, (_permutestrides(st_src, p), offs_src))
            end
        end
    end

    deg_offsets, deg_nwork = _work_offsets(prod(blk.sz) for blk in degenerate)
    work_offsets, nwork = _work_offsets(blk.rows * prod(blk.sz) for blk in blocks)
    return DeviceGenericTreeTransformer(
        DeviceAbelianTreeTransformer(degenerate, deg_offsets, deg_nwork),
        blocks, work_offsets, nwork, coeffs, structs_dst, structs_src
    )
end

"""
    StorageAdaptor(proto)

`Adapt` adaptor moving arrays onto the same device and array type as `proto`, preserving
their element type. `adapt(CuVector{Float64}, ::Vector{Int})` would force-convert the Int
to Float64, while `similar(proto, Int, n)` doesn't.
"""
struct StorageAdaptor{A <: AbstractArray}
    proto::A
end
function Adapt.adapt_storage(a::StorageAdaptor, x::AbstractArray)
    dst = similar(a.proto, eltype(x), size(x))
    isempty(x) && return dst
    return copyto!(dst, x)
end

function Adapt.adapt_structure(to, t::DeviceAbelianTreeTransformer)
    return DeviceAbelianTreeTransformer(
        Adapt.adapt(to, t.blocks), Adapt.adapt(to, t.work_offsets), t.nwork
    )
end
function Adapt.adapt_structure(to, t::DeviceGenericTreeTransformer)
    return DeviceGenericTreeTransformer(
        Adapt.adapt(to, t.degenerate), Adapt.adapt(to, t.blocks),
        Adapt.adapt(to, t.work_offsets), t.nwork, Adapt.adapt(to, t.coeffs),
        Adapt.adapt(to, t.structs_dst), Adapt.adapt(to, t.structs_src)
    )
end

# Copying a transformer to GPU is more expensive than running it, and transformers are
# themselves cached (and thus long-lived) by `treebraider`/`treetransposer`, so we cache the
# device copy for as long as the CPU original is "alive". The key is:
# - `transformer.data`
# - the storage type
# - `p`, which is baked into the permuted source strides.
# Using `objectid` avoids walking every recoupling matrix on every lookup.
# TODO: should this live in the main package?
const DEVICE_TRANSFORMER_CACHE = Dict{UInt, Tuple{WeakRef, Dict{Any, Any}}}()
const DEVICE_TRANSFORMER_LOCK = ReentrantLock()

# We have this complicated setup because a naive `adapt` doesn't work.
# Rather we copy everything to GPU-native arrays and have kernels that can work
# with that.
function device_transformer(proto::AbstractArray, transformer, p)
    key = transformer.data
    return Base.@lock DEVICE_TRANSFORMER_LOCK begin
        entry = get(DEVICE_TRANSFORMER_CACHE, objectid(key), nothing)
        if isnothing(entry) || entry[1].value !== key
            filter!(kv -> !isnothing(last(kv)[1].value), DEVICE_TRANSFORMER_CACHE)
            entry = (WeakRef(key), Dict{Any, Any}())
            DEVICE_TRANSFORMER_CACHE[objectid(key)] = entry
        end
        get!(last(entry), (typeof(proto), p)) do
            # be careful about the lifetime of these, since they live as long as their
            # "parent" on the CPU, so they can persist beyond the call
            GPUArrays.@uncached Adapt.adapt(
                StorageAdaptor(proto), _device_transformer(transformer, p)
            )
        end
    end
end

_device_transformer(t::TensorKit.AbelianTreeTransformer, p) = DeviceAbelianTreeTransformer(t, p)
_device_transformer(t::TensorKit.GenericTreeTransformer, p) = DeviceGenericTreeTransformer(t, p)

# COV_EXCL_START
# kernels are not reachable by coverage

# largest `i` with `offsets[i] <= w`. This corresponds to the
# block which  this kernel thread will work on.
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
# Computed once per thread and then reused for every strided view of that subblock.
# This avoids `StridedView` redoing these divisions on every single element access.
# Integer division on GPU is usually pretty slow.
@inline function _coordinates(w, sz::NTuple{N, Int}, densestrides::NTuple{N, Int}) where {N}
    return ntuple(n -> (w ÷ densestrides[n]) % sz[n], Val(N))
end

# finds the overall offset in the output and input arrays corresponding to the **sublock**
# coordinates currently being worked on
@inline function _offset(coords::NTuple{N, Int}, st::NTuple{N, Int}, offs) where {N}
    return offs + sum(ntuple(n -> coords[n] * st[n], Val(N))) + 1
end

# One thread per destination element in `data_dst`.
@kernel function abelian_batched_permute!(
        data_dst, data_src, blocks, work_offsets, α, β, nwork, ::Val{N}
    ) where {N}
    w = @index(Global, Linear) - 1
    if w < nwork
        b = _searchblock(work_offsets, w)
        blk = @inbounds blocks[b]
        coords = _coordinates(w - (@inbounds work_offsets[b]), blk.sz, blk.densestrides)
        i_dst = _offset(coords, blk.st_dst, blk.offs_dst)
        i_src = _offset(coords, blk.pst_src, blk.offs_src)
        @inbounds data_dst[i_dst] = α * blk.coeff * data_src[i_src] + β * data_dst[i_dst]
    end
end

# One thread per destination element in `data_dst`. This makes much better use of the
# GPU "massive parallelism" as compared to the one-thread-per-subtransformer approach.
# It also more evenly divides the work among threads so the work profile is less
# jagged. Unlike the CPU implementation, there is no extract → recouple → insert process:
# BLAS is not generally reachable from inside a kernel, and fusing the recoupling into
# the strided gather lets us remove the buffer entirely.
@kernel function generic_batched_permute!(
        data_dst, data_src, blocks, work_offsets, coeffs, structs_dst, structs_src,
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
        i_dst = _offset(coords, st_dst, offs_dst)

        # dst_i = β * dst_i + α * Σ_j U[i, j] * permute(src_j, p): each output tree is a
        # linear combination of the input trees weighted by the recoupling coefficients.
        # The permutation of src_j was already done by permuting its strides before the
        # kernel launched.
        acc = zero(promote_type(eltype(data_src), eltype(coeffs)))
        for j in 1:blk.cols
            # TODO is there a more efficient way to do this read?
            coeff = @inbounds coeffs[blk.u_offset + (j - 1) * blk.rows + i + 1]
            iszero(coeff) && continue
            pst_src, offs_src = @inbounds structs_src[blk.src_offset + j]
            acc += coeff * @inbounds data_src[_offset(coords, pst_src, offs_src)]
        end
        @inbounds data_dst[i_dst] = α * acc + β * data_dst[i_dst]
    end
end
# COV_EXCL_STOP

function _launch_abelian!(data_dst, data_src, transformer, α, β, ::Val{N}) where {N}
    nwork = transformer.nwork
    nwork == 0 && return nothing
    abelian_batched_permute!(get_backend(data_dst))(
        data_dst, data_src, transformer.blocks, transformer.work_offsets, α, β, nwork,
        Val(N); ndrange = nwork
    )
    return nothing
end

function _launch_generic!(data_dst, data_src, transformer, α, β, ::Val{N}) where {N}
    nwork = transformer.nwork
    nwork == 0 && return nothing
    generic_batched_permute!(get_backend(data_dst))(
        data_dst, data_src, transformer.blocks, transformer.work_offsets,
        transformer.coeffs, transformer.structs_dst, transformer.structs_src, α, β, nwork,
        Val(N); ndrange = nwork
    )
    return nothing
end

function TensorKit.add_transform_kernel!(
        data_dst::A, data_src::A, p, transformer::TensorKit.AbelianTreeTransformer{T, N},
        α, β, backend, allocator, scheduler
    ) where {T, N, A <: AnyGPUArray}
    # GPU-side object to hold the treetransformer information
    device = device_transformer(data_dst, transformer, linearize(p))::DeviceAbelianTreeTransformer
    _launch_abelian!(data_dst, data_src, device, α, β, Val(N))
    return nothing
end

function TensorKit.add_transform_kernel!(
        data_dst::A, data_src::A, p, transformer::TensorKit.GenericTreeTransformer{T, N},
        α, β, backend, allocator, scheduler
    ) where {T, N, A <: AnyGPUArray}
    # GPU-side object to hold the treetransformer information
    device = device_transformer(data_dst, transformer, linearize(p))::DeviceGenericTreeTransformer
    # one-tree blocks are a scaled permutation, which the Abelian kernel already handles; the
    # two kernels touch disjoint subblocks so the launch order does not matter
    _launch_abelian!(data_dst, data_src, device.degenerate, α, β, Val(N))
    _launch_generic!(data_dst, data_src, device, α, β, Val(N))
    return nothing
end

function TensorKit.TensorOperations.tensorfree!(C::AnyGPUArray, ::DefaultAllocator)
    GPUArrays.unsafe_free!(C)
    return nothing
end

end
