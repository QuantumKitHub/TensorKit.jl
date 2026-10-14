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
using TensorKit: UniqueTreeTransformer, GenericTreeTransformer, StridedLayout, @cached, TensorMapSpace, Index2Tuple, IndexTuple
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
# The tree transformers with `StridedLayout`s only consist of vectors of `isbits` elements, with the
# source strides permuted "in advance" on the CPU side, and are moved to the device as is. The
# running `work_offsets` count of destination elements allows a kernel to run one thread per output
# element and recover which input that element belongs with.
# Some possible TODO here:
# - Try cuTILE as this is a classic tile programming problem
# - Use shared memory to coalesce the reads
# - Use a 2D grid for the Generic case

# strides of a dense array of shape `sz`
_dense_strides(size::Dims) = (1, Base.front(cumprod(size))...)

# Cache the device transformer directly, constructing the cached host version only on a miss.
# Other layouts are transformed on the host, through the positional subblocks.
@cached function TensorKit.treebraider(
        A::Type{TA}, layout_dst::StridedLayout, layout_src::StridedLayout,
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    ) where {TA <: AnyGPUVector}
    host = TensorKit.treebraider(Vector{scalartype(A)}, layout_dst, layout_src, Vdst, Vsrc, p, conjsrc, levels)
    return device_transformer(A, host)
end

@cached function TensorKit.treetransposer(
        A::Type{TA}, layout_dst::StridedLayout, layout_src::StridedLayout,
        Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    ) where {TA <: AnyGPUVector}
    host = TensorKit.treetransposer(Vector{scalartype(A)}, layout_dst, layout_src, Vdst, Vsrc, p, conjsrc)
    return device_transformer(A, host)
end

const DEFAULT_DEVICECACHE_SIZE = 1000

function __init__()
    # extensions can't modify GLOBAL_CACHES during precompilation, so register explicitly
    append!(
        TensorKit.GLOBAL_CACHES, (
            Symbol("TensorKitGPUArraysExt.GLOBAL_TREEBRAIDER_CACHE") => GLOBAL_TREEBRAIDER_CACHE,
            Symbol("TensorKitGPUArraysExt.GLOBAL_TREETRANSPOSER_CACHE") => GLOBAL_TREETRANSPOSER_CACHE,
        )
    )

    # device-caches should probably have smaller sizes:
    resize!(GLOBAL_TREEBRAIDER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)
    resize!(GLOBAL_TREETRANSPOSER_CACHE; maxsize = DEFAULT_DEVICECACHE_SIZE)

    return nothing
end

# Adaptor that uploads arrays to the storage of `proto`, preserving their element types, since
# metadata and coefficients have different element types.
struct DeviceUpload{P <: AnyGPUVector}
    proto::P
end
Adapt.adapt_storage(to::DeviceUpload, x::AbstractArray) = copy!(similar(to.proto, eltype(x), size(x)), x)

# Allocations must outlive any GPUArrays allocation-cache scope around the operation.
device_transformer(A, transformer) = GPUArrays.@uncached adapt(DeviceUpload(A(undef, 0)), transformer)

# COV_EXCL_START
# kernels are not reachable by coverage

# largest `i` with `offsets[i] <= w`. For `0 <= w < offsets[end]`, this corresponds to the
# (nonempty) block which this kernel thread will work on. Since this is
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
@inline function _coordinates(w, size::NTuple{N, Int}, dense_strides::NTuple{N, Int} = _dense_strides(size)) where {N}
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
        data_dst, data_src, op, coeffs, structs_dst, structs_src, work_offsets, α, β, ::Val{N}
    ) where {N}
    # w = @index(Global, Linear) - 1 does not work due to JLArrays bug
    w = @index(Global, Linear)
    w -= 1
    if w < @inbounds work_offsets[end]
        b = _searchblock(work_offsets, w)
        sz, strides_dst, offset_dst = @inbounds structs_dst[b]
        strides_src, offset_src = @inbounds structs_src[b]
        coords = _coordinates(w - (@inbounds work_offsets[b]), sz)
        i_dst = _linear_index(coords, strides_dst, offset_dst)
        i_src = _linear_index(coords, strides_src, offset_src)
        @inbounds data_dst[i_dst] = α * coeffs[b] * op(data_src[i_src]) + β * data_dst[i_dst]
    end
end

# One thread per destination element in `data_dst`. This makes much better use of the
# GPU "massive parallelism" as compared to the one-thread-per-subtransformer approach.
# It also more evenly divides the work among threads so the work profile is less
# jagged. Unlike the CPU implementation, there is no extract → recouple → insert process:
# BLAS is not generally reachable from inside a kernel, and fusing the recoupling into
# the strided gather lets us remove the buffer entirely.
# TODO: what about symmetries like SU(3), where the column by column approach is not
# optimal?
@kernel function generic_batched_permute_kernel!(
        data_dst, data_src, op, coeffs, structs_dst, structs_src, tree_offsets, coeff_offsets, work_offsets,
        α, β, ::Val{N}
    ) where {N}
    # w = @index(Global, Linear) - 1 does not work due to JLArrays bug
    w = @index(Global, Linear)
    w -= 1
    if w < @inbounds work_offsets[end]
        # bookkeeping to figure out where to read from and write to
        b = _searchblock(work_offsets, w)
        t₀ = @inbounds tree_offsets[b]
        L = (@inbounds tree_offsets[b + 1]) - t₀
        u₀ = @inbounds coeff_offsets[b]
        sz = first(@inbounds structs_dst[t₀ + 1])
        blocksize = prod(sz)
        local_w = w - (@inbounds work_offsets[b])
        i = local_w ÷ blocksize  # 0-based destination tree
        coords = _coordinates(local_w % blocksize, sz)

        _, st_dst, offs_dst = @inbounds structs_dst[t₀ + i + 1]
        i_dst = _linear_index(coords, st_dst, offs_dst)

        # dst_i = β * dst_i + α * Σ_j U[i, j] * permute(src_j, p): each output tree is a
        # linear combination of the input trees weighted by the recoupling coefficients.
        # The permutation of src_j was already done by permuting its strides before the
        # kernel launched.
        acc = zero(promote_type(eltype(data_src), eltype(coeffs)))
        @inbounds for j in 1:L
            # TODO is there a more efficient way to do this read?
            coeff = coeffs[u₀ + 1 + i + (j - 1) * L]
            iszero(coeff) && continue
            pst_src, offs_src = structs_src[t₀ + j]
            acc += coeff * op(data_src[_linear_index(coords, pst_src, offs_src)])
        end
        @inbounds data_dst[i_dst] = α * acc + β * data_dst[i_dst]
    end
end

# COV_EXCL_STOP

function _launch_unique!(data_dst, data_src, op, transformer::UniqueTreeTransformer, α, β, ::Val{N}) where {N}
    (; coeffs, structs_dst, structs_src, work_offsets) = transformer
    # the subblocks are disjoint, so the total work is bounded by the destination length
    ndrange = isempty(coeffs) ? 0 : length(data_dst)
    ndrange == 0 && return nothing
    unique_batched_permute_kernel!(get_backend(data_dst))(
        data_dst, data_src, op, coeffs, structs_dst, structs_src, work_offsets, α, β,
        Val(N); ndrange
    )
    return nothing
end

function _launch_generic!(data_dst, data_src, op, transformer::GenericTreeTransformer, α, β, ::Val{N}) where {N}
    (; coeffs, structs_dst, structs_src, tree_offsets, coeff_offsets, work_offsets) = transformer
    # the subblocks are disjoint, so the total work is bounded by the destination length
    ndrange = length(tree_offsets) == 1 ? 0 : length(data_dst)
    ndrange == 0 && return nothing
    generic_batched_permute_kernel!(get_backend(data_dst))(
        data_dst, data_src, op, coeffs, structs_dst, structs_src, tree_offsets, coeff_offsets, work_offsets,
        α, β, Val(N); ndrange
    )
    return nothing
end

function TensorKit.add_transform_kernel!(
        dst::AnyGPUVector, src::AnyGPUVector, p, conjsrc::Bool,
        device::UniqueTreeTransformer, α, β, backend, allocator, ntasks::Int
    )
    N = length(linearize(p))
    op = conjsrc ? conj : identity
    _launch_unique!(dst, src, op, device, α, β, Val(N))
    return nothing
end

function TensorKit.add_transform_kernel!(
        dst::AnyGPUVector, src::AnyGPUVector, p, conjsrc::Bool,
        device::GenericTreeTransformer, α, β, backend, allocator, ntasks::Int
    )
    N = length(linearize(p))
    op = conjsrc ? conj : identity
    # one-tree blocks are a scaled permutation, which the unique kernel already handles; the
    # two kernels touch disjoint subblocks so the launch order does not matter
    _launch_unique!(dst, src, op, device.unique, α, β, Val(N))
    _launch_generic!(dst, src, op, device, α, β, Val(N))
    return nothing
end

end
