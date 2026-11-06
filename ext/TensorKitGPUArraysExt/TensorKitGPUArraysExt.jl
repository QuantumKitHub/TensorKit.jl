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

const GPUDiagonalTensorMap = DiagonalTensorMap{<:Any, <:Any, <:AnyGPUVector}
const GPUTensorMap = TensorMap{<:Any, <:Any, <:Any, <:Any, <:AnyGPUVector}

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

include("truncation.jl")
include("utility.jl")
include("indexmanipulations.jl")
include("trace.jl")
include("diagonal.jl")

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

end
