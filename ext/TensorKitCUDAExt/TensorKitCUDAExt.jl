module TensorKitCUDAExt

using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER, LinearAlgebra
using CUDA: @allowscalar
using cuTENSOR: cuTENSOR
using Strided: StridedViews
import CUDA: rand as curand, rand! as curand!, randn as curandn, randn! as curandn!
using CUDA.KernelAbstractions: @kernel, @index, get_backend

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
import TensorKit: randisometry, rand, randn, fill_braidingsubblock!

using TensorKit: MatrixAlgebraKit

using Random

include("cutensormap.jl")
include("truncation.jl")

function TensorKit.fill_braidingsubblock!(data::TD, val) where {T, TD <: Union{<:CuMatrix{T}, <:StridedViews.StridedView{T, 4, <:CuArray{T}}}}
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

end
