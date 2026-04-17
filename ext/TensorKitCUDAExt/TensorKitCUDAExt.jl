module TensorKitCUDAExt

using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER, LinearAlgebra
using CUDA: @allowscalar
using cuTENSOR: cuTENSOR
using Strided: StridedViews
import CUDA: rand as curand, rand! as curand!, randn as curandn, randn! as curandn!

using CUDA: KernelAbstractions
using CUDA.KernelAbstractions: @kernel, @index

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
import TensorKit: randisometry, rand, randn, _set_subblock!

using TensorKit: MatrixAlgebraKit

using Random

include("cutensormap.jl")
include("truncation.jl")

function TensorKit._set_subblock!(data::TD, val) where {T, TD <: Union{<:CuMatrix{T}, <:StridedViews.StridedView{T, 4, <:CuArray{T}}}}
    @kernel function fill_subblock_kernel!(subblock, val)
        idx = @index(Global, Cartesian)
        @inbounds subblock[idx[1], idx[2], idx[2], idx[1]] = val
    end
    kernel = fill_subblock_kernel!(KernelAbstractions.get_backend(data))
    d1 = size(data, 1)
    d2 = size(data, 2)
    kernel(data, val; ndrange = (d1, d2))
    return data
end

end
