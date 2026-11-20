module TensorKitCUDAExt

using CUDA, CUDA.cuBLAS, CUDA.cuSOLVER, CUDA.cuRAND, LinearAlgebra
import CUDA.cuRAND: rand as curand, rand! as curand!, randn as curandn, randn! as curandn!
using Strided: StridedViews

using Adapt: Adapt

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
import TensorKit: randisometry, rand, randn, fill_braidingsubblock!

using TensorKit: MatrixAlgebraKit

using Random

include("cutensormap.jl")

function TensorKit._nrm2!(
        out::CuVector{R}, x::cuBLAS.StridedCuVecOrDenseMat{T}
    ) where {R <: Union{Float32, Float64}, T <: Union{R, Complex{R}}}
    cuBLAS.nrm2(x, out)
    return out
end

end
