module TensorKitAMDGPUExt

using AMDGPU, AMDGPU.rocBLAS, AMDGPU.rocSOLVER, LinearAlgebra
import AMDGPU: rand as rocrand, rand! as rocrand!, randn as rocrandn, randn! as rocrandn!

using TensorKit
using TensorKit.Factorizations
using Strided
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype
import TensorKit: randisometry
using Base: rand, randn


using Random

function TensorKit.Factorizations.batched_algorithm(
        alg::MatrixAlgebraKit.QRIteration, ::Type{<:ROCArray}
    )
    return MatrixAlgebraKit.QRIterationBatched(; alg.kwargs...)
end

include("roctensormap.jl")

end
