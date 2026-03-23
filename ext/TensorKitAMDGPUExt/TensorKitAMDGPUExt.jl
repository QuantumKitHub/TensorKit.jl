module TensorKitAMDGPUExt

using AMDGPU, AMDGPU.rocBLAS, AMDGPU.rocSOLVER, LinearAlgebra
using AMDGPU: @allowscalar
import AMDGPU: rand as rocrand, rand! as rocrand!, randn as rocrandn, randn! as rocrandn!

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using MatrixAlgebraKit: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
import TensorKit: randisometry
using Base: rand, randn

using TensorKit: MatrixAlgebraKit

using Random

include("roctensormap.jl")

end
