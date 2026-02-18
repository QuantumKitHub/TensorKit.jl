module TensorKitCUDAExt

using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER, LinearAlgebra
using CUDA: @allowscalar
using cuTENSOR: cuTENSOR
import CUDA: rand as curand, rand! as curand!, randn as curandn, randn! as curandn!

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype, project_symmetric_and_check
import TensorKit: randisometry, rand, randn, _copyto!, _add_general_kernel_nonthreaded!, blocktype

using TensorKit: MatrixAlgebraKit

using Random

include("cutensormap.jl")
include("truncation.jl")
include("auxiliary.jl")

end
