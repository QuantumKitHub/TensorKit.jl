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
using TensorKit: @cached, TensorMapSpace, blockstructure, compose
import TensorKit: randisometry, rand, randn, fill_braidingsubblock!

using TensorKit: MatrixAlgebraKit

using Random

include("cutensormap.jl")
include("mul.jl")

function __init__()
    # extensions can't modify GLOBAL_CACHES during precompilation, so register explicitly
    push!(
        TensorKit.GLOBAL_CACHES,
        Symbol("TensorKitCUDAExt.GLOBAL_GROUPEDGEMMPLAN_CACHE") => GLOBAL_GROUPEDGEMMPLAN_CACHE
    )
    return nothing
end

end
