module TensorKitCUDAExt

using CUDA, CUDA.CUBLAS, CUDA.CUSOLVER, LinearAlgebra
using CUDA: @allowscalar
using cuTENSOR: cuTENSOR
import CUDA: rand as curand, rand! as curand!, randn as curandn, randn! as curandn!

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype
import TensorKit: randisometry

using TensorKit.MatrixAlgebraKit

using Random

include("cutensormap.jl")

# TODO
# add VectorInterface extensions for proper CUDA promotion
function TensorKit.VectorInterface.promote_add(TA::Type{<:CUDA.StridedCuMatrix{Tx}}, TB::Type{<:CUDA.StridedCuMatrix{Ty}}, α::Tα = TensorKit.VectorInterface.One(), β::Tβ = TensorKit.VectorInterface.One()) where {Tx, Ty, Tα, Tβ}
    return Base.promote_op(add, Tx, Ty, Tα, Tβ)
end

end
