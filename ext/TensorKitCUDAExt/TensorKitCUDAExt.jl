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

# for ambiguity
function Base.convert(A::Type{CuArray}, f::TensorKit.FusionTree{I, 0}) where {I}
    return convert(A, TensorKit.fusiontensor(unit(I), unit(I), unit(I)))[1, 1, :]
end
function Base.convert(A::Type{CuArray}, f::TensorKit.FusionTree{I, 1}) where {I}
    c = f.coupled
    if f.isdual[1]
        sqrtdc = TensorKit.sqrtdim(c)
        Zcbartranspose = sqrtdc * convert(A, TensorKit.fusiontensor(dual(c), c, unit(c)))[:, :, 1, 1]
        X = conj!(Zcbartranspose) # we want Zcbar^†
    else
        X = convert(A, TensorKit.fusiontensor(c, unit(c), c))[:, 1, :, 1, 1]
    end
    return X
end
# needed because the Int eltype isn't supported by CuTENSOR
function Base.convert(A::Type{CuArray}, f::TensorKit.FusionTree{I, 2}) where {I}
    a, b = f.uncoupled
    isduala, isdualb = f.isdual
    c = f.coupled
    μ = (TensorKit.FusionStyle(I) isa TensorKit.GenericFusion) ? f.vertices[1] : 1
    C = convert(A, TensorKit.fusiontensor(a, b, c))[:, :, :, μ]
    X = C
    fX = reinterpret(Float64, X)
    if isduala
        Za = convert(A, TensorKit.FusionTree((a,), a, (isduala,), ()))
        # reinterpret all these as Float64 since cuTENSOR does not support Int64
        fZa = reinterpret(Float64, Za)
        @tensor fX[a′, b, c] := fZa[a′, a] * fX[a, b, c]
    end
    if isdualb
        Zb = convert(A, TensorKit.FusionTree((b,), b, (isdualb,), ()))
        fZb = reinterpret(Float64, Zb)
        @tensor fX[a, b′, c] := fZb[b′, b] * fX[a, b, c]
    end
    return X
end

function Base.convert(A::Type{CuArray}, f::TensorKit.FusionTree{I, N}) where {I, N}
    tailout = (f.innerlines[1], TensorKit.TupleTools.tail2(f.uncoupled)...)
    isdualout = (false, TensorKit.TupleTools.tail2(f.isdual)...)
    ftail = TensorKit.FusionTree(tailout, f.coupled, isdualout, Base.tail(f.innerlines), Base.tail(f.vertices))
    Ctail = convert(A, ftail)
    f₁ = TensorKit.FusionTree(
        (f.uncoupled[1], f.uncoupled[2]), f.innerlines[1],
        (f.isdual[1], f.isdual[2]), (), (f.vertices[1],)
    )
    C1 = convert(A, f₁)
    dtail = size(Ctail)
    d1 = size(C1)
    X = similar(C1, (d1[1], d1[2], Base.tail(dtail)...))
    trivialtuple = ntuple(identity, Val(N))
    # reinterpret all these as Float64 since cuTENSOR does not support Int64
    fX = reinterpret(Float64, X)
    fC1 = reinterpret(Float64, C1)
    fCtail = reinterpret(Float64, Ctail)
    TensorKit.TensorOperations.tensorcontract!(
        fX,
        fC1, ((1, 2), (3,)), false,
        fCtail, ((1,), Base.tail(trivialtuple)), false,
        ((trivialtuple..., N + 1), ())
    )
    return X
end
# TODO
# add VectorInterface extensions for proper CUDA promotion
function TensorKit.VectorInterface.promote_add(TA::Type{<:CUDA.StridedCuMatrix{Tx}}, TB::Type{<:CUDA.StridedCuMatrix{Ty}}, α::Tα = TensorKit.VectorInterface.One(), β::Tβ = TensorKit.VectorInterface.One()) where {Tx, Ty, Tα, Tβ}
    return Base.promote_op(add, Tx, Ty, Tα, Tβ)
end

end
