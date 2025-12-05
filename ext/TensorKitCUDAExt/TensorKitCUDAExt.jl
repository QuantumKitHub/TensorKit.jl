module TensorKitCUDAExt

using CUDA, CUDA.CUBLAS, LinearAlgebra
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

const CuDiagonalTensorMap{T, S} = DiagonalTensorMap{T, S, CuVector{T, CUDA.DeviceMemory}}

"""
    CuDiagonalTensorMap{T}(undef, domain::S) where {T,S<:IndexSpace}
    # expert mode: select storage type `A`
    DiagonalTensorMap{T,S,A}(undef, domain::S) where {T,S<:IndexSpace,A<:DenseVector{T}}

Construct a `DiagonalTensorMap` with uninitialized data.
"""
function CuDiagonalTensorMap{T}(::UndefInitializer, V::TensorMapSpace) where {T}
    (numin(V) == numout(V) == 1 && domain(V) == codomain(V)) ||
        throw(ArgumentError("DiagonalTensorMap requires a space with equal domain and codomain and 2 indices"))
    return CuDiagonalTensorMap{T}(undef, domain(V))
end
function CuDiagonalTensorMap{T}(::UndefInitializer, V::ProductSpace) where {T}
    length(V) == 1 ||
        throw(ArgumentError("DiagonalTensorMap requires `numin(d) == numout(d) == 1`"))
    return CuDiagonalTensorMap{T}(undef, only(V))
end
function CuDiagonalTensorMap{T}(::UndefInitializer, V::S) where {T, S <: IndexSpace}
    return CuDiagonalTensorMap{T, S}(undef, V)
end
CuDiagonalTensorMap(::UndefInitializer, V::IndexSpace) = CuDiagonalTensorMap{Float64}(undef, V)

function CuDiagonalTensorMap(data::CuVector{T}, V::S) where {T, S}
    return CuDiagonalTensorMap{T, S}(data, V)
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(svd_full!), t::CuDiagonalTensorMap, alg::DiagonalAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = CuDiagonalTensorMap{real(scalartype(t))}(undef, V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(svd_vals!), t::CuTensorMap, alg::AbstractAlgorithm)
    V_cod = infimum(fuse(codomain(t)), fuse(domain(t)))
    return CuDiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(svd_compact!), t::CuTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, codomain(t) ← V_cod)
    S = CuDiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eigh_full!), t::CuTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = CuDiagonalTensorMap{T}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eig_full!), t::CuTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = CuDiagonalTensorMap{Tc}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eigh_vals!), t::CuTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    return D = CuDiagonalTensorMap{Tc}(undef, V_D)
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eig_vals!), t::CuTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    return D = CuDiagonalTensorMap{Tc}(undef, V_D)
end


# TODO
# add VectorInterface extensions for proper CUDA promotion
function TensorKit.VectorInterface.promote_add(TA::Type{<:CUDA.StridedCuMatrix{Tx}}, TB::Type{<:CUDA.StridedCuMatrix{Ty}}, α::Tα = TensorKit.VectorInterface.One(), β::Tβ = TensorKit.VectorInterface.One()) where {Tx, Ty, Tα, Tβ}
    return Base.promote_op(add, Tx, Ty, Tα, Tβ)
end

end
