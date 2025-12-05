module TensorKitAMDGPUExt

using AMDGPU, AMDGPU.rocBLAS, LinearAlgebra
using AMDGPU: @allowscalar
import AMDGPU: rand as rocrand, rand! as rocrand!, randn as rocrandn, randn! as rocrandn!

using TensorKit
using TensorKit.Factorizations
using TensorKit.Strided
using TensorKit.Factorizations: AbstractAlgorithm
using TensorKit: SectorDict, tensormaptype, scalar, similarstoragetype, AdjointTensorMap, scalartype

using TensorKit.MatrixAlgebraKit

using Random

include("roctensormap.jl")

const ROCDiagonalTensorMap{T, S} = DiagonalTensorMap{T, S, ROCVector{T, AMDGPU.Mem.HIPBuffer}}

"""
    ROCDiagonalTensorMap{T}(undef, domain::S) where {T,S<:IndexSpace}
    # expert mode: select storage type `A`
    DiagonalTensorMap{T,S,A}(undef, domain::S) where {T,S<:IndexSpace,A<:DenseVector{T}}

Construct a `DiagonalTensorMap` with uninitialized data.
"""
function ROCDiagonalTensorMap{T}(::UndefInitializer, V::TensorMapSpace) where {T}
    (numin(V) == numout(V) == 1 && domain(V) == codomain(V)) ||
        throw(ArgumentError("DiagonalTensorMap requires a space with equal domain and codomain and 2 indices"))
    return ROCDiagonalTensorMap{T}(undef, domain(V))
end
function ROCDiagonalTensorMap{T}(::UndefInitializer, V::ProductSpace) where {T}
    length(V) == 1 ||
        throw(ArgumentError("DiagonalTensorMap requires `numin(d) == numout(d) == 1`"))
    return ROCDiagonalTensorMap{T}(undef, only(V))
end
function ROCDiagonalTensorMap{T}(::UndefInitializer, V::S) where {T, S <: IndexSpace}
    return ROCDiagonalTensorMap{T, S}(undef, V)
end
ROCDiagonalTensorMap(::UndefInitializer, V::IndexSpace) = ROCDiagonalTensorMap{Float64}(undef, V)

function ROCDiagonalTensorMap(data::ROCVector{T}, V::S) where {T, S}
    return ROCDiagonalTensorMap{T, S}(data, V)
end

function ROCDiagonalTensorMap(data::Vector{T}, V::S) where {T, S}
    return ROCDiagonalTensorMap{T, S}(ROCVector{T}(data), V)
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(svd_full!), t::ROCDiagonalTensorMap, alg::DiagonalAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = ROCDiagonalTensorMap{real(scalartype(t))}(undef, V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(svd_vals!), t::ROCTensorMap, alg::AbstractAlgorithm)
    V_cod = infimum(fuse(codomain(t)), fuse(domain(t)))
    return ROCDiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(svd_compact!), t::ROCTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, codomain(t) ← V_cod)
    S = ROCDiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eigh_full!), t::ROCTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = ROCDiagonalTensorMap{T}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eig_full!), t::ROCTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = ROCDiagonalTensorMap{Tc}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eigh_vals!), t::ROCTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    return D = ROCDiagonalTensorMap{Tc}(undef, V_D)
end

function TensorKit.Factorizations.MAK.initialize_output(::typeof(eig_vals!), t::ROCTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    return D = ROCDiagonalTensorMap{Tc}(undef, V_D)
end


# TODO
# add VectorInterface extensions for proper AMDGPU promotion
function TensorKit.VectorInterface.promote_add(TA::Type{<:AMDGPU.StridedROCMatrix{Tx}}, TB::Type{<:AMDGPU.StridedROCMatrix{Ty}}, α::Tα = TensorKit.VectorInterface.One(), β::Tβ = TensorKit.VectorInterface.One()) where {Tx, Ty, Tα, Tβ}
    return Base.promote_op(add, Tx, Ty, Tα, Tβ)
end

end
