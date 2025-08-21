# Tensor factorization
#----------------------
# using submodule here to import MatrixAlgebraKit functions without polluting namespace
module Factorizations

export eig, eig!, eigh, eigh!
export tsvd, tsvd!, svdvals, svdvals!
export leftorth, leftorth!, rightorth, rightorth!
export leftnull, leftnull!, rightnull, rightnull!
export copy_oftype, permutedcopy_oftype
export TruncationScheme, notrunc, truncbelow, truncerr, truncdim, truncspace

using ..TensorKit
using ..TensorKit: AdjointTensorMap, SectorDict, OFA, blocktype, foreachblock

using LinearAlgebra: LinearAlgebra, BlasFloat, Diagonal, svdvals, svdvals!
import LinearAlgebra: eigen, eigen!, isposdef, isposdef!, ishermitian

using TensorOperations: Index2Tuple

using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncatedAlgorithm, TruncationStrategy,
                        NoTruncation, TruncationKeepAbove, TruncationKeepBelow,
                        TruncationIntersection, TruncationKeepFiltered
import MatrixAlgebraKit: default_algorithm,
                         copy_input, check_input, initialize_output,
                         qr_compact!, qr_full!, qr_null!, lq_compact!, lq_full!, lq_null!,
                         svd_compact!, svd_full!, svd_trunc!, svd_vals!,
                         eigh_full!, eigh_trunc!, eigh_vals!,
                         eig_full!, eig_trunc!, eig_vals!,
                         left_polar!, left_orth_polar!, right_polar!, right_orth_polar!,
                         left_null_svd!, right_null_svd!,
                         left_orth!, right_orth!, left_null!, right_null!,
                         truncate!, findtruncated, findtruncated_sorted,
                         diagview, isisometry

include("utility.jl")
include("interface.jl")
include("implementations.jl")
include("matrixalgebrakit.jl")
include("truncation.jl")
include("deprecations.jl")

function isisometry(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    t = permute(t, (p₁, p₂); copy=false)
    return isisometry(t)
end

# Orthogonal factorizations (mutation for recycling memory):
# only possible if scalar type is floating point
# only correct if Euclidean inner product
#------------------------------------------------------------------------------------------
const RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

# AdjointTensorMap
# ----------------
function leftorth!(t::AdjointTensorMap; alg::OFA=QRpos())
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftorth!)
    return map(adjoint, reverse(rightorth!(adjoint(t); alg=alg')))
end

function rightorth!(t::AdjointTensorMap; alg::OFA=LQpos())
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightorth!)
    return map(adjoint, reverse(leftorth!(adjoint(t); alg=alg')))
end

function leftnull!(t::AdjointTensorMap; alg::OFA=QR(), kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftnull!)
    return adjoint(rightnull!(adjoint(t); alg=alg', kwargs...))
end

function rightnull!(t::AdjointTensorMap; alg::OFA=LQ(), kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightnull!)
    return adjoint(leftnull!(adjoint(t); alg=alg', kwargs...))
end

function tsvd!(t::AdjointTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    u, s, vt, err = tsvd!(adjoint(t); trunc=trunc, p=p, alg=alg)
    return adjoint(vt), adjoint(s), adjoint(u), err
end

# DiagonalTensorMap
# -----------------
function leftorth!(d::DiagonalTensorMap; alg=QR(), kwargs...)
    @assert alg isa Union{QR,QL}
    return one(d), d # TODO: this is only correct for `alg = QR()` or `alg = QL()`
end
function rightorth!(d::DiagonalTensorMap; alg=LQ(), kwargs...)
    @assert alg isa Union{LQ,RQ}
    return d, one(d) # TODO: this is only correct for `alg = LQ()` or `alg = RQ()`
end
leftnull!(d::DiagonalTensorMap; kwargs...) = leftnull!(TensorMap(d); kwargs...)
rightnull!(d::DiagonalTensorMap; kwargs...) = rightnull!(TensorMap(d); kwargs...)

function tsvd!(d::DiagonalTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    return _tsvd!(d, alg, trunc, p)
end

# helper function
function _compute_svddata!(d::DiagonalTensorMap, alg::Union{SVD,SDD})
    InnerProductStyle(d) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(d)
    dims = SectorDict{I,Int}()
    generator = Base.Iterators.map(blocks(d)) do (c, b)
        lb = length(b.diag)
        U = zerovector!(similar(b.diag, lb, lb))
        V = zerovector!(similar(b.diag, lb, lb))
        p = sortperm(b.diag; by=abs, rev=true)
        for (i, pi) in enumerate(p)
            U[pi, i] = safesign(b.diag[pi])
            V[i, pi] = 1
        end
        Σ = abs.(view(b.diag, p))
        dims[c] = lb
        return c => (U, Σ, V)
    end
    SVDdata = SectorDict(generator)
    return SVDdata, dims
end

eig!(d::DiagonalTensorMap) = d, one(d)
eigh!(d::DiagonalTensorMap{<:Real}) = d, one(d)
eigh!(d::DiagonalTensorMap{<:Complex}) = DiagonalTensorMap(real(d.data), d.domain), one(d)

function LinearAlgebra.svdvals(d::DiagonalTensorMap)
    return SectorDict(c => LinearAlgebra.svdvals(b) for (c, b) in blocks(d))
end
function LinearAlgebra.eigvals(d::DiagonalTensorMap)
    return SectorDict(c => LinearAlgebra.eigvals(b) for (c, b) in blocks(d))
end

function LinearAlgebra.cond(d::DiagonalTensorMap, p::Real=2)
    return LinearAlgebra.cond(Diagonal(d.data), p)
end
#------------------------------#
# Singular value decomposition #
#------------------------------#
function LinearAlgebra.svdvals!(t::TensorMap{<:RealOrComplexFloat})
    return SectorDict(c => LinearAlgebra.svdvals!(b) for (c, b) in blocks(t))
end
LinearAlgebra.svdvals!(t::AdjointTensorMap) = svdvals!(adjoint(t))

#--------------------------#
# Eigenvalue decomposition #
#--------------------------#

function LinearAlgebra.eigvals!(t::TensorMap{<:RealOrComplexFloat}; kwargs...)
    return SectorDict(c => complex(LinearAlgebra.eigvals!(b; kwargs...))
                      for (c, b) in blocks(t))
end
function LinearAlgebra.eigvals!(t::AdjointTensorMap{<:RealOrComplexFloat}; kwargs...)
    return SectorDict(c => conj!(complex(LinearAlgebra.eigvals!(b; kwargs...)))
                      for (c, b) in blocks(t))
end

#--------------------------------------------------#
# Checks for hermiticity and positive definiteness #
#--------------------------------------------------#
function LinearAlgebra.ishermitian(t::TensorMap)
    domain(t) == codomain(t) || return false
    InnerProductStyle(t) === EuclideanInnerProduct() || return false # hermiticity only defined for euclidean
    for (c, b) in blocks(t)
        ishermitian(b) || return false
    end
    return true
end

function LinearAlgebra.isposdef!(t::TensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanInnerProduct() || return false
    for (c, b) in blocks(t)
        isposdef!(b) || return false
    end
    return true
end

# TODO: tolerances are per-block, not global or weighted - does that matter?
function MatrixAlgebraKit.is_left_isometry(t::AbstractTensorMap; kwargs...)
    domain(t) ≾ codomain(t) || return false
    f((c, b)) = MatrixAlgebraKit.is_left_isometry(b; kwargs...)
    return all(f, blocks(t))
end
function MatrixAlgebraKit.is_right_isometry(t::AbstractTensorMap; kwargs...)
    domain(t) ≿ codomain(t) || return false
    f((c, b)) = MatrixAlgebraKit.is_right_isometry(b; kwargs...)
    return all(f, blocks(t))
end

end
