# Tensor factorization
#----------------------
# using submodule here to import MatrixAlgebraKit functions without polluting namespace
module Factorizations

export eig, eig!, eigh, eigh!
export tsvd, tsvd!, svdvals, svdvals!
export leftorth, leftorth!, rightorth, rightorth!
export leftnull, leftnull!, rightnull, rightnull!
export qr_full, qr_compact, qr_null
export qr_full!, qr_compact!, qr_null!
export lq_full, lq_compact, lq_null
export lq_full!, lq_compact!, lq_null!
export copy_oftype, factorisation_scalartype, one!
export TruncationScheme, notrunc, trunctol, truncerror, truncrank, truncspace, truncfilter,
       PolarViaSVD

using ..TensorKit
using ..TensorKit: AdjointTensorMap, SectorDict, blocktype, foreachblock, one!

using LinearAlgebra: LinearAlgebra, BlasFloat, Diagonal, svdvals, svdvals!
import LinearAlgebra: eigen, eigen!, isposdef, isposdef!, ishermitian

using TensorOperations: Index2Tuple

using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncatedAlgorithm, DiagonalAlgorithm
using MatrixAlgebraKit: TruncationStrategy, NoTruncation, TruncationByValue,
                        TruncationByError, TruncationIntersection, TruncationByFilter,
                        TruncationByOrder
using MatrixAlgebraKit: PolarViaSVD
using MatrixAlgebraKit: LAPACK_SVDAlgorithm, LAPACK_QRIteration, LAPACK_HouseholderQR,
                        LAPACK_HouseholderLQ, LAPACK_HouseholderQL, LAPACK_HouseholderRQ
import MatrixAlgebraKit: default_algorithm,
                         copy_input, check_input, initialize_output,
                         qr_compact!, qr_full!, qr_null!, lq_compact!, lq_full!, lq_null!,
                         svd_compact!, svd_full!, svd_trunc!, svd_vals!,
                         eigh_full!, eigh_trunc!, eigh_vals!,
                         eig_full!, eig_trunc!, eig_vals!,
                         left_polar!, left_orth_polar!, right_polar!, right_orth_polar!,
                         left_null_svd!, right_null_svd!, left_orth_svd!, right_orth_svd!,
                         left_orth!, right_orth!, left_null!, right_null!,
                         truncate!, findtruncated, findtruncated_svd,
                         diagview, isisometry
using MatrixAlgebraKit: qr_compact_pullback!, lq_compact_pullback!,
                        svd_pullback!, svd_trunc_pullback!,
                        eig_pullback!, eig_trunc_pullback!,
                        eigh_pullback!, eigh_trunc_pullback!,
                        left_polar_pullback!, right_polar_pullback!

include("utility.jl")
include("matrixalgebrakit.jl")
include("truncation.jl")
include("adjoint.jl")
include("diagonal.jl")
include("pullbacks.jl")

TensorKit.one!(A::AbstractMatrix) = MatrixAlgebraKit.one!(A)

function isisometry(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    t = permute(t, (p₁, p₂); copy=false)
    return isisometry(t)
end

#------------------------------#
# LinearAlgebra overloads
#------------------------------#

function eigen(t::AbstractTensorMap; kwargs...)
    return ishermitian(t) ? eigh_full(t; kwargs...) : eig_full(t; kwargs...)
end
function eigen!(t::AbstractTensorMap; kwargs...)
    return ishermitian(t) ? eigh_full!(t; kwargs...) : eig_full!(t; kwargs...)
end

function LinearAlgebra.eigvals(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, factorisation_scalartype(eigen, t))
    return LinearAlgebra.eigvals!(tcopy; kwargs...)
end
LinearAlgebra.eigvals!(t::AbstractTensorMap; kwargs...) = diagview(eig_vals!(t))

function LinearAlgebra.svdvals(t::AbstractTensorMap)
    tcopy = copy_oftype(t, factorisation_scalartype(tsvd, t))
    return LinearAlgebra.svdvals!(tcopy)
end
LinearAlgebra.svdvals!(t::AbstractTensorMap) = diagview(svd_vals!(t))

#--------------------------------------------------#
# Checks for hermiticity and positive definiteness #
#--------------------------------------------------#
function LinearAlgebra.ishermitian(t::AbstractTensorMap)
    domain(t) == codomain(t) || return false
    InnerProductStyle(t) === EuclideanInnerProduct() || return false # hermiticity only defined for euclidean
    for (c, b) in blocks(t)
        ishermitian(b) || return false
    end
    return true
end

function LinearAlgebra.isposdef(t::AbstractTensorMap)
    return isposdef!(copy_oftype(t, factorisation_scalartype(isposdef, t)))
end
function LinearAlgebra.isposdef!(t::AbstractTensorMap)
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
