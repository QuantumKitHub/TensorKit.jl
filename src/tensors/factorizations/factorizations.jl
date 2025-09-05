# Tensor factorization
#----------------------
# using submodule here to import MatrixAlgebraKit functions without polluting namespace
module Factorizations

export eig, eig!, eigh, eigh!
export tsvd, tsvd!, svdvals, svdvals!
export leftorth, leftorth!, rightorth, rightorth!
export leftnull, leftnull!, rightnull, rightnull!
export copy_oftype, permutedcopy_oftype, one!
export TruncationScheme, notrunc, truncbelow, truncerr, truncdim, truncspace

using ..TensorKit
using ..TensorKit: AdjointTensorMap, SectorDict, OFA, blocktype, foreachblock, one!

using LinearAlgebra: LinearAlgebra, BlasFloat, Diagonal, svdvals, svdvals!
import LinearAlgebra: eigen, eigen!, isposdef, isposdef!, ishermitian

using TensorOperations: Index2Tuple

using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm, TruncatedAlgorithm, TruncationStrategy,
                        NoTruncation, TruncationKeepAbove, TruncationKeepBelow,
                        TruncationIntersection, TruncationKeepFiltered, DiagonalAlgorithm
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
include("adjoint.jl")
include("diagonal.jl")

TensorKit.one!(A::AbstractMatrix) = MatrixAlgebraKit.one!(A)

function isisometry(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    t = permute(t, (p₁, p₂); copy=false)
    return isisometry(t)
end

# Orthogonal factorizations (mutation for recycling memory):
# only possible if scalar type is floating point
# only correct if Euclidean inner product
#------------------------------------------------------------------------------------------
const RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

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
