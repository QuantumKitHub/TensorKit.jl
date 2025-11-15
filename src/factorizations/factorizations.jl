# Tensor factorization
#----------------------
# using submodule here to import MatrixAlgebraKit functions without polluting namespace
module Factorizations

export copy_oftype, factorisation_scalartype, one!, truncspace

using ..TensorKit
using ..TensorKit: AdjointTensorMap, SectorDict, blocktype, foreachblock, one!

using LinearAlgebra: LinearAlgebra, BlasFloat, Diagonal, svdvals, svdvals!, eigen, eigen!,
    isposdef, isposdef!, ishermitian

using TensorOperations: Index2Tuple

using MatrixAlgebraKit
import MatrixAlgebraKit as MAK
using MatrixAlgebraKit: AbstractAlgorithm, TruncatedAlgorithm, DiagonalAlgorithm
using MatrixAlgebraKit: TruncationStrategy, NoTruncation, TruncationByValue,
    TruncationByError, TruncationIntersection, TruncationByFilter, TruncationByOrder
using MatrixAlgebraKit: diagview

include("utility.jl")
include("matrixalgebrakit.jl")
include("truncation.jl")
include("adjoint.jl")
include("diagonal.jl")
include("pullbacks.jl")

TensorKit.one!(A::AbstractMatrix) = MatrixAlgebraKit.one!(A)

#------------------------------#
# LinearAlgebra overloads
#------------------------------#

function LinearAlgebra.eigen(t::AbstractTensorMap; kwargs...)
    return ishermitian(t) ? eigh_full(t; kwargs...) : eig_full(t; kwargs...)
end
function LinearAlgebra.eigen!(t::AbstractTensorMap; kwargs...)
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
function MAK.ishermitian(t::AbstractTensorMap; kwargs...)
    return InnerProductStyle(t) === EuclideanInnerProduct() &&
        domain(t) == codomain(t) &&
        all(cb -> MatrixAlgebraKit.ishermitian(cb[2]; kwargs...), blocks(t))
end
function MAK.isantihermitian(t::AbstractTensorMap; kwargs...)
    return InnerProductStyle(t) === EuclideanInnerProduct() &&
        domain(t) == codomain(t) &&
        all(cb -> MatrixAlgebraKit.isantihermitian(cb[2]; kwargs...), blocks(t))
end
LinearAlgebra.ishermitian(t::AbstractTensorMap) = MAK.ishermitian(t)

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
function MAK.is_left_isometric(t::AbstractTensorMap; kwargs...)
    domain(t) ≾ codomain(t) || return false
    f((c, b)) = MAK.is_left_isometric(b; kwargs...)
    return all(f, blocks(t))
end
function MAK.is_right_isometric(t::AbstractTensorMap; kwargs...)
    domain(t) ≿ codomain(t) || return false
    f((c, b)) = MAK.is_right_isometric(b; kwargs...)
    return all(f, blocks(t))
end
function MAK.isisometric(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple; kwargs...)
    t = permute(t, (p₁, p₂); copy = false)
    return MAK.isisometric(t; kwargs...)
end

end
