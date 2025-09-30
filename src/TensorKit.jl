# TensorKit.jl
#
# Main file for module TensorKit, a Julia package for working with
# with tensors, tensor operations and tensor factorizations

module TensorKit

# Exports
#---------
# Types:
export Sector, AbstractIrrep, Irrep
export FusionStyle, UniqueFusion, MultipleFusion, MultiplicityFreeFusion,
       SimpleFusion, GenericFusion
export UnitStyle, SimpleUnit, GenericUnit
export BraidingStyle, SymmetricBraiding, Bosonic, Fermionic, Anyonic, NoBraiding
export Trivial, Z2Irrep, Z3Irrep, Z4Irrep, ZNIrrep, U1Irrep, SU2Irrep, CU1Irrep
export ProductSector
export FermionParity, FermionNumber, FermionSpin
export FibonacciAnyon, IsingAnyon, IsingBimodule
export unit, rightunit, leftunit, allunits, isunit

export VectorSpace, Field, ElementarySpace # abstract vector spaces
export unitspace, zerospace, leftunitspace, rightunitspace
export InnerProductStyle, NoInnerProduct, HasInnerProduct, EuclideanInnerProduct
export ComplexSpace, CartesianSpace, GeneralSpace, GradedSpace # concrete spaces
export ZNSpace, Z2Space, Z3Space, Z4Space, U1Space, CU1Space, SU2Space
export Vect, Rep # space constructors
export CompositeSpace, ProductSpace # composite spaces
export FusionTree
export IndexSpace, HomSpace, TensorSpace, TensorMapSpace
export AbstractTensorMap, AbstractTensor, TensorMap, Tensor # tensors and tensor properties
export DiagonalTensorMap, BraidingTensor
export TruncationScheme
export SpaceMismatch, SectorMismatch, IndexError # error types

# general vector space methods
export space, field, dual, dim, reduceddim, dims, fuse, flip, isdual, oplus,
       insertleftunitspace, insertrightunitspace, removeunitspace

# partial order for vector spaces
export infimum, supremum, isisomorphic, ismonomorphic, isepimorphic

# methods for sectors and properties thereof
export sectortype, sectors, hassector, Nsymbol, Fsymbol, Rsymbol, Bsymbol,
       frobeniusschur, twist, otimes, sectorscalartype, deligneproduct
export fusiontrees, braid, permute, transpose
export ZNSpace, SU2Irrep, U1Irrep, CU1Irrep
# other fusion tree manipulations, should not be exported:
# export insertat, split, merge, repartition, artin_braid,
#        bendleft, bendright, foldleft, foldright, cycleclockwise, cycleanticlockwise

# some unicode
export ⊕, ⊗, ×, ⊠, ℂ, ℝ, ℤ, ←, →, ≾, ≿, ≅, ≺, ≻
export ℤ₂, ℤ₃, ℤ₄, U₁, SU, SU₂, CU₁
export fℤ₂, fU₁, fSU₂
export ℤ₂Space, ℤ₃Space, ℤ₄Space, U₁Space, CU₁Space, SU₂Space

# tensor maps
export domain, codomain, numind, numout, numin, domainind, codomainind, allind
export spacetype, sectortype, storagetype, scalartype, tensormaptype
export blocksectors, blockdim, block, blocks

# random methods for constructor
export randisometry, randisometry!, rand, rand!, randn, randn!

# special purpose constructors
export zero, one, one!, id, id!, isomorphism, isomorphism!, unitary, unitary!, isometry,
       isometry!

# reexport most of VectorInterface and some more tensor algebra
export zerovector, zerovector!, zerovector!!, scale, scale!, scale!!, add, add!, add!!
export inner, dot, norm, normalize, normalize!, tr

# factorizations
export mul!, lmul!, rmul!, adjoint!, pinv, axpy!, axpby!
export leftorth, rightorth, leftnull, rightnull,
       leftorth!, rightorth!, leftnull!, rightnull!,
       tsvd!, tsvd, eigen, eigen!, eig, eig!, eigh, eigh!, exp, exp!,
       isposdef, isposdef!, ishermitian, sylvester, rank, cond
export braid, braid!, permute, permute!, transpose, transpose!, twist, twist!, repartition,
       repartition!
export catdomain, catcodomain, absorb, absorb!

export OrthogonalFactorizationAlgorithm, QR, QRpos, QL, QLpos, LQ, LQpos, RQ, RQpos,
       SVD, SDD, Polar

# tensor operations
export @tensor, @tensoropt, @ncon, ncon, @planar, @plansor
export scalar, add!, contract!

# truncation schemes
export notrunc, truncerr, truncdim, truncspace, truncbelow

# cache management
export empty_globalcaches!

# Imports
#---------
using TupleTools
using TupleTools: StaticLength

using Strided

using VectorInterface

using TensorOperations: TensorOperations, @tensor, @tensoropt, @ncon, ncon
using TensorOperations: IndexTuple, Index2Tuple, linearize, AbstractBackend
const TO = TensorOperations

using LRUCache

using TensorKitSectors
import TensorKitSectors: dim, BraidingStyle, FusionStyle, ⊠, ⊗
import TensorKitSectors: dual, type_repr
import TensorKitSectors: twist

using Base: @boundscheck, @propagate_inbounds, @constprop,
            OneTo, tail, front,
            tuple_type_head, tuple_type_tail, tuple_type_cons,
            SizeUnknown, HasLength, HasShape, IsInfinite, EltypeUnknown, HasEltype
using Base.Iterators: product, filter

using LinearAlgebra: LinearAlgebra
using LinearAlgebra: norm, dot, normalize, normalize!, tr,
                     axpy!, axpby!, lmul!, rmul!, mul!, ldiv!, rdiv!,
                     adjoint, adjoint!, transpose, transpose!,
                     lu, pinv, sylvester,
                     eigen, eigen!, svd, svd!,
                     isposdef, isposdef!, ishermitian, rank, cond,
                     Diagonal, Hermitian

import Base.Meta

using Random: Random, rand!, randn!

using PackageExtensionCompat

# Auxiliary files
#-----------------
include("auxiliary/auxiliary.jl")
include("auxiliary/caches.jl")
include("auxiliary/dicts.jl")
include("auxiliary/iterators.jl")
include("auxiliary/linalg.jl")
include("auxiliary/random.jl")

#--------------------------------------------------------------------
# experiment with different dictionaries
const SectorDict{K,V} = SortedVectorDict{K,V}
const FusionTreeDict{K,V} = Dict{K,V}
#--------------------------------------------------------------------

# Exception types:
#------------------
abstract type TensorException <: Exception end

# Exception type for all errors related to sector mismatch
struct SectorMismatch{S<:Union{Nothing,AbstractString}} <: TensorException
    message::S
end
SectorMismatch() = SectorMismatch{Nothing}(nothing)
Base.showerror(io::IO, ::SectorMismatch{Nothing}) = print(io, "SectorMismatch()")
Base.showerror(io::IO, e::SectorMismatch) = print(io, "SectorMismatch(\"", e.message, "\")")

# Exception type for all errors related to vector space mismatch
struct SpaceMismatch{S<:Union{Nothing,AbstractString}} <: TensorException
    message::S
end
SpaceMismatch() = SpaceMismatch{Nothing}(nothing)
Base.showerror(io::IO, ::SpaceMismatch{Nothing}) = print(io, "SpaceMismatch()")
Base.showerror(io::IO, e::SpaceMismatch) = print(io, "SpaceMismatch(\"", e.message, "\")")

# Exception type for all errors related to invalid tensor index specification.
struct IndexError{S<:Union{Nothing,AbstractString}} <: TensorException
    message::S
end
IndexError() = IndexError{Nothing}(nothing)
Base.showerror(io::IO, ::IndexError{Nothing}) = print(io, "IndexError()")
Base.showerror(io::IO, e::IndexError) = print(io, "IndexError(", e.message, ")")

# Constructing and manipulating fusion trees and iterators thereof
#------------------------------------------------------------------
include("fusiontrees/fusiontrees.jl")

# Definitions and methods for vector spaces
#-------------------------------------------
include("spaces/vectorspaces.jl")

# Multithreading settings
#-------------------------
const TRANSFORMER_THREADS = Ref(1)

get_num_transformer_threads() = TRANSFORMER_THREADS[]

function set_num_transformer_threads(n::Int)
    N = Base.Threads.nthreads()
    if n > N
        n = N
        Strided._set_num_threads_warn(n)
    end
    return TRANSFORMER_THREADS[] = n
end

# Definitions and methods for tensors
#-------------------------------------
# general definitions
include("tensors/abstracttensor.jl")
include("tensors/blockiterator.jl")
include("tensors/tensor.jl")
include("tensors/adjoint.jl")
include("tensors/linalg.jl")
include("tensors/vectorinterface.jl")
include("tensors/tensoroperations.jl")
include("tensors/treetransformers.jl")
include("tensors/indexmanipulations.jl")
include("tensors/diagonal.jl")
include("tensors/truncation.jl")
include("tensors/factorizations.jl")
include("tensors/braidingtensor.jl")

# # Planar macros and related functionality
# #-----------------------------------------
@nospecialize
using Base.Meta: isexpr
include("planar/analyzers.jl")
include("planar/preprocessors.jl")
include("planar/postprocessors.jl")
include("planar/macros.jl")
@specialize
include("planar/planaroperations.jl")

# deprecations: to be removed in version 1.0 or sooner
include("auxiliary/deprecate.jl")

# Extensions
# ----------
function __init__()
    @require_extensions
end

end
