# FIELDS:
#==============================================================================#
"""
    abstract type Field end

Abstract type at the top of the type hierarchy for denoting fields over which vector spaces
(or more generally, linear categories) can be defined. Two common fields are `ℝ` and `ℂ`,
representing the field of real or complex numbers respectively.
"""
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()

Base.show(io::IO, ::RealNumbers) = print(io, "ℝ")
Base.show(io::IO, ::ComplexNumbers) = print(io, "ℂ")

Base.in(::Any, ::Field) = false
Base.in(::Real, ::RealNumbers) = true
Base.in(::Number, ::ComplexNumbers) = true

Base.issubset(::Type, ::Field) = false
Base.issubset(::Type{<:Real}, ::RealNumbers) = true
Base.issubset(::Type{<:Number}, ::ComplexNumbers) = true
Base.issubset(::RealNumbers, ::RealNumbers) = true
Base.issubset(::RealNumbers, ::ComplexNumbers) = true
Base.issubset(::ComplexNumbers, ::RealNumbers) = false
Base.issubset(::ComplexNumbers, ::ComplexNumbers) = true

# VECTOR SPACES:
#==============================================================================#
"""
    abstract type VectorSpace end

Abstract type at the top of the type hierarchy for denoting vector spaces, or, more
accurately, 𝕜-linear categories. All instances of subtypes of VectorSpace will
represent objects in 𝕜-linear monoidal categories.
"""
abstract type VectorSpace end

"""
    field(a) -> Type{𝔽<:Field}
    field(::Type{T}) -> Type{𝔽<:Field}

Return the type of field over which object `a` (e.g. a vector space or a tensor) is defined.
Also works in type domain.
"""
field(x) = field(typeof(x))
field(::Type{T}) where {T} = field(spacetype(T))

# Basic vector space methods
#----------------------------
"""
    space(a) -> VectorSpace

Return the vector space associated to object `a`.
"""
function space end

@doc """
    dim(V::VectorSpace) -> Int

Return the total dimension of the vector space `V` as an Int.
""" dim(::VectorSpace)

"""
    dual(V::VectorSpace) -> VectorSpace

Return the dual space of `V`; also obtained via `V'`. This should satisfy
`dual(dual(V)) == V`. It is assumed that `typeof(V) == typeof(V')`.
"""
function dual end

# convenience definitions:
Base.adjoint(V::VectorSpace) = dual(V)

"""
    isdual(V::ElementarySpace) -> Bool

Return wether an ElementarySpace `V` is normal or rather a dual space. Always returns
`false` for spaces where `V == dual(V)`.
"""
function isdual end

# Hierarchy of elementary vector spaces
#---------------------------------------
"""
    abstract type ElementarySpace <: VectorSpace end

Elementary finite-dimensional vector space over a field that can be used as the index
space corresponding to the indices of a tensor. ElementarySpace is a supertype for all
vector spaces (objects) that can be associated with the individual indices of a tensor,
as hinted to by its alias IndexSpace.

Every elementary vector space should respond to the methods [`conj`](@ref) and
[`dual`](@ref), returning the complex conjugate space and the dual space respectively. The
complex conjugate of the dual space is obtained as `dual(conj(V)) === conj(dual(V))`. These
different spaces should be of the same type, so that a tensor can be defined as an element
of a homogeneous tensor product of these spaces.
"""
abstract type ElementarySpace <: VectorSpace end
const IndexSpace = ElementarySpace

# field(::Type{<:ElementarySpace{𝕜}}) where {𝕜} = 𝕜

@doc """
    dim(V::ElementarySpace, s::Sector) -> Int

Return the degeneracy dimension corresponding to the sector `s` of the vector space `V`.
""" dim(::ElementarySpace, ::Sector)

@doc """
    reduceddim(V::ElementarySpace) -> Int

Return the sum of all degeneracy dimensions of the vector space `V`.
"""
reduceddim(V::ElementarySpace) = sum(Base.Fix1(dim, V), sectors(V); init=0)

"""
    oneunit(V::S) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the trivial
one-dimensional space, i.e. the space that is isomorphic to the corresponding field. Note
that this is different from `one(V::S)`, which returns the empty product space
`ProductSpace{S,0}(())`.
"""
Base.oneunit(V::ElementarySpace) = oneunit(typeof(V))

"""
    zero(V::S) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the zero-dimensional or empty space.
This is, with a slight abuse of notation, the zero element of the direct sum of vector spaces. 
"""
Base.zero(V::ElementarySpace) = zero(typeof(V))

"""
    ⊕(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} -> S
    oplus(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the direct sum sum of the
spaces `V₁`, `V₂`, ... Note that all the individual spaces should have the same value for
[`isdual`](@ref), as otherwise the direct sum is not defined.
"""
function ⊕ end
⊕(V₁::ElementarySpace, V₂::ElementarySpace) = ⊕(promote(V₁, V₂)...)
⊕(V::Vararg{ElementarySpace}) = foldl(⊕, V)
const oplus = ⊕

"""
    ⊖(V::ElementarySpace, W::ElementarySpace) -> X::ElementarySpace
    ominus(V::ElementarySpace, W::ElementarySpace) -> X::ElementarySpace

Return the set difference of two elementary spaces, i.e. an instance `X::ElementarySpace`
such that `V = W ⊕ X`.
"""
⊖(V₁::S, V₂::S) where {S<:ElementarySpace}
⊖(V₁::VectorSpace, V₂::VectorSpace) = ⊖(promote(V₁, V₂)...)
const ominus = ⊖

"""
    ⊗(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} -> S

Create a [`ProductSpace{S}(V₁, V₂, V₃...)`](@ref) representing the tensor product of several
elementary vector spaces. For convience, Julia's regular multiplication operator `*` applied
to vector spaces has the same effect.

The tensor product structure is preserved, see [`fuse`](@ref) for returning a single
elementary space of type `S` that is isomorphic to this tensor product.
"""
⊗(V₁::VectorSpace, V₂::VectorSpace) = ⊗(promote(V₁, V₂)...)
⊗(V::Vararg{VectorSpace}) = foldl(⊗, V)

# convenience definitions:
Base.:*(V₁::VectorSpace, V₂::VectorSpace) = ⊗(V₁, V₂)

"""
    fuse(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} -> S
    fuse(P::ProductSpace{S}) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that is isomorphic to the fusion product of the
individual spaces `V₁`, `V₂`, ..., or the spaces contained in `P`.
"""
function fuse end
fuse(V::ElementarySpace) = isdual(V) ? flip(V) : V
fuse(V::ElementarySpace, W::ElementarySpace) = fuse(promote(V, W)...)
function fuse(V₁::VectorSpace, V₂::VectorSpace, V₃::VectorSpace...)
    return fuse(fuse(fuse(V₁), fuse(V₂)), V₃...)
end
# calling fuse on V₁ and V₂ will allow these to be `ProductSpace`

"""
    flip(V::S) where {S<:ElementarySpace} -> S

Return a single vector space of type `S` that has the same value of [`isdual`](@ref) as
`dual(V)`, but yet is isomorphic to `V` rather than to `dual(V)`. The spaces `flip(V)` and
`dual(V)` only differ in the case of [`GradedSpace{I}`](@ref).
"""
function flip end

"""
    conj(V::S) where {S<:ElementarySpace} -> S

Return the conjugate space of `V`. This should satisfy `conj(conj(V)) == V`.

For `field(V)==ℝ`, `conj(V) == V`. It is assumed that `typeof(V) == typeof(conj(V))`.
"""
function Base.conj(V::ElementarySpace)
    @assert field(V) == ℝ "default conj only defined for Vector spaces over ℝ"
    return V
end

# In the following, X can be a ProductSpace, a HomSpace or an AbstractTensorMap
# TODO: should we deprecate those in the future?
@constprop :aggressive function insertleftunit(X, i::Int; kwargs...)
    return insertleftunit(X, Val(i); kwargs...)
end
@constprop :aggressive function insertrightunit(X, i::Int; kwargs...)
    return insertrightunit(X, Val(i); kwargs...)
end
@constprop :aggressive function removeunit(X, i::Int; kwargs...)
    return removeunit(X, Val(i); kwargs...)
end

# trait to describe the inner product type of vector spaces
abstract type InnerProductStyle end
struct NoInnerProduct <: InnerProductStyle end # no inner product

abstract type HasInnerProduct <: InnerProductStyle end # inner product defined
struct EuclideanInnerProduct <: HasInnerProduct end # euclidean inner product

"""
    InnerProductStyle(V::VectorSpace) -> ::InnerProductStyle
    InnerProductStyle(S::Type{<:VectorSpace}) -> ::InnerProductStyle

Return the type of inner product for vector spaces, which can be either
*   `NoInnerProduct()`: no mapping from `dual(V)` to `conj(V)`, i.e. no metric
*   subtype of `HasInnerProduct`: a metric exists, but no further support is implemented.
*   `EuclideanInnerProduct()`: the metric is the identity, such that dual and conjugate spaces are isomorphic.
"""
InnerProductStyle(V::VectorSpace) = InnerProductStyle(typeof(V))
InnerProductStyle(::Type{<:VectorSpace}) = NoInnerProduct()

@noinline function throw_invalid_innerproduct(fname)
    throw(ArgumentError("$fname requires Euclidean inner product"))
end

dual(V::VectorSpace) = dual(InnerProductStyle(V), V)
dual(::EuclideanInnerProduct, V::VectorSpace) = conj(V)

"""
    sectortype(a) -> Type{<:Sector}
    sectortype(::Type) -> Type{<:Sector}

Return the type of sector over which object `a` (e.g. a representation space or a tensor) is
defined. Also works in type domain.
"""
sectortype(x) = sectortype(typeof(x))
sectortype(::Type{T}) where {T} = sectortype(spacetype(T))
sectortype(::Type{S}) where {S<:Sector} = S

"""
    hassector(V::VectorSpace, a::Sector) -> Bool

Return whether a vector space `V` has a subspace corresponding to sector `a` with non-zero
dimension, i.e. `dim(V, a) > 0`.
"""
function hassector end

"""
    sectors(V::ElementarySpace)

Return an iterator over the different sectors of `V`.
"""
function sectors end

# Composite vector spaces
#-------------------------
"""
    abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

Abstract type for composite spaces that are defined in terms of a number of elementary
vector spaces of a homogeneous type `S<:ElementarySpace`.
"""
abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

InnerProductStyle(::Type{<:CompositeSpace{S}}) where {S} = InnerProductStyle(S)

"""
    spacetype(a) -> Type{S<:IndexSpace}
    spacetype(::Type) -> Type{S<:IndexSpace}

Return the type of the elementary space `S` of object `a` (e.g. a tensor). Also works in
type domain.
"""
spacetype(x) = spacetype(typeof(x))
function spacetype(::Type{T}) where {T}
    throw(MethodError(spacetype, (T,)))
end
spacetype(::Type{E}) where {E<:ElementarySpace} = E
spacetype(::Type{S}) where {E,S<:CompositeSpace{E}} = E

# make ElementarySpace instances behave similar to ProductSpace instances
blocksectors(V::ElementarySpace) = collect(sectors(V))
blockdim(V::ElementarySpace, c::Sector) = dim(V, c)

# Specific realizations of ElementarySpace types
#------------------------------------------------
# spaces without internal structure
include("cartesianspace.jl")
include("complexspace.jl")
include("generalspace.jl")

# space with internal structure corresponding to the irreducible representations of
# a group, or more generally, the simple objects of a fusion category.
include("gradedspace.jl")
include("planarspace.jl")

# Specific realizations of CompositeSpace types
#-----------------------------------------------
# a tensor product of N elementary spaces of the same type S
include("productspace.jl")
# deligne tensor product
include("deligne.jl")

# Other examples might include:
# symmetric and antisymmetric subspace of a tensor product of identical vector spaces
# ...

# HomSpace: space of morphisms
#------------------------------
include("homspace.jl")

# Partial order for vector spaces
#---------------------------------
"""
    isisomorphic(V₁::VectorSpace, V₂::VectorSpace)
    V₁ ≅ V₂

Return if `V₁` and `V₂` are isomorphic, meaning that there exists isomorphisms from `V₁` to
`V₂`, i.e. morphisms with left and right inverses.
"""
function isisomorphic(V₁::VectorSpace, V₂::VectorSpace)
    spacetype(V₁) == spacetype(V₂) || return false
    for c in union(blocksectors(V₁), blocksectors(V₂))
        if blockdim(V₁, c) != blockdim(V₂, c)
            return false
        end
    end
    return true
end

"""
    ismonomorphic(V₁::VectorSpace, V₂::VectorSpace)
    V₁ ≾ V₂

Return whether there exist monomorphisms from `V₁` to `V₂`, i.e. 'injective' morphisms with
left inverses.
"""
function ismonomorphic(V₁::VectorSpace, V₂::VectorSpace)
    spacetype(V₁) == spacetype(V₂) || return false
    for c in blocksectors(V₁)
        if blockdim(V₁, c) > blockdim(V₂, c)
            return false
        end
    end
    return true
end

"""
    isepimorphic(V₁::VectorSpace, V₂::VectorSpace)
    V₁ ≿ V₂

Return whether there exist epimorphisms from `V₁` to `V₂`, i.e. 'surjective' morphisms with
right inverses.
"""
function isepimorphic(V₁::VectorSpace, V₂::VectorSpace)
    spacetype(V₁) == spacetype(V₂) || return false
    for c in blocksectors(V₂)
        if blockdim(V₁, c) < blockdim(V₂, c)
            return false
        end
    end
    return true
end

# unicode alternatives
const ≅ = isisomorphic
const ≾ = ismonomorphic
const ≿ = isepimorphic

≺(V₁::VectorSpace, V₂::VectorSpace) = V₁ ≾ V₂ && !(V₁ ≿ V₂)
≻(V₁::VectorSpace, V₂::VectorSpace) = V₁ ≿ V₂ && !(V₁ ≾ V₂)

"""
    infimum(V₁::ElementarySpace, V₂::ElementarySpace, V₃::ElementarySpace...)

Return the infimum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≾ V₁`, `V ≾ V₂`, ... and no other `W ≻ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
infimum(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} = infimum(infimum(V₁, V₂), V₃...)

"""
    supremum(V₁::ElementarySpace, V₂::ElementarySpace, V₃::ElementarySpace...)

Return the supremum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≿ V₁`, `V ≿ V₂`, ... and no other `W ≺ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
function supremum(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace}
    return supremum(supremum(V₁, V₂), V₃...)
end
