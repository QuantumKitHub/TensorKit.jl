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
generally, objects in linear monoidal categories.
"""
abstract type VectorSpace end

"""
    field(a) -> Type{𝔽 <: Field}
    field(::Type{T}) -> Type{𝔽 <: Field}

Return the type of field over which object `a` (e.g. a vector space or a tensor) is defined.
This also works in type domain.
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

@doc """
    dual(V::VectorSpace) -> VectorSpace

Return the dual space of `V`; also obtained via `V'`. This should satisfy
`dual(dual(V)) == V`. It is assumed that `typeof(V) == typeof(V')`.
""" dual(::VectorSpace)

@doc """
    conj(V::VectorSpace) -> VectorSpace

Return the conjugate space of `V`. This should satisfy `conj(conj(V)) == V`.
For vector spaces over the real numbers, it must hold that `conj(V) == V`.
For vector spaces with a Euclidean inner product, it must hold that `conj(V) == dual(V)`.
""" conj(::VectorSpace)

# convenience definitions:
Base.adjoint(V::VectorSpace) = dual(V)

# Hierarchy of elementary vector spaces
#---------------------------------------
"""
    abstract type ElementarySpace <: VectorSpace

Elementary finite-dimensional vector space over a field that can be used as the index
space corresponding to the indices of a tensor. ElementarySpace is a supertype for all
vector spaces (objects) that can be associated with the individual indices of a tensor,
as hinted to by its alias `IndexSpace`.

Every elementary vector space should respond to the methods [`conj`](@ref) and
[`dual`](@ref), returning the complex conjugate space and the dual space respectively. The
complex conjugate of the dual space is obtained as `dual(conj(V)) === conj(dual(V))`. These
different spaces should be of the same type, so that a tensor can be defined as an element
of a homogeneous tensor product of these spaces.
"""
abstract type ElementarySpace <: VectorSpace end
const IndexSpace = ElementarySpace

@doc """
    dim(V::ElementarySpace, s::Sector) -> Int

Return the degeneracy dimension corresponding to the sector `s` of the vector space `V`.
""" dim(::ElementarySpace, ::Sector)

"""
    reduceddim(V::ElementarySpace) -> Int

Return the sum of all degeneracy dimensions of the vector space `V`.
"""
reduceddim(V::ElementarySpace) = sum(Base.Fix1(dim, V), sectors(V); init = 0)

@doc """
    isdual(V::ElementarySpace) -> Bool

Return whether an ElementarySpace `V` is normal or rather a dual space.
Always returns `false` for spaces where `V == dual(V)`.
""" isdual(::ElementarySpace)

@doc """
    isconj(V::ElementarySpace) -> Bool

Return whether an ElementarySpace `V` is normal or rather the conjugated space.
Always returns `false` for spaces where `V == conj(V)`, i.e. vector spaces over `ℝ`.
""" isconj(::ElementarySpace)

"""
    unitspace(V::S) where {S <: ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the trivial
one-dimensional space, i.e. the space that is isomorphic to the corresponding field.
For vector spaces where `I = sectortype(S)` has a semi-simple unit structure
(`UnitStyle(I) == GenericUnit()`), this returns a multi-dimensional space corresponding to all unit sectors:
`dim(unitspace(V), s) == 1` for all `s in allunits(I)`. 

!!! note
    `unitspace(V)`is different from `one(V)`. The latter returns the empty product space
    `ProductSpace{S,0}(())`. `Base.oneunit` falls back to `unitspace`.
"""
unitspace(V::ElementarySpace) = unitspace(typeof(V))
Base.oneunit(V::ElementarySpace) = unitspace(V)
Base.oneunit(::Type{V}) where {V <: ElementarySpace} = unitspace(V)

"""
    zerospace(V::S) where {S <: ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the zero-dimensional or empty space.
This is the zero element of the direct sum of vector spaces.
`Base.zero` falls back to `zerospace`.
"""
zerospace(V::ElementarySpace) = zerospace(typeof(V))
Base.zero(V::ElementarySpace) = zerospace(V)
Base.zero(::Type{V}) where {V <: ElementarySpace} = zerospace(V)

"""
    leftunitspace(V::S) where {S <: ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the trivial
one-dimensional space, i.e. the space that is isomorphic to the corresponding field. For vector spaces 
of type `GradedSpace{I}`, this one-dimensional space contains the unique left unit of the objects in `Sector` `I` present
in the vector space.
"""
function leftunitspace(V::ElementarySpace)
    I = sectortype(V)
    if UnitStyle(I) isa SimpleUnit
        return unitspace(typeof(V))
    else
        !isempty(sectors(V)) || throw(ArgumentError("Cannot determine the left unit of an empty space"))
        _allequal(leftunit, sectors(V)) ||
            throw(ArgumentError("sectors of $V do not have the same left unit"))

        sector = leftunit(first(sectors(V)))
        return spacetype(V)(sector => 1)
    end
end

"""
    rightunitspace(V::S) where {S <: ElementarySpace} -> S

Return the corresponding vector space of type `ElementarySpace` that represents the trivial
one-dimensional space, i.e. the space that is isomorphic to the corresponding field. For vector spaces 
of type `GradedSpace{I}`, this corresponds to the right unit of the objects in `Sector` `I` present
in the vector space.
"""
function rightunitspace(V::ElementarySpace)
    I = sectortype(V)
    if UnitStyle(I) isa SimpleUnit
        return unitspace(typeof(V))
    else
        !isempty(sectors(V)) || throw(ArgumentError("Cannot determine the right unit of an empty space"))
        _allequal(rightunit, sectors(V)) ||
            throw(ArgumentError("sectors of $V do not have the same right unit"))

        sector = rightunit(first(sectors(V)))
        return spacetype(V)(sector => 1)
    end
end

"""
    isunitspace(V::S) where {S <: ElementarySpace} -> Bool

Return whether the elementary space `V` is a unit space, i.e. is isomorphic to the
trivial one-dimensional space. For vector spaces of type `GradedSpace{I}` where `Sector` `I` has a
semi-simple unit structure, this returns `true` if `V` is isomorphic to either the left, right or
semi-simple unit space.
"""
function isunitspace(V::ElementarySpace)
    I = sectortype(V)
    return if isa(UnitStyle(I), SimpleUnit)
        isisomorphic(V, unitspace(V))
    else
        (dim(V) == 0 || !all(isunit, sectors(V))) && return false
        return true
    end
end

"""
    ⊕(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} -> S
    oplus(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} -> S

Return the corresponding vector space of type `S` that represents the direct sum sum of the
spaces `V₁`, `V₂`, ... Note that all the individual spaces should have the same value for
[`isdual`](@ref), as otherwise the direct sum is not defined.
"""
⊕(V₁::S, V₂::S) where {S <: ElementarySpace}
⊕(V₁::ElementarySpace, V₂::ElementarySpace) = ⊕(promote(V₁, V₂)...)
⊕(V::Vararg{ElementarySpace}) = foldl(⊕, V)
const oplus = ⊕

"""
    ⊖(V::ElementarySpace, W::ElementarySpace) -> X::ElementarySpace
    ominus(V::ElementarySpace, W::ElementarySpace) -> X::ElementarySpace

Return a space that is equivalent to the orthogonal complement of `W` in `V`,
i.e. an instance `X::ElementarySpace` such that `V = W ⊕ X`.
"""
⊖(V₁::S, V₂::S) where {S <: ElementarySpace}
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
    abstract type InnerProductStyle end
    InnerProductStyle(V::VectorSpace) -> ::InnerProductStyle
    InnerProductStyle(S::Type{<:VectorSpace}) -> ::InnerProductStyle

Trait to describe wether vector spaces exhibit an inner product structure, a.k.a. a unitary structure,
which can take the following values:
*   `EuclideanInnerProduct()`: the metric is the identity, making dual and conjugate spaces equivalent
*   `NoInnerProduct()`: no metric and thus no relation between `dual(V)` or `conj(V)`

Furthermore, `EuclideanInnerProduct` is a subtype of `HasInnerProduct`, indicating that an inner
product exists, and an isomorphism between the dual space and the conjugate space can be constructed.
New inner product styles can be defined that subtype `HasInnerProduct`, for example to work with
vector spaces with non-trivial metrics. However, at the moment TensorKit does not provide built-in
support for such non-standard inner products.
"""
InnerProductStyle(V::VectorSpace) = InnerProductStyle(typeof(V))
InnerProductStyle(::Type{<:VectorSpace}) = NoInnerProduct()

@noinline function throw_invalid_innerproduct(fname)
    throw(ArgumentError("$fname requires Euclidean inner product"))
end

"""
    sectortype(a) -> Type{<:Sector}
    sectortype(::Type) -> Type{<:Sector}

Return the type of sector over which object `a` (e.g. a representation space or a tensor) is
defined. Also works in type domain.
"""
sectortype(x) = sectortype(typeof(x))
sectortype(::Type{T}) where {T} = sectortype(spacetype(T))
sectortype(::Type{S}) where {S <: Sector} = S

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

function sectorequal(V₁::ElementarySpace, V₂::ElementarySpace)
    isdual(V₁) == isdual(V₂) || return false
    return issetequal(sectors(V₁), sectors(V₂))
end
function sectorhash(V::ElementarySpace, h::UInt)
    h = hash(isdual(V), h)
    for s in sectors(V)
        h = hash(s, h)
    end
    return h
end

# Composite vector spaces
#-------------------------
"""
    abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end

Abstract type for composite spaces that are defined in terms of a number of elementary
vector spaces of a homogeneous type `S<:ElementarySpace`.
"""
abstract type CompositeSpace{S <: ElementarySpace} <: VectorSpace end

InnerProductStyle(::Type{<:CompositeSpace{S}}) where {S} = InnerProductStyle(S)

"""
    spacetype(a) -> Type{S <: IndexSpace}
    spacetype(::Type) -> Type{S <: IndexSpace}

Return the type of the elementary space `S` of object `a` (e.g. a tensor).
Also works in type domain.
"""
spacetype(x) = spacetype(typeof(x))
spacetype(::Type{T}) where {T} = throw(MethodError(spacetype, (T,)))
spacetype(::Type{E}) where {E <: ElementarySpace} = E
spacetype(::Type{S}) where {E, S <: CompositeSpace{E}} = E

"""
    check_spacetype(Bool, x, y, z...) -> Bool
    check_spacetype(x, y, z...) -> Type{<:IndexSpace}

Check whether the given inputs have matching spacetypes.
The first signature returns a `Bool` to indicate whether all spacetypes are equal,
while the second will return the spacetype if all types are equal, and throw a [`SpaceMismatch`](@ref) if not.
"""
check_spacetype(::Type{Bool}, x, y, z...) = _allequal(spacetype, (x, y, z...))
@noinline function check_spacetype(x, y, z...)
    check_spacetype(Bool, x, y, z...) || throw(SpaceMismatch(lazy"incompatible space types $(type_repr.(spacetype.((x, y, z...))))"))
    return spacetype(x)
end

# make ElementarySpace instances behave similar to ProductSpace instances
blocksectors(V::ElementarySpace) = collect(sectors(V))
blockdim(V::ElementarySpace, c::Sector) = dim(V, c)

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
infimum(V₁::S, V₂::S, V₃::S...) where {S <: ElementarySpace} = infimum(infimum(V₁, V₂), V₃...)

"""
    supremum(V₁::ElementarySpace, V₂::ElementarySpace, V₃::ElementarySpace...)

Return the supremum of a number of elementary spaces, i.e. an instance `V::ElementarySpace`
such that `V ≿ V₁`, `V ≿ V₂`, ... and no other `W ≺ V` has this property. This requires
that all arguments have the same value of `isdual( )`, and also the return value `V` will
have the same value.
"""
function supremum(V₁::S, V₂::S, V₃::S...) where {S <: ElementarySpace}
    return supremum(supremum(V₁, V₂), V₃...)
end
