# [Vector spaces](@id s_libvectorspaces)

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

The following types are defined to characterise vector spaces and their properties:

```@docs
Field
VectorSpace
ElementarySpace
GeneralSpace
CartesianSpace
ComplexSpace
GradedSpace
CompositeSpace
ProductSpace
HomSpace
```

together with the following specific type for encoding the inner product structure of
a space:

```@docs
InnerProductStyle
```

## Useful constants

The following constants are defined to easily create the concrete type of `GradedSpace`
associated with a given type of sector.

```@docs
Vect
Rep
```

In this respect, there are also a number of type aliases for the `GradedSpace` types
associated with the most common sectors, namely

```julia
const ZNSpace{N} = Vect[ZNIrrep{N}]
const Z2Space = ZNSpace{2}
const Z3Space = ZNSpace{3}
const Z4Space = ZNSpace{4}
const U1Space = Rep[U₁]
const CU1Space = Rep[CU₁]
const SU2Space = Rep[SU₂]

# Unicode alternatives
const ℤ₂Space = Z2Space
const ℤ₃Space = Z3Space
const ℤ₄Space = Z4Space
const U₁Space = U1Space
const CU₁Space = CU1Space
const SU₂Space = SU2Space
```

## [Methods](@id s_spacemethods)

Methods often apply similar to e.g. spaces and corresponding tensors or tensor maps, e.g.:

```@docs
field
sectortype
sectors
hassector
dim(::VectorSpace)
dim(::ElementarySpace, ::Sector)
reduceddim
dim(P::ProductSpace{<:ElementarySpace,N}, sector::NTuple{N,<:Sector}) where {N}
dim(::HomSpace)
dims
blocksectors(P::ProductSpace{S,N}) where {S,N}
blocksectors(::HomSpace)
hasblock
blockdim
fusiontrees(P::ProductSpace{S,N}, blocksector::I) where {S,N,I}
space
```

The following methods act specifically on `ElementarySpace` spaces:

```@docs
dual(::VectorSpace)
conj
isconj
isdual
flip
zerospace
unitspace
⊕
⊖
supremum
infimum
```

while the following also work on both `ElementarySpace` and `ProductSpace`

```@docs
fuse
one(::VectorSpace)
⊗(::VectorSpace, ::VectorSpace)
⊠(::VectorSpace, ::VectorSpace)
ismonomorphic
isepimorphic
isisomorphic
```

Inserting trivial space factors or removing such factors for `ProductSpace` instances
can be done with the following methods.
```@docs
insertleftunit(::ProductSpace, ::Val{i}) where {i}
insertrightunit(::ProductSpace, ::Val{i}) where {i}
removeunit(::ProductSpace, ::Val{i}) where {i}
```

There are also specific methods for `HomSpace` instances, that are used in determining
the resuling `HomSpace` after applying certain tensor operations.

```@docs
flip(W::HomSpace{S}, I) where {S}
TensorKit.permute(::HomSpace, ::Index2Tuple)
TensorKit.select(::HomSpace{S}, ::Index2Tuple{N₁,N₂}) where {S,N₁,N₂}
TensorKit.compose(::HomSpace{S}, ::HomSpace{S}) where {S}
insertleftunit(::HomSpace, ::Val{i}) where {i}
insertrightunit(::HomSpace, ::Val{i}) where {i}
removeunit(::HomSpace, ::Val{i}) where {i}
```
