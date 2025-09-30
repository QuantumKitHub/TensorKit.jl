# Tensors

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

The abstract supertype of all tensors in TensorKit is given by `AbstractTensorMap`:
```@docs
AbstractTensorMap
```

The following concrete subtypes are provided within the TensorKit library:
```@docs
TensorMap
DiagonalTensorMap
AdjointTensorMap
BraidingTensor
```

Of those, `TensorMap` provides the generic instantiation of our tensor concept. It supports
various constructors, which are discussed in the next subsection.

Furthermore, some aliases are provided for convenience:
```@docs
AbstractTensor
Tensor
```

## `TensorMap` constructors

### General constructors

A `TensorMap` with undefined data can be constructed by specifying its domain and codomain:
```@docs
TensorMap{T}(::UndefInitializer, V::TensorMapSpace{S,N₁,N₂}) where {T,S,N₁,N₂}
```

The resulting object can then be filled with data using the `setindex!` method as discussed
below, using functions such as `VectorInterface.zerovector!`, `rand!` or `fill!`, or it can 
be used as an output argument in one of the many methods that accept output arguments, or
in an `@tensor output[...] = ...` expression.

Alternatively, a `TensorMap` can be constructed by specifying its data, codmain and domain
in one of the following ways:
```@docs
TensorMap(data::AbstractDict{<:Sector,<:AbstractMatrix}, V::TensorMapSpace{S,N₁,N₂}) where {S,N₁,N₂}
TensorMap(data::AbstractArray, V::TensorMapSpace{S,N₁,N₂}; tol) where {S<:IndexSpace,N₁,N₂}
```

Finally, we also support the following `Array`-like constructors
```@docs
zeros(::Type, V::TensorMapSpace)
ones(::Type, V::TensorMapSpace)
rand(::Type, V::TensorMapSpace)
randn(::Type, V::TensorMapSpace)
Random.randexp(::Type, V::TensorMapSpace)
```
as well as a `similar` constructor
```@docs
Base.similar(::AbstractTensorMap, args...)
```

### Specific constructors

Additionally, the following methods can be used to construct specific `TensorMap` instances.
```@docs
id
isomorphism
unitary
isometry
```

## `AbstractTensorMap` properties and data access

The following methods exist to obtain type information:

```@docs
Base.eltype(::Type{<:AbstractTensorMap{T}}) where {T}
spacetype(::Type{<:AbstractTensorMap{<:Any,S}}) where {S}
sectortype(::Type{TT}) where {TT<:AbstractTensorMap}
field(::Type{TT}) where {TT<:AbstractTensorMap}
storagetype
blocktype
```

To obtain information about the indices, you can use:
```@docs
space(::AbstractTensorMap, ::Int)
domain
codomain
numin
numout
numind
codomainind
domainind
allind
```

In `TensorMap` instances, all data is gathered in a single `AbstractVector`, which has an internal structure into blocks associated to total coupled charge, within which live subblocks
associated with the different possible fusion-splitting tree pairs.

To obtain information about the structure of the data, you can use:
```@docs
fusionblockstructure(::AbstractTensorMap)
dim(::AbstractTensorMap)
blocksectors(::AbstractTensorMap)
hasblock(::AbstractTensorMap, ::Sector)
fusiontrees(t::AbstractTensorMap)
```

Data can be accessed (and modified) in a number of ways. To access the full matrix block associated with the coupled charges, you can use:
```@docs
block
blocks
```

To access the data associated with a specific fusion tree pair, you can use:
```@docs
Base.getindex(::TensorMap{T,S,N₁,N₂}, ::FusionTree{I,N₁}, ::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}
Base.setindex!(::TensorMap{T,S,N₁,N₂}, ::Any, ::FusionTree{I,N₁}, ::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}
```

For a tensor `t` with `FusionType(sectortype(t)) isa UniqueFusion`, fusion trees are 
completely determined by the outcoming sectors, and the data can be accessed in a more
straightforward way:
```@docs
Base.getindex(::TensorMap, ::Tuple{I,Vararg{I}}) where {I<:Sector}
```

For tensor `t` with `sectortype(t) == Trivial`, the data can be accessed and manipulated
directly as multidimensional arrays:
```@docs
Base.getindex(::AbstractTensorMap)
Base.getindex(::AbstractTensorMap, ::Vararg{SliceIndex})
Base.setindex!(::AbstractTensorMap, ::Any, ::Vararg{SliceIndex})
```

The tensor data can also be filled with random numbers via
```@docs
Random.rand!
Random.randn!
Random.randexp!
```

## `AbstractTensorMap` operations

The operations that can be performed on an `AbstractTensorMap` can be organized into the
following categories:

* *vector operations*: these do not change the `space` or index strucure of a tensor and
  can be straightforwardly implemented on on the full data. All the methods described in
  [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl) are supported. For
  compatibility reasons, we also provide implementations for equivalent methods from
  LinearAlgebra.jl, such as `axpy!`, `axpby!`.

* *index manipulations*: these change (permute) the index structure of a tensor, which
  affects the data in a way that is fully determined by the categorical data of the
  `sectortype` of the tensor.
  
* *(planar) contractions* and *(planar) traces* (i.e., contractions with identity tensors).
  Tensor contractions correspond to a combination of some index manipulations followed by
  a composition or multiplication of the tensors in their role as linear maps.
  Tensor contractions are however of such important and frequency that they require a
  dedicated implementation.

* *tensor factorisations*, which relies on their identification of tensors with linear maps
  between tensor spaces. The factorisations are applied as ordinary matrix factorisations
  to the matrix blocks associated with the coupled charges.

### Index manipulations

A general index manipulation of a `TensorMap` object can be built up by considering some
transformation of the fusion trees, along with a permutation of the stored data. They come
in three flavours, which are either of the type `transform(!)` which are exported, or of the
type `add_transform!`, for additional expert-mode options that allows for addition and
scaling, as well as the selection of a custom backend.

```@docs
permute(::AbstractTensorMap, ::Index2Tuple{N₁,N₂}) where {N₁,N₂}
braid(::AbstractTensorMap, ::Index2Tuple, ::IndexTuple)
transpose(::AbstractTensorMap, ::Index2Tuple)
repartition(::AbstractTensorMap, ::Int, ::Int)
flip(t::AbstractTensorMap, I)
twist(::AbstractTensorMap, ::Int)
insertleftunitspace(::AbstractTensorMap, ::Val{i}) where {i}
insertrightunitspace(::AbstractTensorMap, ::Val{i}) where {i}
removeunitspace(::AbstractTensorMap, ::Val{i}) where {i}
```

```@docs
permute!(::AbstractTensorMap, ::AbstractTensorMap, ::Index2Tuple)
braid!
transpose!
repartition!
twist!
```

```@docs
TensorKit.add_permute!
TensorKit.add_braid!
TensorKit.add_transpose!
```

### Tensor map composition, traces, contractions and tensor products

```@docs
compose(::AbstractTensorMap, ::AbstractTensorMap)
trace_permute!
contract!
⊗(::AbstractTensorMap, ::AbstractTensorMap)
```

## `TensorMap` factorizations

The factorisation methods come in two flavors, namely a non-destructive version where you
can specify an additional permutation of the domain and codomain indices before the
factorisation is performed (provided that `sectorstyle(t)` has a symmetric braiding) as
well as a destructive version The non-destructive methods are given first:

```@docs
leftorth
rightorth
leftnull
rightnull
tsvd
eigh
eig
eigen
isposdef
```

The corresponding destructive methods have an exclamation mark at the end of their name,
and only accept the `TensorMap` object as well as the method-specific algorithm and keyword
arguments.


TODO: document svd truncation types
