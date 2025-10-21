# Fusion trees

```@meta
CurrentModule = TensorKit
```

# Type hierarchy

```@docs
FusionTree
```

## Methods for defining and generating fusion trees

```@docs
fusiontrees(uncoupled::NTuple{N,I}, coupled::I,
                     isdual::NTuple{N,Bool}) where {N,I<:Sector}
```

## Methods for manipulating fusion trees

For manipulating single fusion trees, the following internal methods are defined:
```@docs
insertat
split
merge
elementary_trace
planar_trace(f::FusionTree{I,N}, q1::IndexTuple{N₃}, q2::IndexTuple{N₃}) where {I<:Sector,N,N₃}
artin_braid
braid(f::FusionTree{I,N}, levels::NTuple{N,Int}, p::NTuple{N,Int}) where {I<:Sector,N}
permute(f::FusionTree{I,N}, p::NTuple{N,Int}) where {I<:Sector,N}
```

These can be composed to implement elementary manipulations of fusion-splitting tree pairs,
according to the following methods

```julia
# TODO: add documentation for the following methods
TensorKit.bendright
TensorKit.bendleft
TensorKit.foldright
TensorKit.foldleft
TensorKit.cycleclockwise
TensorKit.cycleanticlockwise
```

Finally, these are used to define large manipulations of fusion-splitting tree pairs, which
are then used in the index manipulation of `AbstractTensorMap` objects. The following methods
defined on fusion splitting tree pairs have an associated definition for tensors.
```@docs
repartition(::FusionTree{I,N₁}, ::FusionTree{I,N₂}, ::Int) where {I<:Sector,N₁,N₂}
transpose(::FusionTree{I}, ::FusionTree{I}, ::IndexTuple{N₁}, ::IndexTuple{N₂}) where {I<:Sector,N₁,N₂}
braid(::FusionTree{I}, ::FusionTree{I}, ::IndexTuple, ::IndexTuple, ::IndexTuple{N₁}, ::IndexTuple{N₂}) where {I<:Sector,N₁,N₂}
permute(::FusionTree{I}, ::FusionTree{I}, ::IndexTuple{N₁}, ::IndexTuple{N₂}) where {I<:Sector,N₁,N₂}
```
