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
fusiontrees(uncoupled::NTuple{N, I}, coupled::I, isdual::NTuple{N, Bool}) where {N, I<:Sector}
```

## Methods for manipulating fusion trees

For manipulating single fusion trees, the following internal methods are defined:
```@docs
insertat
split
merge
elementary_trace
planar_trace(f::FusionTree{I, N}, q::Index2Tuple{N₃, N₃}) where {I, N, N₃}
artin_braid
braid(f::FusionTree{I, N}, p::IndexTuple{N}, levels::IndexTuple{N}) where {I, N}
permute(f::FusionTree{I, N}, p::IndexTuple{N}) where {I, N}
```

These can be composed to implement elementary manipulations of fusion-splitting tree pairs, according to the following methods

```@docs
TensorKit.bendright
TensorKit.bendleft
TensorKit.foldright
TensorKit.foldleft
```

Finally, these are used to define large manipulations of fusion-splitting tree pairs, which are then used in the index manipulation of `AbstractTensorMap` objects.
The following methods defined on fusion splitting tree pairs have an associated definition for tensors.
```@docs
repartition(src::Union{FusionTreePair, FusionTreeBlock}, N::Int)
Base.transpose(src::Union{FusionTreePair, FusionTreeBlock}, p::Index2Tuple)
braid(src::Union{FusionTreePair, FusionTreeBlock}, p::Index2Tuple, levels::Index2Tuple)
permute(src::Union{FusionTreePair, FusionTreeBlock}, p::Index2Tuple)
```
