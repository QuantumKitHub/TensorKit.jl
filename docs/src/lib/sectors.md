# Symmetry sectors and fusion trees

```@meta
CurrentModule = TensorKit
```

## Type hierarchy
```@docs
Sector
SectorValues
FusionStyle
BraidingStyle
AbstractIrrep
Trivial
ZNIrrep
U1Irrep
SU2Irrep
CU1Irrep
ProductSector
FermionParity
FermionNumber
FermionSpin
FibonacciAnyon
IsingAnyon
```

## Useful constants
```@docs
Irrep
```

## Methods for defining and characterizing `Sector` subtypes
```@docs
unit(::Sector)
dual
Nsymbol
⊗
Fsymbol
Rsymbol
Bsymbol
dim(::Sector)
frobenius_schur_phase
twist(::Sector)
Base.isreal(::Type{<:Sector})
TensorKitSectors.sectorscalartype
deligneproduct(::Sector, ::Sector)
```

Compile all revelant methods for a sector:

```@docs
TensorKitSectors.precompile_sector
```


## Types and methods for groups

Types and constants:

```julia
# TODO: add documentation for the following types
Group
TensorKitSectors.AbelianGroup
U₁
ℤ{N} where N
SU{N} where N
const SU₂ = SU{2}
ProductGroup
```

Specific methods:

```@docs
×
```


## Methods for defining and generating fusion trees
```@docs
FusionTree
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
