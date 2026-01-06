# Symmetry sectors

```@meta
CurrentModule = TensorKit
```

## Type hierarchy

The fundamental abstract supertype for symmetry sectors is `Sector`:

```@docs
Sector
```

Various concrete subtypes of `Sector` are provided within the TensorKitSectors library:

```@docs
Trivial
AbstractIrrep
ZNIrrep
DNIrrep
U1Irrep
SU2Irrep
CU1Irrep
AbstractGroupElement
ZNElement
FermionParity
FermionNumber
FermionSpin
FibonacciAnyon
IsingAnyon
PlanarTrivial
IsingBimodule
TimeReversed
ProductSector
```

Several more concrete sector types can be found in other packages such as
[SUNRepresentations.jl](https://github.com/QuantumKitHub/SUNRepresentations.jl),
[CategoryData.jl](https://github.com/QuantumKitHub/CategoryData.jl),
[QWignerSymbols.jl](https://github.com/lkdvos/QWignerSymbols.jl), ...:

Some of these types are parameterized by a type parameter that represents a group.
We therefore also provide a number of types to represent groups:

```@docs
TensorKitSectors.Group
TensorKitSectors.AbelianGroup
TensorKitSectors.Cyclic
TensorKitSectors.U₁
TensorKitSectors.CU₁
TensorKitSectors.SU
TensorKitSectors.Dihedral
TensorKitSectors.ProductGroup
```

The following types are used to characterise different properties of the different types
of sectors:

```@docs
FusionStyle
BraidingStyle
UnitStyle
```

Finally, the following auxiliary types are defined to facilitate the implementation
of some of the methods on sectors:

```@docs
TensorKitSectors.SectorValues
TensorKitSectors.SectorProductIterator
```

## Useful constants

The following constants are defined to facilitate obtaining the type associated
with the group elements or the irreducible representations of a given group:

```@docs
Irrep
GroupElement
```

## Methods for characterizing and manipulating `Sector` objects

The following methods can be used to obtain properties such as topological data
of sector objects, or to manipulate them or create related sectors:

```@docs
unit
isunit
leftunit
rightunit
allunits
dual(::Sector)
Nsymbol
⊗
Fsymbol
Rsymbol
Bsymbol
dim(::Sector)
frobenius_schur_phase
frobenius_schur_indicator
twist(::Sector)
Base.isreal(::Type{<:Sector})
TensorKitSectors.sectorscalartype
deligneproduct(::Sector, ::Sector)
```

We have also the following methods that are specific to certain types of sectors
and serve as accessors to their fields:

```@docs
charge
modulus
```

Furthermore, we also have one specific method acting on groups, represented as types

```@docs
×
```

Mapping between sectors and linear indices is only used for sectors `I` for
which `Base.IteratorSize(values(I)) == HasLength()`.
In that case, we map an index `i` to a sector `c` via `c = getindex(values(I), i)`,
and provide an inverse mapping
```@docs
TensorKitSectors.findindex
```

Because we sometimes want to customize the string representation of our sector types,
we also have the following method:

```@docs
TensorKitSectors.type_repr
```

Finally, we provide functionality to compile all revelant methods for a sector:

```@docs
TensorKitSectors.precompile_sector
```

