# [Sectors](@id ss_sectors)

```@setup sectors
using TensorKit
using TensorKitSectors
```

The first ingredient in order to define and construct symmetric tensors,
is a framework to define symmetry sectors and their assocated fusion rules and topological data.
[TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl) defines an abstract supertype `Sector` that all sectors will be subtypes of
```julia
abstract type Sector end
```
Any concrete subtype of `Sector` should be such that its instances represent a consistent set of sectors,
corresponding to the irreps of some group, or, more generally,
the simple objects of a (unitary) fusion category. 
Throughout TensorKit.jl,
the method [`sectortype`](@ref) can be used to query the subtype of `Sector` associated with a particular object,
i.e. a vector space, fusion tree, tensor map, or a sector.
It works on both instances and in the type domain, and its use will be illustrated further on.

## [Minimal sector interface](@id ss_sectorinterface)

The minimal data to completely specify a type of sector closely matches the [topological data](@ref ss_topologicalfusion) of a [fusion category](@ref ss_fusion) as reviewed in the appendix on [category theory](@ref s_categories), and is given by:
*   The fusion rules, i.e. `` a ⊗ b = ⨁ N^{ab}_{c} c ``,
    implemented as the function [`Nsymbol(a, b, c)`](@ref).
*   The list of fusion outputs from ``a ⊗ b``;
    while this information is contained in ``N^{ab}_c``,
    it might be costly or impossible to iterate over all possible values of `c` and test `Nsymbol(a,b,c)`;
    instead we require for [`a ⊗ b`](@ref), or equivalently, `otimes(a, b)`, to return an iterable object
    (e.g. tuple or array, but see [below](@ref ss_sectoradditionaltools) for a dedicated iterator struct)
    that generates all *unique* `c` for which ``N^{ab}_c ≠ 0`` (so only once for all ``c`` with ``N^{ab}_c ≥ 1``).
*   The identity object `u`, such that ``a ⊗ u = a = u ⊗ a``,
    implemented as the function [`unit(a)`](@ref) (and also in type domain),
    but `one(a)` from Julia Base also works as an alias to `unit(a)`.
*   The dual or conjugate object ``\overline{a}`` for which ``N^{a\bar{a}}_{u} = 1``,
    implemented as the function [`dual(a)`](@ref).
    Because we restrict to unitary categories,
    `conj(a)` from the Julia `Base` library is also defined as an alias to `dual(a)`.
*   The F-symbol or recoupling coefficients ``[F^{abc}_{d}]^f_e``;
    implemented as the function [`Fsymbol(a, b, c, d, e, f)`](@ref).
*   If the category is braided (see below), the R-symbol ``R^{ab}_c``;
    implemented as the function [`Rsymbol(a, b, c)`](@ref).

Furthermore, sectors should provide information about the structure of their fusion rules.
For irreps of Abelian groups, we have that for every ``a`` and ``b``,
there exists a unique ``c`` such that ``a ⊗ b = c``, i.e. there is only a single fusion channel.
This follows simply from the fact that all irreps are one-dimensional.
In all other cases, there is at least one pair of (``a``, ``b``) exists such that ``a ⊗ b`` has multiple fusion outputs.
This is often referred to as non-abelian fusion,
and is the case for the irreps of a non-abelian group or some more general fusion category.
We however still distinguish between the case where all entries of ``N^{ab}_c ≦ 1``,
i.e. they are zero or one.
In that case, ``[F^{abc}_{d}]^f_e`` and ``R^{ab}_c`` are scalars.
If some ``N^{ab}_c > 1``,
it means that the same sector ``c`` can appear more than once in the fusion product of ``a`` and ``b``,
and we need to introduce some multiplicity label ``μ`` for the different copies,
and ``[F^{abc}_{d}]^f_e`` and ``R^{ab}_c`` are respectively 4- and 2-dimensional arrays labelled by these multiplicity indices.
To encode these different possibilities,
we define a Holy-trait called [`FusionStyle`](@ref), i.e. a type hierarchy 
```julia
abstract type FusionStyle end
struct UniqueFusion <: FusionStyle end # unique fusion output when fusing two sectors
abstract type MultipleFusion <: FusionStyle end
struct SimpleFusion <: MultipleFusion end # multiple fusion but multiplicity free
struct GenericFusion <: MultipleFusion end # multiple fusion with multiplicities
const MultiplicityFreeFusion = Union{UniqueFusion, SimpleFusion}
```
New sector types `I<:Sector` should then indicate which fusion style they have by defining `FusionStyle(::Type{I})`.

In a similar manner,
it is useful to distinguish between the structure and the different styles of the braiding of a sector type.
Remember that for group representations, braiding acts as swapping or permuting the vector spaces involved.
By definition, applying this operation twice leads us back to the original situation.
If that is the case, the braiding is said to be symmetric.
For more general fusion categories, associated with the physics of anyonic particles,
this is generally not the case.
Some categories do not even support a braiding rule,
as this requires at least that ``a ⊗ b`` and ``b ⊗ a`` have the same fusion outputs for every ``a`` and ``b``.
When braiding is possible, it might not be symmetric, and as a result,
permutations of tensor indices are not unambiguously defined.
The correct description is in terms of the braid group.
This will be discussed in more detail below.
Fermions are somewhat in between, as their braiding is symmetric, but they have a non-trivial *twist*.
We thereto define a new trait [`BraidingStyle`](@ref) with associated the type hierarchy
```julia
abstract type HasBraiding <: BraidingStyle end
struct NoBraiding <: BraidingStyle end
abstract type SymmetricBraiding <: HasBraiding end # symmetric braiding => actions of permutation group are well defined
struct Bosonic <: SymmetricBraiding end # all twists are one
struct Fermionic <: SymmetricBraiding end # twists one and minus one
struct Anyonic <: HasBraiding end
```
New sector types `I<:Sector` should then indicate which fusion style they have by defining `BraidingStyle(::Type{I})`.

Note that `Bosonic()` braiding does not mean that all permutations are trivial and ``R^{ab}_c = 1``,
but that ``R^{ab}_c R^{ba}_c = 1``.
For example, for the irreps of ``\mathsf{SU}_2``,
the R-symbol associated with the fusion of two spin-1/2 particles to spin zero is ``-1``,
i.e. the singlet of two spin-1/2 particles is antisymmetric under swapping the two constituents.
For a `Bosonic()` braiding style, all twists are simply ``+1``.
The case of fermions and anyons are discussed below.

For practical reasons, we also require some additional methods to be defined:
*   `hash(a, h)` creates a hash of sectors,
    because sectors and objects created from them are used as keys in lookup tables (i.e. dictionaries).
    Julia provides a default implementation of `hash` for every new type, but it can be useful
    to overload it for efficiency, or to ensure that the same hash is obtained for different instances
    that represent the same sector (e.g. when the sector type is not a bitstype).
*   `isless(a, b)` associates a canonical order to sectors (of the same type),
    in order to unambiguously represent representation spaces ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``.

Lastly, we sometimes need to iterate over different values of a sector type `I<:Sector`,
or at least have some basic information about the number of possible values of `I`,
Hereto, TensorKitSectors.jl defines `Base.values(I::Type{<:Sector})` to return the singleton instance of the parametric type [`SectorValues{I}`](@ref),
which should behave as an iterator over all possible values of the sector type `I`.
This means the following methods should be implemented for a new sector type `I<:Sector`:

*   `Base.iterate(::Type{SectorValues{I}} [, state])` should implement the iterator interface 
    so as to enable iterating over all values of the sector `I` according to the canonical order defined by `isless`.
*   `Base.IteratorSize(::Type{SectorValues{I}})` should return `HasLength()` if the number of different values
    of sector `I` is finite and rather small, and `SizeUnknown()` or `IsInfinite()` otherwise.
    This is used to encode the degeneracies of the different sectors in a `GradedSpace` object efficiently,
    as discussed in the next section on [Graded spaces](@ref ss_rep).
*   If `IteratorSize(::Type{SectorValues{I}}) == HasLength()`,
    then `Base.length(::Type{SectorValues{I}})` should return the number of different values of sector `I`.

Furthermore, the standard definitions `Base.IteratorEltype(::Type{SectorValues{I}}) = HasEltype()` and
`Base.eltype(::Type{SectorValues{I}}) = I` are provided by default in TensorKitSectors.jl.

!!! note
    A recent update in TensorKitSectors.jl has extended the minimal interface to also support multi-fusion categories,
    for which in particular the unit object is non-simple.
    We do not discuss this extension here, but refer to the documentation of [`UnitStyle`](@ref),
    [`leftunit`](@ref), [`rightunit`](@ref) and [`allunits`](@ref) for more details.

## [Additional methods](@id ss_sectoradditional)

The sector interface contains a number of additional methods, that are useful,
but whose return value can be computed from the minimal interface defined in the previous subsection.
However, new sector types can override these default fallbacks with more efficient implementations.

Firstly, the canonical order of sectors allows to enumerate the different values,
and thus to associate each value with an integer.
Hereto, the following methods are defined:

*   `Base.getindex(::SectorValues{I}, i::Int)`:
    returns the sector instance of type `I` that is associated with integer `i`.
    The fallback implementation simply iterates through `values(I)` up to the `i`th value.
*   `findindex(::SectorValues{I}, c::I)`:
    reverse mapping that associates an index `i::Integer ∈ 1:length(values(I))` to a given sector `c::I`.
    The fallback implementation simply searches linearly through the `values(I)` iterator.

Note that `findindex` acts similar to `Base.indexin`,
but with the order of the arguments reversed (so that is more similar to `getindex`),
and returns an `Int` rather than an `Array{0,Union{Int,Nothing}}`.

Secondly, it is often useful to know the scalar type in which the topological data in the F- and R-symbols are expressed.
For this, the method [`sectorscalartype(I::Type{<:Sector})`](@ref) is provided,
which has a default implementation that uses type inference on the return values of `Fsymbol` and `Rsymbol`.
This function is also used to define `Base.isreal(I::Type{<:Sector})`, 
which indicates whether all topological data are real numbers.
This is important because, if complex numbers appear in the topological data,
it means tensor data will necessarily become complex after simple manipulations such as permuting indices,
and should therefore probably be stored as complex numbers from the start.

Finally, additional topological data can be extracted from the minimal interface.
In particular, the quantum dimensions ``d_a`` and 
Frobenius-Schur phase ``χ_a`` and indicator (only if ``a == \overline{a}``) are encoded in the F-symbol.
They are obtained as [`dim(a)`](@ref), [`frobenius_schur_phase(a)`](@ref) and [`frobenius_schur_indicator(a)`](@ref).
These functions have default definitions which compute the requested data from `Fsymbol(a, conj(a), a, a, one(a), one(a))`,
but they can be overloaded in case the value can be computed more efficiently.
The same holds for related fusion manipulations such as the B-symbol,
which is obtained as [`Bsymbol(a, b, c)`](@ref).
Finally, the twist associated with a sector `a` is obtained as [`twist(a)`](@ref),
which also has a default implementation in terms of the R-symbol.
In addition, tThe function `isunit` is provided to facilitate checking whether a sector is a unit sector,
in particular for the non-trivial case of the multi-fusion category case, which we do not discuss here.

## [Additional tools](@id ss_sectoradditionaltools)

The fusion product `a ⊗ b` of two sectors `a` and `b` is required to return an iterable object
that generates all unique fusion outputs `c` for which ``N^{ab}_c ≥ 0``.
When this list can easily be computed or constructed, it can be returned as a tuple or an array.
However, when taking type stability and (memory) efficiency into account,
it is often preferable to return a lazy iterator object that generates the different fusion outputs on the fly.
Indeed, a tuple result is only type stable when the number of fusion outputs is constant for all possible inputs `a` and `b`,
whereas a `Vector` result requires heap allocation.

By default, [TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl) defines
```julia
⊗(a::I, b::I) where {I <: Sector} = SectorProductIterator(a, b)
```
where [`TensorKitSectors.SectorProductIterator`](@ref) is defined as
```julia
struct SectorProductIterator{I <: Sector}
    a::I
    b::I
end
```
can serve as a general iterator type.
For defining the fusion rules of a sector `I`, 
instead of implementing `⊗(::I, ::I)` directly,
it is thus possible to instead implement the iterator interface for `SectorProductIterator{I}`,
i.e. provide definitions for
*   `Base.iterate(::SectorProductIterator{I}[, state])`
*   `Base.IteratorSize(::Type{SectorProductIterator{I}})`
*   `Base.length(::SectorProductIterator{I})` (if applicable)

[TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl) already defines
```julia
Base.eltype(::Type{SectorProductIterator{I}}) where {I} = I
```
and sets `Base.IteratorEltype(::Type{SectorProductIterator{I}})` accordingly.
Furthermore, it provides custom pretty printing, so that `SectorProductIterator{I}(a, b)` is displayed as `a ⊗ b`.

## [Group representations](@id ss_groups)

In this subsection, we give an overview of some existing sector types provided by [TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl).
We also discuss the implementation of some of them in more detail, in order to illustrate the interface defined above.

The first sector type is called `Trivial`,
and corresponds to the case where there is actually no symmetry, or thus,
the symmetry is the trivial group with only an identity operation and a trivial representation.
Its representation theory is particularly simple:
```julia
struct Trivial <: Sector end

# basic properties
unit(::Type{Trivial}) = Trivial()
dual(::Trivial) = Trivial()
Base.isless(::Trivial, ::Trivial) = false

# fusion rules
⊗(::Trivial, ::Trivial) = (Trivial(),)
Nsymbol(::Trivial, ::Trivial, ::Trivial) = true
FusionStyle(::Type{Trivial}) = UniqueFusion()
Fsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1

# braiding rules
Rsymbol(::Trivial, ::Trivial, ::Trivial) = 1
BraidingStyle(::Type{Trivial}) = Bosonic()

# values iterator
Base.IteratorSize(::Type{SectorValues{Trivial}}) = HasLength()
Base.length(::SectorValues{Trivial}) = 1
Base.iterate(::SectorValues{Trivial}, i = false) = return i ? nothing : (Trivial(), true)
function Base.getindex(::SectorValues{Trivial}, i::Int)
    return i == 1 ? Trivial() : throw(BoundsError(values(Trivial), i))
end
findindex(::SectorValues{Trivial}, c::Trivial) = 1
```
The `Trivial` sector type is special cased in the construction of tensors,
so that most of these definitions are not actually used.

The most important class of sectors are irreducible representations of groups.
As we often use the group itself as a type parameter,
an associated type hierarchy for groups has been defined, namely
```julia
abstract type Group end
abstract type AbelianGroup <: Group end

abstract type Cyclic{N} <: AbelianGroup end
abstract type Dihedral{N} <: Group end
abstract type U₁ <: AbelianGroup end
abstract type CU₁ <: Group end

const ℤ{N} = Cyclic{N}
const ℤ₂ = ℤ{2}
const ℤ₃ = ℤ{3}
const ℤ₄ = ℤ{4}
const D₃ = Dihedral{3}
const D₄ = Dihedral{4}
const SU₂ = SU{2}
```
Groups themselves are abstract types without any functionality (at least for now).
However, as will become clear instantly, it is useful to identify abelian groups,
because their representation theory is particularly simple.
We also provide a number of convenient Unicode aliases.
These group names are probably self-explanatory, except for `CU₁` which is explained below.

Irreps of groups will then be defined as subtypes of the abstract type
```julia
abstract type AbstractIrrep{G<:Group} <: Sector end # irreps have integer quantum dimensions
BraidingStyle(::Type{<:AbstractIrrep}) = Bosonic()
```

We will need different data structures to represent irreps of different groups,
but it would be convenient to easily obtain the relevant structure for a given group `G` in a uniform manner.
Hereto, we define a singleton type `IrrepTable` with an associated exported constant `Irrep = IrrepTable()` as the only instance.
When a concrete type for representing the the irreps of a certain group `G` is iplemented,
this type can the be "discovered" or obtained as `Irrep[G]`,
provided it was registered by defining `Base.getindex(::IrrepTable, ::Type{G})` to return the concrete type.

Furthermore, we combine the more common functionality for irreps of abelian groups
```julia
const AbelianIrrep{G} = AbstractIrrep{G} where {G <: AbelianGroup}
FusionStyle(::Type{<:AbelianIrrep}) = UniqueFusion()
Base.sectorscalartype(::Type{<:AbelianIrrep}) = Int

Nsymbol(a::I, b::I, c::I) where {I <: AbelianIrrep} = c == first(a ⊗ b)
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I <: AbelianIrrep}
    return Int(Nsymbol(a, b, e) * Nsymbol(e, c, d) * Nsymbol(b, c, f) * Nsymbol(a, f, d))
end
frobenius_schur_phase(a::AbelianIrrep) = 1
Asymbol(a::I, b::I, c::I) where {I <: AbelianIrrep} = Int(Nsymbol(a, b, c))
Bsymbol(a::I, b::I, c::I) where {I <: AbelianIrrep} = Int(Nsymbol(a, b, c))
Rsymbol(a::I, b::I, c::I) where {I <: AbelianIrrep} = Int(Nsymbol(a, b, c))
```

With these common definition in place,
we implement the representation theory of the most common Abelian groups,
starting with ``\mathsf{U}_1``, the full implementation of which is given by

```julia
struct U1Irrep <: AbstractIrrep{U₁}
    charge::HalfInt
end
Base.getindex(::IrrepTable, ::Type{U₁}) = U1Irrep
Base.convert(::Type{U1Irrep}, c::Real) = U1Irrep(c)

# basic properties
charge(c::U1Irrep) = c.charge
unit(::Type{U1Irrep}) = U1Irrep(0)
dual(c::U1Irrep) = U1Irrep(-charge(c))
@inline function Base.isless(c1::U1Irrep, c2::U1Irrep)
    return isless(abs(charge(c1)), abs(charge(c2))) || zero(HalfInt) < charge(c1) == -charge(c2)
end

# fusion rules
⊗(c1::U1Irrep, c2::U1Irrep) = (U1Irrep(charge(c1) + charge(c2)),)

# values iterator
Base.IteratorSize(::Type{SectorValues{U1Irrep}}) = IsInfinite()
function Base.iterate(::SectorValues{U1Irrep}, i::Int = 0)
    return i <= 0 ? (U1Irrep(half(i)), (-i + 1)) : (U1Irrep(half(i)), -i)
end
function Base.getindex(::SectorValues{U1Irrep}, i::Int)
    i < 1 && throw(BoundsError(values(U1Irrep), i))
    return U1Irrep(iseven(i) ? half(i >> 1) : -half(i >> 1))
end
function findindex(::SectorValues{U1Irrep}, c::U1Irrep)
    return (n = twice(charge(c)); 2 * abs(n) + (n <= 0))
end

# hashing
Base.hash(c::U1Irrep, h::UInt) = hash(c.charge, h)
```

A few comments are in order:
The `getindex` definition just below the type definition provides the mechanism to obtain `U1Irrep` as `Irrep[U₁]`, as discussed above.
The `Base.convert` definition, while not required by the minimal sector interface,
allows to convert real numbers to the corresponding type of sector,
and thus to omit the type information of the sector whenever this is clear from the context.
The `charge` function is again not part of the minimal sector interface,
and is specific to `U1Irrep` (and `ZNIrrep` discussed next),
as a mere convenience function to access the charge value.
Finally, in the definition of `U1Irrep`,
`HalfInt<:Number` is a Julia type defined in [HalfIntegers.jl](https://github.com/sostock/HalfIntegers.jl),
which is also used for `SU2Irrep` below,
that stores integer or half integer numbers using twice their value.
Strictly speaking, the linear representations of `U₁` can only have integer charges,
and fractional charges lead to a projective representation.
It can be useful to allow half integers in order to describe spin 1/2 systems with an axis rotation symmetry.
As a user, you should not worry about the details of `HalfInt`
and additional methods for automatic conversion and pretty printing are provided,
as illustrated by the following example
```@repl sectors
Irrep[U₁](0.5)
U1Irrep(0.4)
U1Irrep(1) ⊗ Irrep[U₁](1//2)
u = first(U1Irrep(1) ⊗ Irrep[U₁](1//2))
Nsymbol(u, conj(u), one(u))
```

We similarly implement the irreps of the finite cyclic groups ``\mathbb{Z}_N``,
where we distinguish between small and large values of `N` to optimize storage.
The implementation is given by
```julia
const SMALL_ZN_CUTOFF = (typemax(UInt8) + 1) ÷ 2
struct ZNIrrep{N} <: AbstractIrrep{ℤ{N}}
    n::UInt8
    function ZNIrrep{N}(n::Integer) where {N}
        N ≤ SMALL_ZN_CUTOFF || throw(DomainError(N, "N exceeds the maximal value, use `LargeZNIrrep` instead"))
        return new{N}(UInt8(mod(n, N)))
    end
end
struct LargeZNIrrep{N} <: AbstractIrrep{ℤ{N}}
    n::UInt
    function LargeZNIrrep{N}(n::Integer) where {N}
        N ≤ (typemax(UInt) ÷ 2) || throw(DomainError(N, "N exceeds the maximal value"))
        return new{N}(UInt(mod(n, N)))
    end

end
Base.getindex(::IrrepTable, ::Type{ℤ{N}}) where {N} = N ≤ SMALL_ZN_CUTOFF ? ZNIrrep{N} : LargeZNIrrep{N}
...
```
and continues along simular lines of the `U1Irrep` implementation above,
by replacing the arithmetic with modulo `N` arithmetic.

The storage benefits for small `N` are not only due to a smaller integer type in the sector itself,
but emerges as a result of the following distinction in the iterator size:
```julia
Base.IteratorSize(::Type{SectorValues{<:ZNIrrep}}) = HasLength()
Base.IteratorSize(::Type{SectorValues{<:LargeZNIrrep}}) = SizeUnknown()
```
As a result, the `GradedSpace` implementation (see next section on [Graded spaces](@ref ss_rep))
to store general direct sum objects ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}`` will use a very different internal representation for those two cases.

We furthermore define some aliases for the first (and most commonly used `ℤ{N}` irreps)
```julia
const Z2Irrep = ZNIrrep{2}
const Z3Irrep = ZNIrrep{3}
const Z4Irrep = ZNIrrep{4}
```
which we can illustrate via
```@repl sectors
z = Z3Irrep(1)
ZNIrrep{3}(1) ⊗ Irrep[ℤ₃](1)
dual(z)
unit(z)
```

As a final remark on the irreps of abelian groups,
note that even though `a ⊗ b` is equivalent to a single new label `c`,
we return this result as an iterable container, in this case a one-element tuple `(c,)`.

The first example of irreps of a non-abelian group is that of ``\mathsf{SU}_2``,
the implementation of which is summarized by
```julia
struct SU2Irrep <: AbstractIrrep{SU₂}
    j::HalfInt
    function SU2Irrep(j)
        j >= zero(j) || error("Not a valid SU₂ irrep")
        return new(j)
    end
end
Base.getindex(::IrrepTable, ::Type{SU₂}) = SU2Irrep
Base.convert(::Type{SU2Irrep}, j::Real) = SU2Irrep(j)

# basic properties
const _su2one = SU2Irrep(zero(HalfInt))
unit(::Type{SU2Irrep}) = _su2one
dual(s::SU2Irrep) = s
dim(s::SU2Irrep) = twice(s.j) + 1
Base.isless(s1::SU2Irrep, s2::SU2Irrep) = isless(s1.j, s2.j)

# fusion product iterator
const SU2IrrepProdIterator = SectorProductIterator{SU2Irrep}
Base.IteratorSize(::Type{SU2IrrepProdIterator}) = Base.HasLength()
Base.length(it::SU2IrrepProdIterator) = length(abs(it.a.j - it.b.j):(it.a.j + it.b.j))
function Base.iterate(it::SU2IrrepProdIterator, state = abs(it.a.j - it.b.j))
    return state > (it.a.j + it.b.j) ? nothing : (SU2Irrep(state), state + 1)
end

# fusion and braidingdata
FusionStyle(::Type{SU2Irrep}) = SimpleFusion()
sectorscalartype(::Type{SU2Irrep}) = Float64

Nsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep) = WignerSymbols.δ(sa.j, sb.j, sc.j)
function Fsymbol(
        s1::SU2Irrep, s2::SU2Irrep, s3::SU2Irrep,
        s4::SU2Irrep, s5::SU2Irrep, s6::SU2Irrep
    )
    if all(==(_su2one), (s1, s2, s3, s4, s5, s6))
        return 1.0
    else
        return sqrtdim(s5) * sqrtdim(s6) *
            WignerSymbols.racahW(
            sectorscalartype(SU2Irrep), s1.j, s2.j, s4.j, s3.j,
            s5.j, s6.j
        )
    end
end
function Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)
    Nsymbol(sa, sb, sc) || return zero(sectorscalartype(SU2Irrep))
    return iseven(convert(Int, sa.j + sb.j - sc.j)) ? one(sectorscalartype(SU2Irrep)) :
        -one(sectorscalartype(SU2Irrep))
end

# values iterator
Base.IteratorSize(::Type{SectorValues{SU2Irrep}}) = IsInfinite()
Base.iterate(::SectorValues{SU2Irrep}, i::Int = 0) = (SU2Irrep(half(i)), i + 1)
function Base.getindex(::SectorValues{SU2Irrep}, i::Int)
    return 1 <= i ? SU2Irrep(half(i - 1)) : throw(BoundsError(values(SU2Irrep), i))
end
findindex(::SectorValues{SU2Irrep}, s::SU2Irrep) = twice(s.j) + 1

# hashing
Base.hash(s::SU2Irrep, h::UInt) = hash(s.j, h)
```
and some methods for pretty printing and converting from real numbers to irrep labels.
Here, the fusion rules are implemented lazily using the `SectorProductIterator` defined above.
Furthermore, the topological data (i.e. `Nsymbol` and `Fsymbol`) are provided by the package [WignerSymbols.jl](https://github.com/Jutho/WignerSymbols.jl).
Note that, while WignerSymbols.jl is able to generate the required data in arbitrary precision,
we have explicitly restricted the scalar type of `SU2Irrep` to `Float64` for efficiency.

The following example illustrates the usage of `SU2Irrep`
```@repl sectors
s = SU2Irrep(3//2)
conj(s)
dim(s)
collect(s ⊗ s)
for s2 in s ⊗ s
    @show s2
    @show Nsymbol(s, s, s2)
    @show Rsymbol(s, s, s2)
end
```

Other non-abelian groups for which the irreps are implemented are the dihedral groups ``\mathsf{D}_N``,
and the semidirect product ``\mathsf{U}₁ ⋉ ℤ_2``.
In the context of quantum systems, the latter occurs in the case of systems with particle hole symmetry and
the non-trivial element of ``ℤ_2`` acts as charge conjugation ``C``.
It has the effect of interchaning ``\mathsf{U}_1`` irreps ``n`` and ``-n``,
and turns them together in a joint 2-dimensional index, except for the case ``n=0``.
Irreps are therefore labeled by integers ``n ≧ 0``,
however for ``n=0`` the ``ℤ₂`` symmetry can be realized trivially or non-trivially,
resulting in an even and odd one-dimensional irrep with ``\mathsf{U})_1`` charge ``0``.
Given ``\mathsf{U}_1 ≂ \mathsf{SO}_2``, this group is also simply known as ``\mathsf{O}_2``,
and the two representations with `` n = 0`` are the scalar and pseudo-scalar, respectively.
However, because we also allow for half integer representations, we refer to it as `Irrep[CU₁]` or `CU1Irrep` in full.
```julia
struct CU1Irrep <: AbstractIrrep{CU₁}
    j::HalfInt # value of the U1 charge
    s::Int # rep of charge conjugation:
    # if j == 0, s = 0 (trivial) or s = 1 (non-trivial),
    # else s = 2 (two-dimensional representation)
    # Let constructor take the actual half integer value j
    function CU1Irrep(j::Real, s::Int = ifelse(j>zero(j), 2, 0))
        if ((j > zero(j) && s == 2) || (j == zero(j) && (s == 0 || s == 1)))
            new(j, s)
        else
            error("Not a valid CU₁ irrep")
        end
    end
end

unit(::Type{CU1Irrep}) = CU1Irrep(zero(HalfInt), 0)
dual(c::CU1Irrep) = c
dim(c::CU1Irrep) = ifelse(c.j == zero(HalfInt), 1, 2)

FusionStyle(::Type{CU1Irrep}) = SimpleFusion()
...
```
The rest of the implementation can be read in the source code,
but is rather long due to all the different cases for the arguments of `Fsymbol`.
For the dihedrial groups ``\mathsf{D}_N``, which can be intepreted as the semidirect product ``\mathbb{Z}_N ⋉ ℤ_2``,
the representation theory is obtained quite similarly, and is implmented as the type [`DNIrrep{N}`](@ref).

By default,
none of the groups mentioned above have a reprenenstation theory for which `FusionStyle(I) == GenericFusion()`,
i.e. where fusion mulitplicities are required.
An example where this does appear is for the irreps of `SU{N}` for `N>2`.
Such sectors are supported through [SUNRepresentations.jl](https://github.com/QuantumKitHub/SUNRepresentations.jl),
which implements numerical routines to compute the topological data of the representation theory of these groups,
as no general analytic formula is available.

## [Combining different sectors](@id ss_productsectors)

It is also possible to combine two or more different types of symmetry sectors,
e.g. when the total symmetry group is a direct product of individual simple groups.
Such combined sectors are obtained using the binary operator `⊠`, which can be entered as `\boxtimes`+TAB.
The resulting type is called [`ProductSector`](@ref), which simply wraps the individual sectors,
but knows how to combine their fusion and braiding data correctly.
First some examples
```@repl sectors
a = Z3Irrep(1) ⊠ Irrep[U₁](1)
typeof(a)
conj(a)
one(a)
dim(a)
collect(a ⊗ a)
FusionStyle(a)
b = Irrep[ℤ₃](1) ⊠ Irrep[SU₂](3//2)
typeof(b)
conj(b)
one(b)
dim(b)
collect(b ⊗ b)
FusionStyle(b)
c = Irrep[SU₂](1) ⊠ SU2Irrep(3//2)
typeof(c)
conj(c)
one(c)
dim(c)
collect(c ⊗ c)
FusionStyle(c)
```
We refer to the source file of [`ProductSector`](@ref) for implementation details.

The symbol `⊠` refers to the [Deligne tensor product](https://ncatlab.org/nlab/show/Deligne+tensor+product+of+abelian+categories) within the literature on category theory.
Indeed, the category of representation of a product group `G₁ × G₂` corresponds to
the Deligne tensor product of the categories of representations of the two groups separately.
But this definition also extends to other categories which are not associated with the representation theory of a group, as discussed below.
Note that `⊠` also works in the type domain, i.e. `Irrep[ℤ₃] ⊠ Irrep[CU₁]` can be used to create `ProductSector{Tuple{Irrep[ℤ₃], Irrep[CU₁]}}`.
Instances of this type can be constructed by giving a number of arguments,
where the first argument is used to construct the first sector, and so forth.
Furthermore, for representations of groups, we also enabled the notation `Irrep[ℤ₃ × CU₁]`,
with `×` obtained using `\times+TAB`.
However, this is merely for convience,
as `Irrep[ℤ₃] ⊠ Irrep[CU₁]` is not a subtype of the abstract type `AbstractIrrep{ℤ₃ × CU₁}`.
As is often the case with the Julia type system,
the purpose of subtyping `AbstractIrrep` was to share common functionality and
thereby simplify the implementation of irreps of the different groups discussed above,
but not to express a mathematical hierarchy.

Some more examples:
```@repl sectors
a = Z3Irrep(1) ⊠ Irrep[CU₁](1.5)
a isa Irrep[ℤ₃] ⊠ CU1Irrep
a isa Irrep[ℤ₃ × CU₁]
a isa AbstractIrrep{ℤ₃ × CU₁}
a == Irrep[ℤ₃ × CU₁](1, 1.5)
```

## [Defining a new type of sector](@id ss_newsectors)

By now, it should be clear how to implement a new `Sector` subtype.
Ideally, a new `I<:Sector` type is a `struct I ... end` (immutable) that has `isbitstype(I) == true` (see Julia's manual),
and implements the following minimal set of methods
```julia
TensorKit.unit(::Type{I}) = I(...)
TensorKit.dual(a::I) = I(...)
Base.isless(a::I, b::I)

TensorKit.FusionStyle(::Type{I}) = ... # UniqueFusion(), SimpleFusion(), GenericFusion()
TensorKit.Nsymbol(a::I, b::I, c::I) = ... # Bool or Integer if FusionStyle(I) == GenericFusion()

TensorKit.:⊗(a::I, b::I) = ... # some iterable object that generates all possible fusion outputs
# or
Base.iterate(::SectorProductIterator{I}[, state]) = ...
Base.IteratorSize(::Type{SectorProductIterator{I}}) = ... # HasLength() or IsInfinite()
Base.length(::SectorProductIterator{I}) = ... # if previous function returns HasLength()

TensorKit.Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) = ...

TensorKit.BraidingStyle(::Type{I}) = ... # NoBraiding(), Bosonic(), Fermionic(), Anyonic()
TensorKit.Rsymbol(a::I, b::I, c::I) = ... # only if BraidingStyle(I) != NoBraiding()

Base.iterate(::TensorKit.SectorValues{I}[, state]) = ...
Base.IteratorSize(::Type{TensorKit.SectorValues{I}}) = ... # HasLenght() or IsInfinite()
# if previous function returns HasLength():
Base.length(::TensorKit.SectorValues{I}) = ...
# optional, but recommended if IteratorSize returns HasLength():
Base.getindex(::TensorKit.SectorValues{I}, i::Int) = ...
TensorKit.findindex(::TensorKit.SectorValues{I}, c::I) = ...

Base.hash(a::I, h::UInt)
```

Additionally, suitable definitions can be given for
```julia
TensorKit.sectorscalartype(::Type{I}) = ... # Int, Float64, ComplexF64, ...
TensorKit.dim(a::I) = ...
TensorKit.frobeniusschur_phase(a::I) = ...
TensorKit.Bsymbol(a::I, b::I, c::I) = ...
```

## [Fermionic sectors](@id ss_fermions)

All of the sectors discussed in [Group representations](@ref ss_groups) have a bosonic braiding style.
This does not mean that `Rsymbol` is always trivial, as for example for `SU2Irrep` the definition was given by
```julia
function Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)
    Nsymbol(sa, sb, sc) || return zero(sectorscalartype(SU2Irrep))
    return iseven(convert(Int, sa.j + sb.j - sc.j)) ? one(sectorscalartype(SU2Irrep)) :
        -one(sectorscalartype(SU2Irrep))
end
```
It does however mean that all twists ``θ_a`` are trivial (equal to ``1``).
We refer to the appendix on [Category theory](@ref s_categories) for more details on the meaning of the twist.
In summary, triviality of the twists implies that self-crossings of lines in tensor diagrams can be ignored,
i.e. they can be removed without changing the value of the diagram.


As is well known, this becomes more subtle when fermionic degrees are involved.
Technically, fermions are described using super vector spaces,
which are ``ℤ₂``-graded vector spaces ``V = V_0 ⊕ V_1``,
i.e. the vector space is decomposed as an (orthogonal) direct sum into an even and odd subspace,
corresponding to states with even and odd fermion parity, respectively.
The tensor product of two super vector spaces ``V`` and ``W`` is again graded as
``(V ⊗ W)_0 = (V_0 ⊗ W_0) ⊕ (V_1 ⊗ W_1)`` and ``(V ⊗ W)_1 = (V_0 ⊗ W_1) ⊕ (V_1 ⊗ W_0)``.
However, when exchanging two super vector spaces in such a tensor product,
the natural isomorphism ``V ⊗ W → W ⊗ V`` takes into account the fermionic nature by acting with a minus sign in the subspace ``V_1 ⊗ W_1``.
This is known as the Koszul sign rule.

The super vector space structure fits naturally in the framework of TensorKit.jl.
Indeed, the grading naturally corresponds to a ``ℤ₂``-valued sector structure,
which we implement as [`FermionParity`](@ref):
```julia
struct FermionParity <: Sector
    isodd::Bool
end
const fℤ₂ = FermionParity
fermionparity(f::FermionParity) = f.isodd
```
with straightforward fusion rules and associators
```julia
⊗(a::FermionParity, b::FermionParity) = (FermionParity(a.isodd ⊻ b.isodd),)
function Nsymbol(a::FermionParity, b::FermionParity, c::FermionParity)
    return (a.isodd ⊻ b.isodd) == c.isodd
end
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I <: FermionParity}
    return Int(Nsymbol(a, b, e) * Nsymbol(e, c, d) * Nsymbol(b, c, f) * Nsymbol(a, f, d))
end
```
but with non-trivial braiding and twist
```julia
function Rsymbol(a::I, b::I, c::I) where {I <: FermionParity}
    return a.isodd && b.isodd ? -Int(Nsymbol(a, b, c)) : Int(Nsymbol(a, b, c))
end
twist(a::FermionParity) = a.isodd ? -1 : +1
```

The super vector space structure can also be combined with other sector types using the `⊠` operator discussed [above](#ss_productsectors).
In some cases, there is a richer symmetry than ``ℤ₂`` associated with the fermionic degrees of freedom,
and there is a natural fermion parity associated with the sectors of that symmetry.
An example would be a ``\mathsf{U}_1`` symmetry associated with fermion number conservation,
where odd ``\mathsf{U}_1`` charges correspond to odd fermion parity.
However, it is then always possible to separate out the fermion parity structure as a separate sector,
and treat the original sectors as bosonic, by only restricting to combinations of sectors that
satisfy the natural fermion parity association.

For convenience (and partially due to legacy reasons),
TensorKitSectors.jl does provide [`FermionNumber`](@ref) and [`FermionSpin`](@ref) constructors,
which are defined as
```julia
const FermionNumber = U1Irrep ⊠ FermionParity
const fU₁ = FermionNumber
FermionNumber(a::Int) = U1Irrep(a) ⊠ FermionParity(isodd(a))

const FermionSpin = SU2Irrep ⊠ FermionParity
const fSU₂ = FermionSpin
FermionSpin(j::Real) = (s = SU2Irrep(j); s ⊠ FermionParity(isodd(twice(s.j))))
```

We conclude this subsection with some examples.
```@repl sectors
p = FermionParity(true)
p ⊗ p
twist(p)
FusionStyle(p)
BraidingStyle(p)

s = FermionSpin(3//2)
dim(s)
twist(s)
typeof(s)
FusionStyle(s)
BraidingStyle(s)
collect(s ⊗ s)
for s2 in s ⊗ s
    @show s2
    @show Rsymbol(s, s, s2)
end
```
Note in particular how the `Rsymbol` values have opposite signs to the bosonic case,
where the fusion of two equal half-integer spins to the trivial sector is antisymmetric and would thus have `Rsymbol` value `-1`.

## Anyons

Both `Bosonic` and `Fermionic` braiding styles are `SymmetricBraiding` styles,
which means that exchanging two sectors twice is equivalent to the identity operation.
In tensor network diagrams,
this implies that lines that cross twice are equivalent to them not crossing at all,
or also, that there is no distinction betweeen a line crossing "above" or "below" another line.
More technically, the relevant group describing the exchange processes is the permutation group,
whereas in more general cases it would be the braid group.

This more general case is denoted as the `Anyonic` braiding style in TensorKit.jl,
because examples of this behaviour appear in the context of anyons in topological phases of matter.

There are currently two well-known sector types with `Anyonic` braiding style implemented in TensorKitSectors.jl,
namely [`FibonacciAnyon`](@ref) and [`IsingAnyon`](@ref).
Their values represent the (equivalence classes of) simple objects of the well-known Fibonicci and Ising fusion categories.
As an example, we illustrate below the Fibonacci anyons, which has only two distinct sectors,
namely the unit sector `𝟙` and one non-trivial sector denoted as `τ`.
The fusion rules are given by `τ ⊗ τ = 𝟙 ⊕ τ`, and the topological data is summarized by the following code

```@repl sectors
𝟙 = FibonacciAnyon(:I)
τ = FibonacciAnyon(:τ)
collect(τ ⊗ τ)
FusionStyle(τ)
BraidingStyle(τ)
dim(𝟙)
dim(τ)
F𝟙 = Fsymbol(τ,τ,τ,𝟙,τ,τ)
Fτ = [Fsymbol(τ,τ,τ,τ,𝟙,𝟙) Fsymbol(τ,τ,τ,τ,𝟙,τ); Fsymbol(τ,τ,τ,τ,τ,𝟙) Fsymbol(τ,τ,τ,τ,τ,τ)]
Fτ'*Fτ
polar(x) = rationalize.((abs(x), angle(x)/(2pi)))
Rsymbol(τ,τ,𝟙) |> polar
Rsymbol(τ,τ,τ) |> polar
twist(τ) |> polar
```

## [Further generalizations](@id ss_generalsectors)

The `Anyonic` braiding style is one generalization beyond the bosonic and fermionic representation theory of groups,
i.e. the action of groups on vector spaces and super vector spaces.
It is also possible to consider fusion categories without braiding structure,
represented as `NoBraiding` in TensorKitSectors.jl.
Indeed, the framework for sectors outlined above is in one-to-one correspondence to the topological data for 
specifying a unitary (spherical and braided, and hence ribbon) [fusion category](https://en.wikipedia.org/wiki/Fusion_category), which is reviewed in the appendix on [category theory](@ref s_categories).
For such categories, the objects are not necessarily vector spaces and
the fusion and splitting tensors ``X^{ab}_{c,μ}`` do not necessarily exist as actual tensors.
However, the morphism spaces ``c → a ⊗ b`` still behave as vector spaces,
and the ``X^{ab}_{c,μ}`` act as generic basis for that space.
As TensorKit.jl does not rely on the ``X^{ab}_{c,μ}`` themselves (even when they do exist),
it can also deal with such general fusion categories.
An extensive list of (the topological data of) such fusion categories, with and without braiding,
is provided in [CategoryData.jl](https://github.com/lkdvos/CategoryData.jl).

Within TensorKit.jl, the only sector with `NoBraiding` is the [`PlanarTrivial`](@ref) sector,
which is actually equivalent to the `Trivial` sector,
but where the braiding has been "disabled" for testing purposes.

Finally, as mentioned above, a recent extension prepares TensorKitSectors.jl
to deal with multi-fusion categories,
where the sectors (simple objects) are organized in a matrix-like structure
and thus have an additional row and column index.
Fusion between sectors is only possible when the row and column indices match appropriately;
otherwise the fusion product is empty.
In this structure, the different groups of "diagonal" sectors define separate fusion categories,
whereas the off-diagonal sectors define bimodule categories between these fusion categories.
Every diagonal group has its own unit sector,
which also acts as the left / right unit for other sectors in the same column / row. 
The global unit object is not simple, but rather given by the direct sum of all diagonal unit sectors.
We do not document or illustrate this structure here,
but refer to the relevant functions [`leftunit`](@ref), [`rightunit`](@ref), [`allunits`](@ref)
and [`UnitStyle`](@ref) for more information.
Furthermore, we refer to [MultiTensorKit.jl](https://github.com/QuantumKitHub/MultiTensorKit.jl)
for examples and ongoing development work on using multi-fusion categories.