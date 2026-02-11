# [Graded spaces](@id s_gradedspaces)

```@setup gradedspaces
using TensorKit
```

We have introduced `Sector` subtypes as a way to label the irreps or sectors in the
decomposition ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``. To actually represent such spaces, we now also
introduce a corresponding type `GradedSpace`, which is a subtype of `ElementarySpace`, i.e.
```julia
struct GradedSpace{I<:Sector, D} <: ElementarySpace
    dims::D
    dual::Bool
end
```
Here, `D` is a type parameter to denote the data structure used to store the degeneracy or
multiplicity dimensions ``n_a`` of the different sectors. For convenience, `Vect[I]` will
return the fully concrete type with `D` specified.

Note that, conventionally, a graded vector space is a space that has a natural direct sum
decomposition over some set of labels, i.e. ``V = ⨁_{a ∈ I} V_a`` where the label set ``I``
has the structure of a semigroup ``a ⊗ b = c ∈ I``. Here, we generalize this notation by
using for ``I`` the fusion ring of a fusion category,
``a ⊗ b = ⨁_{c ∈ I} ⨁_{μ = 1}^{N_{a,b}^c} c``. However, this is mostly to lower the barrier,
as really the instances of `GradedSpace` represent just general objects in a fusion category
(or strictly speaking, a pre-fusion category, as we allow for an infinite number of simple
objects, e.g. the irreps of a continuous group).

### Implementation details

As mentioned, the way in which the degeneracy dimensions ``n_a`` are stored depends on the
specific sector type `I`, more specifically on the `IteratorSize` of `values(I)`. If
`IteratorSize(values(I)) isa Union{IsInfinite, SizeUnknown}`, the different sectors ``a``
and their corresponding degeneracy ``n_a`` are stored as key value pairs in an `Associative`
array, i.e. a dictionary `dims::SectorDict`. As the total number of sectors in `values(I)`
can be infinite, only sectors ``a`` for which ``n_a`` are stored. Here, `SectorDict` is a
constant type alias for a specific dictionary implementation, which currently resorts to
`SortedVectorDict` implemented in TensorKit.jl. Hence, the sectors and their corresponding
dimensions are stored as two matching lists (`Vector` instances), which are ordered based on
the property `isless(a::I, b::I)`. This ensures that the space ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``
has some unique canonical order in the direct sum decomposition, such that two different but
equal instances created independently always match.

If `IteratorSize(values(I)) isa Union{HasLength, HasShape}`, the degeneracy dimensions `n_a`
are stored for all sectors `a ∈ values(I)` (also if `n_a == 0`) in a tuple, more
specifically a `NTuple{N, Int}` with `N = length(values(I))`. The methods
`getindex(values(I), i)` and `findindex(values(I), a)` are used to map between a sector
`a ∈ values(I)` and a corresponding index `i ∈ 1:N`. As `N` is a compile time constant,
these types can be created in a type stable manner. Note however that this implies that for
large values of `N`, it can be beneficial to define
`IteratorSize(values(a)) = SizeUnknown()` to not overly burden the compiler.

### Constructing instances

As mentioned, the convenience method `Vect[I]` will return the concrete type
`GradedSpace{I,D}` with the matching value of `D`, so that should never be a user's concern.
In fact, for consistency, `Vect[Trivial]` will just return `ComplexSpace`, which is not even
a specific type of `GradedSpace`. For the specific case of group irreps as sectors, one can
use `Rep[G]` with `G` the group, as inspired by the categorical name
``\mathbf{Rep}_{\mathsf{G}}``. Some illustrations:
```@repl gradedspaces
Vect[Trivial]
Vect[U1Irrep]
Vect[Irrep[U₁]]
Rep[U₁]
Rep[ℤ₂ × SU₂]
Vect[Irrep[ℤ₂ × SU₂]]
```
Note that we also have the specific alias `U₁Space`. In fact, for all the common groups we
have a number of aliases, both in ASCII and using Unicode:
```julia
# ASCII type aliases
const ZNSpace{N} = GradedSpace{ZNIrrep{N}, NTuple{N,Int}}
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

To create specific instances of those types, one can e.g. just use
`V = GradedSpace(a=>n_a, b=>n_b, c=>n_c)` or `V = GradedSpace(iterator)` where `iterator` is
any iterator (e.g. a dictionary or a generator) that yields `Pair{I,Int}` instances. With
those constructions, `I` is inferred from the type of sectors. However, it is often more
convenient to specify the sector type explicitly (using one of the many alias provided),
since then the sectors are automatically converted to the correct type. Thereto, one can use
`Vect[I]`, or when `I` corresponds to the irreducible representations of a group, `Rep[G]`.
Some examples:
```@repl gradedspaces
Vect[Irrep[U₁]](0=>3, 1=>2, -1=>1) ==
    GradedSpace(U1Irrep(0)=>3, U1Irrep(1)=>2, U1Irrep(-1)=>1) == 
        U1Space(0=>3, 1=>2, -1=>1)
```
The fact that `Rep[G]` also works with product groups makes it easy to specify e.g.
```@repl gradedspaces
Rep[ℤ₂ × SU₂]((0,0) => 3, (1,1/2) => 2, (0,1) => 1) == 
    GradedSpace((Z2Irrep(0) ⊠ SU2Irrep(0)) => 3, (Z2Irrep(1) ⊠ SU2Irrep(1/2)) => 2, (Z2Irrep(0) ⊠ SU2Irrep(1)) => 1)
```

### Methods

There are a number of methods to work with instances `V` of `GradedSpace`. The function
[`sectortype`](@ref) returns the type of the sector labels. It also works on other vector
spaces, in which case it returns [`Trivial`](@ref). The function [`sectors`](@ref) returns
an iterator over the different sectors `a` with non-zero `n_a`, for other `ElementarySpace`
types it returns `(Trivial,)`. The degeneracy dimensions `n_a` can be extracted as
`dim(V, a)`, it properly returns `0` if sector `a` is not present in the decomposition of
`V`. With [`hassector(V, a)`](@ref) one can check if `V` contains a sector `a` with
`dim(V, a) > 0`. Finally, `dim(V)` returns the total dimension of the space `V`, i.e.
``∑_a n_a d_a`` or thus `dim(V) = sum(dim(V, a) * dim(a) for a in sectors(V))`. Note that a
representation space `V` has certain sectors `a` with dimensions `n_a`, then its dual `V'`
will report to have sectors `dual(a)`, and `dim(V', dual(a)) == n_a`. There is a subtelty
regarding the difference between the dual of a representation space ``R_a^*``, on which the
conjugate representation acts, and the representation space of the irrep
`dual(a) == conj(a)` that is isomorphic to the conjugate representation, i.e.
``R_{\overline{a}} ≂ R_a^*`` but they are not equal. We return to this in the section on
[fusion trees](@ref ss_fusiontrees). This is true also in more general fusion categories
beyond the representation categories of groups.

Other methods for `ElementarySpace`, such as [`dual`](@ref), [`fuse`](@ref) and
[`flip`](@ref) also work. In fact, `GradedSpace` is the reason `flip` exists, because in
this case it is different than `dual`. The existence of flip originates from the non-trivial
isomorphism between ``R_{\overline{a}}`` and ``R_{a}^*``, i.e. the representation space of
the dual ``\overline{a}`` of sector ``a`` and the dual of the representation space of sector
``a``. In order for `flip(V)` to be isomorphic to `V`, it is such that, if
`V = GradedSpace(a=>n_a,...)` then `flip(V) = dual(GradedSpace(dual(a)=>n_a,....))`.

Furthermore, for two spaces `V1 = GradedSpace(a=>n1_a, ...)` and
`V2 = GradedSpace(a=>n2_a, ...)`, we have
`infimum(V1, V2) = GradedSpace(a=>min(n1_a, n2_a), ....)` and similarly for `supremum`, i.e.
they act on the degeneracy dimensions of every sector separately. Therefore, it can be that
the return value of `infimum(V1, V2)` or `supremum(V1, V2)` is neither equal to `V1` or
`V2`.

For `W` a `ProductSpace{Vect[I], N}`, [`sectors(W)`](@ref) returns an iterator that
generates all possible combinations of sectors `as` represented as `NTuple{I,N}`. The
function [`dims(W, as)`](@ref) returns the corresponding tuple with degeneracy dimensions,
while [`dim(W, as)`](@ref) returns the product of these dimensions.
[`hassector(W, as)`](@ref) is equivalent to `dim(W, as) > 0`. Finally, there is the function
[`blocksectors(W)`](@ref) which returns a list (of type `Vector`) with all possible "block
sectors" or total/coupled sectors that can result from fusing the individual uncoupled
sectors in `W`. Correspondingly, [`blockdim(W, a)`](@ref) counts the total degeneracy
dimension of the coupled sector `a` in `W`. The machinery for computing this is the topic of
the next section on [Fusion trees](@ref ss_fusiontrees), but first, it's time for some
examples.

### Examples

Let's start with an example involving ``\mathsf{U}_1``:
```@repl gradedspaces
V1 = Rep[U₁](0=>3, 1=>2, -1=>1)
V1 == U1Space(0=>3, 1=>2, -1=>1) == U₁Space(-1=>1, 1=>2,0=>3) # order doesn't matter
(sectors(V1)...,)
dim(V1, U1Irrep(1))
dim(V1', Irrep[U₁](1)) == dim(V1, conj(U1Irrep(1))) == dim(V1, U1Irrep(-1))
hassector(V1, Irrep[U₁](1))
hassector(V1, Irrep[U₁](2))
dual(V1)
flip(V1)
dual(V1) ≅ V1
flip(V1) ≅ V1
V2 = U1Space(0=>2, 1=>1, -1=>1, 2=>1, -2=>1)
infimum(V1, V2)
supremum(V1, V2)
⊕(V1,V2)
W = ⊗(V1,V2)
collect(sectors(W))
dims(W, (Irrep[U₁](0), Irrep[U₁](0)))
dim(W, (Irrep[U₁](0), Irrep[U₁](0)))
hassector(W, (Irrep[U₁](0), Irrep[U₁](0)))
hassector(W, (Irrep[U₁](2), Irrep[U₁](0)))
fuse(W)
(blocksectors(W)...,)
blockdim(W, Irrep[U₁](0))
```
and then with ``\mathsf{SU}_2``:
```@repl gradedspaces
V1 = Vect[Irrep[SU₂]](0=>3, 1//2=>2, 1=>1)
V1 == SU2Space(0=>3, 1/2=>2, 1=>1) == SU₂Space(0=>3, 0.5=>2, 1=>1)
(sectors(V1)...,)
dim(V1, SU2Irrep(1))
dim(V1', SU2Irrep(1)) == dim(V1, conj(SU2Irrep(1))) == dim(V1, Irrep[SU₂](1))
dim(V1)
hassector(V1, Irrep[SU₂](1))
hassector(V1, Irrep[SU₂](2))
dual(V1)
flip(V1)
V2 = SU2Space(0=>2, 1//2=>1, 1=>1, 3//2=>1, 2=>1)
infimum(V1, V2)
supremum(V1, V2)
⊕(V1,V2)
W = ⊗(V1,V2)
collect(sectors(W))
dims(W, (Irrep[SU₂](0), Irrep[SU₂](0)))
dim(W, (Irrep[SU₂](0), Irrep[SU₂](0)))
hassector(W, (SU2Irrep(0), SU2Irrep(0)))
hassector(W, (SU2Irrep(2), SU2Irrep(0)))
fuse(W)
(blocksectors(W)...,)
blockdim(W, SU2Irrep(0))
```
