# [Vector spaces](@id s_spaces)

```@setup tensorkit
using TensorKit
```

From the [Introduction](@ref s_intro), it should be clear that an important aspect in the
definition of a tensor (map) is specifying the vector spaces and their structure in the
domain and codomain of the map. The starting point is an abstract type
[`VectorSpace`](@ref)
```julia
abstract type VectorSpace end
```

Technically speaking, this name does not capture the full generality that TensorKit.jl
supports, as instances of subtypes of `VectorSpace` can encode general objects in linear
monoidal categories, which are not necessarily vector spaces. However, in order not to make
the remaining discussion too abstract or complicated, we will simply use the nomenclature of
vector spaces. In particular, we define two abstract subtypes
```julia
abstract type ElementarySpace <: VectorSpace end
const IndexSpace = ElementarySpace

abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end
```
Here, [`ElementarySpace`](@ref) is a super type for all vector spaces (objects) that can be
associated with the individual indices of a tensor, as hinted to by its alias `IndexSpace`.

On the other hand, subtypes of [`CompositeSpace{S}`](@ref) where `S<:ElementarySpace` are
composed of a number of elementary spaces of type `S`. So far, there is a single concrete
type [`ProductSpace{S,N}`](@ref) that represents the tensor product of `N` vector spaces of
a homogeneous type `S`. Its properties are discussed in the section on
[Composite spaces](@ref ss_compositespaces), together with possible extensions for the
future.

Throughout TensorKit.jl, the function [`spacetype`](@ref) returns the type of
`ElementarySpace` associated with e.g. a composite space or a tensor. It works both on
instances and in the type domain. Its use will be illustrated below.

## [Fields](@id ss_fields)

Vector spaces (and linear categories more generally) are defined over a field of scalars
``𝔽``. We define a type hierarchy to specify the scalar field, but so far only support real
and complex numbers, via
```julia
abstract type Field end

struct RealNumbers <: Field end
struct ComplexNumbers <: Field end

const ℝ = RealNumbers()
const ℂ = ComplexNumbers()
```
Note that `ℝ` and `ℂ` can be typed as `\bbR`+TAB and `\bbC`+TAB. One reason for defining
this new type hierarchy instead of recycling the types from Julia's `Number` hierarchy is to
introduce some syntactic sugar without committing type piracy. In particular, we now have
```@repl tensorkit
3 ∈ ℝ
5.0 ∈ ℂ
5.0 + 1.0 * im ∈ ℝ
Float64 ⊆ ℝ
ComplexF64 ⊆ ℂ
ℝ ⊆ ℂ
ℂ ⊆ ℝ
```
and furthermore —probably more usefully— `ℝ^n` and `ℂ^n` create specific elementary vector
spaces as described in the next section. The underlying field of a vector space or tensor
`a` can be obtained with [`field(a)`](@ref):

```@docs; canonical=false
field
```

## [Elementary spaces](@id ss_elementaryspaces)

As mentioned at the beginning of this section, vector spaces that are associated with the
individual indices of a tensor should be implemented as subtypes of `ElementarySpace`. As
the domain and codomain of a tensor map will be the tensor product of such objects which all
have the same type, it is important that associated vector spaces, such as the dual space,
are objects of the same concrete type (i.e. with the same type parameters in case of a
parametric type). In particular, every `ElementarySpace` should implement the following
methods

```@docs; canonical=false
dim(::ElementarySpace)
field(::ElementarySpace)
dual(::S) where {S<:ElementarySpace}
conj(::S) where {S<:ElementarySpace}
```

For convenience, the dual of a space `V` can also be obtained as `V'`. Furthermore, it is
sometimes necessary to test whether a space is a dual or conjugate space, for which the
methods [`isdual(::ElementarySpace)`](@ref) and [`isconj(::ElementarySpace)`](@ref) should
be implemented.

We furthermore define a trait type
```@docs; canonical=false
InnerProductStyle
```
to denote for a vector space `V` whether it has an inner product and thus a canonical
mapping from `dual(V)` to `V` (for real fields `𝔽 ⊆ ℝ`) or from `dual(V)` to `conj(V)` (for
complex fields). This mapping is provided by the metric, but no further support for working
with vector spaces with general metrics is currently implemented.

A number of concrete elementary spaces are implemented in TensorKit.jl. There is concrete
type `GeneralSpace` which is completely characterized by its field `𝔽`, its dimension and
whether its the dual and/or complex conjugate of $𝔽^d$.
```julia
struct GeneralSpace{𝔽} <: ElementarySpace
    d::Int
    dual::Bool
    conj::Bool
end
```

However, as the `InnerProductStyle` of `GeneralSpace` is currently set to
`NoInnerProduct()`, this type of vector space is currently quite limited, though it supports
constructing tensors and contracting them. However, most tensor factorizations will depend
on the presence of an Euclidean inner product.

Spaces with the `EuclideanInnerProduct()` style, i.e. with a standard Euclidean metric, have
the natural isomorphisms `dual(V) == V` (for `𝔽 == ℝ`) or `dual(V) == conj(V)` (for
`𝔽 == ℂ`). In the language of the appendix on [categories](@ref s_categories), this trait
represents [dagger or unitary categories](@ref ss_adjoints), and these vector spaces support
an `adjoint` operation.

In particular, two concrete types are provided:
```julia
struct CartesianSpace <: ElementarySpace
    d::Int
end
struct ComplexSpace <: ElementarySpace
  d::Int
  dual::Bool
end
```
They represent the Euclidean spaces $ℝ^d$ or $ℂ^d$ without further inner structure. They can
be created using the syntax `CartesianSpace(d) == ℝ^d` and `ComplexSpace(d) == ℂ^d`, or
`ComplexSpace(d, true) == ComplexSpace(d; dual = true) == (ℂ^d)'` for the dual space of the
latter. Note that the brackets are required because of the precedence rules, since `d' == d`
for `d::Integer`.

Some examples:
```@repl tensorkit
dim(ℝ^10)
(ℝ^10)' == ℝ^10
isdual((ℂ^5))
isdual((ℂ^5)')
isdual((ℝ^5)')
dual(ℂ^5) == (ℂ^5)' == conj(ℂ^5) == ComplexSpace(5; dual = true)
field(ℂ^5)
field(ℝ^3)
typeof(ℝ^3)
spacetype(ℝ^3)
InnerProductStyle(ℝ^3)
InnerProductStyle(ℂ^5)
```

!!! note For `ℂ^n` the dual space is equal (or naturally isomorphic) to the conjugate space,
but not to the space itself. This means that even for `ℂ^n`, arrows matter in the
diagrammatic notation for categories or for tensors, and in particular that a contraction
between two tensor indices will check that one is living in the space and the other in the
dual space. This is in contrast with several other software packages, especially in the
context of tensor networks, where arrows are only introduced when discussing symmetries. We
believe that our more puristic approach can be useful to detect errors (e.g. unintended
contractions). Only with `ℝ^n` will their be no distinction between a space and its dual.
When creating tensors with indices in `ℝ^n` that have complex data, a one-time warning will
be printed, but most operations should continue to work nonetheless.

One more important concrete implementation of `ElementarySpace` with a
`EuclideanInnerProduct()` is the [`GradedSpace`](@ref) type, which is used to represent a
graded complex vector space, where the grading is provided by the irreducible
representations of a group, or more generally, the simple objects of a unitary fusion
category. We refer to the subsection on [graded spaces](@ref ss_rep) on the
[next page](@ref s_sectorsrepfusion) for further information about `GradedSpace`.

## Operations with elementary spaces

Instances of `ElementarySpace` support a number of useful operations. Firstly, we define the
direct sum of two vector spaces `V1` and `V2` of the same `spacetype` (and with the same
value of `isdual`) as [`V1 ⊕ V2`](@ref), where `⊕` is obtained by typing `\oplus`+TAB.
[`zerospace(V)`](@ref) corresponds to the identity or zero element with respect to this
direct sum operation, i.e. it corresponds to a zero-dimensional space. Furthermore,
[`unitspace(V)`](@ref) applied to an elementary space returns a one-dimensional space, that
is isomorphic to the scalar field underlying the space itself. Finally, we have also
introduced the non-standard convention `V1 ⊖ V2` (obtained by typing `\ominus`+TAB.) in
order to obtain a space that is isomorphic to the quotient space of `V1` by `V2`, or thus, a
particular choice of complement of `V2` in `V1` such that `V1 == V2 ⊕ (V1 ⊖ V2)` is
satisfied.

Some examples illustrate this better.
```@repl tensorkit
ℝ^5 ⊕ ℝ^3
ℂ^5 ⊕ ℂ^3
ℂ^5 ⊕ (ℂ^3)'
zerospace(ℂ^3)
unitspace(ℝ^3)
ℂ^5 ⊕ unitspace(ComplexSpace)
unitspace((ℂ^3)')
(ℂ^5) ⊕ unitspace((ℂ^5))
(ℂ^5)' ⊕ unitspace((ℂ^5)')
(ℝ^5) ⊖ (ℝ^3)
(ℂ^5) ⊖ (ℂ^3)
(ℂ^5)' ⊖ (ℂ^3)'
ℂ^5 == ((ℂ^5) ⊖ (ℂ^3)) ⊕ (ℂ^3) == (ℂ^3) ⊕ ((ℂ^5) ⊖ (ℂ^3))
```

Note, finally, that we have defined `oplus` and `ominus` as ASCII alternatives for `⊕` and
`⊖` respectively.

A second type of operation with elementary spaces is the function
[`flip(V::ElementarySpace)`](@ref), which returns a space that is isomorphic to `V` but has
`isdual(flip(V)) == isdual(V')`, i.e., if `V` is a normal space, then `flip(V)` is a dual
space. `flip(V)` is different from `dual(V)` in the case of [`GradedSpace`](@ref). It is
useful to flip a tensor index from a ket to a bra (or vice versa), by contracting that index
with a unitary map from `V1` to `flip(V1)`.

While we provide some trivial examples here, we refer to the section on
[graded spaces](@ref ss_rep) for examples where `flip` acts non-trivially and produces
results that are different than `dual`.
```@repl tensorkit
flip(ℂ^4)
flip(ℂ^4) ≅ ℂ^4
flip(ℂ^4) == ℂ^4
```

Finally, we provide two methods [`infimum(V1, V2)`](@ref) and [`supremum(V1, V2)`](@ref) for
elementary spaces `V1` and `V2` with the same `spacetype` and value of `isdual`. The former
returns the "largest" elementary space `V::ElementarySpace` with the same value of `isdual`
such that we can construct surjective morphisms from both `V1` and `V2` to `V`. Similarly,
the latter returns the "smallest" elementary space `W::ElementarySpace` with the same value
of `isdual` such that we can construct injective morphisms from both `V1` and `V2` to `W`.
For `CartesianSpace` and `ComplexSpace`, this simply amounts to the space with minimal or
maximal dimension, but it is again more interesting in the case of [`GradedSpace`](@ref), as
discussed on the [next page](@ref ss_). It is that case where `infimum(V1, V2)` might be
different from either `V1` or `V2`, and similar for `supremum(V1, V2)`, which justifies the
choice of these names over simply `min` and `max`. Also note that these methods are a direct
consequence of the partial order that we can define between vector spaces of the same
`spacetype` more generally, as discussed below in the subsection
["More operations with vector spaces"](@ref ss_spaceops).

Some examples:
```@repl tensorkit
infimum(ℝ^5, ℝ^3)
supremum(ℂ^5, ℂ^3)
supremum(ℂ^5, (ℂ^3)')
supremum((ℂ^5)', (ℂ^3)')
```

## [Composite spaces](@id ss_compositespaces)

Composite spaces are vector spaces that are built up out of individual elementary vector
spaces of the same type. The most prominent (and currently only) example is a tensor product
of `N` elementary spaces of the same type `S`, which is implemented as
```julia
struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
end
```
Given some `V1::S`, `V2::S`, `V3::S` of the same type `S<:ElementarySpace`, we can easily
construct `ProductSpace{S,3}((V1, V2, V3))` as `ProductSpace(V1, V2, V3)` or using
`V1 ⊗ V2 ⊗ V3`, where `⊗` is simply obtained by typing `\otimes`+TAB. In fact, for
convenience, also the regular multiplication operator `*` acts as tensor product between
vector spaces, and as a consequence so does raising a vector space to a positive integer
power, i.e.
```@repl tensorkit
V1 = ℂ^2
V2 = ℂ^3
V1 ⊗ V2 ⊗ V1' == V1 * V2 * V1' == ProductSpace(V1, V2, V1') == ProductSpace(V1, V2) ⊗ V1'
V1^3
dim(V1 ⊗ V2)
dims(V1 ⊗ V2)
dual(V1 ⊗ V2 ⊗ V1')
spacetype(V1 ⊗ V2)
spacetype(ProductSpace{ComplexSpace,3})
```
Here, the newly introduced function `dims` maps `dim` to the individual spaces in a
`ProductSpace` and returns the result as a tuple. The rationale for the dual space of a
`ProductSpace` being the tensor product of the dual spaces in reverse order is explained in
the subsection on [duality](@ref ss_dual) in the appendix on [category theory](@ref
s_categories).

Following Julia's Base library, the function `one` applied to an instance of
`ProductSpace{S,N}` or of `S<:ElementarySpace` itself returns the multiplicative identity
for these objects. Similar to Julia Base, `one` also works in the type domain. The
multiplicative identity for vector spaces corresponds to the (monoidal) unit, which is
represented as `ProductSpace{S,0}(())` and simply printed as `one(S)` for the specific type
`S`. Note, however, that `one(S)` is strictly speaking only the multiplicative identity when
multiplied with `ProductSpace{S,N}` instances. For elementary spaces `V::S`, `V ⊗ one(V)`
will yield `ProductSpace{S,1}(V)` and not `V` itself. However, even though `V ⊗ one(V)` is
not strictly equal to `V`, the object `ProductSpace(V)`, which can also be created as
`⊗(V)`, does mathematically encapsulate the same vector space as `V`.

```@repl tensorkit
one(V1)
typeof(one(V1))
V1 * one(V1) == ProductSpace(V1) == ⊗(V1)
V1 * one(V1) == V1
P = V1 * V2;
one(P)
one(typeof(P))
P * one(P) == P == one(P) ⊗ P
```

In the future, other `CompositeSpace` types could be added. For example, the wave function
of an `N`-particle quantum system in first quantization would require the introduction of a
`SymmetricSpace{S,N}` or a `AntiSymmetricSpace{S,N}` for bosons or fermions respectively,
which correspond to the symmetric (permutation invariant) or antisymmetric subspace of
`V^N`, where `V::S` represents the Hilbert space of the single particle system. Other
scientific fields, like general relativity, might also benefit from tensors living in
subspace with certain symmetries under specific index permutations.

## [More operations with vector spaces](@id ss_spaceops)

Vector spaces of the same `spacetype` can be given a partial order, based on whether there
exist injective morphisms (a.k.a *monomorphisms*) or surjective morphisms (a.k.a.
*epimorphisms*) between them. In particular, we define `ismonomorphic(V1, V2)`, with Unicode
synonym `V1 ≾ V2` (obtained as `\precsim+TAB`), to express whether there exist monomorphisms
in `V1→V2`. Similarly, we define `isepimorphic(V1, V2)`, with Unicode synonym `V1 ≿ V2`
(obtained as `\succsim+TAB`), to express whether there exist epimorphisms in `V1→V2`.
Finally, we define `isisomorphic(V1, V2)`, with Unicode alternative `V1 ≅ V2` (obtained as
`\cong+TAB`), to express whether there exist isomorphism in `V1→V2`. In particular `V1 ≅ V2`
if and only if `V1 ≾ V2 && V1 ≿ V2`.

For completeness, we also export the strict comparison operators `≺` and `≻` (`\prec+TAB`
and `\succ+TAB`), with definitions
```julia
≺(V1::VectorSpace, V2::VectorSpace) = V1 ≾ V2 && !(V1 ≿ V2)
≻(V1::VectorSpace, V2::VectorSpace) = V1 ≿ V2 && !(V1 ≾ V2)
```
However, as we expect these to be less commonly used, no ASCII alternative is provided.

In the context of `InnerProductStyle(V) <: EuclideanInnerProduct`, `V1 ≾ V2` implies that
there exists isometries ``W:V1 → V2`` such that ``W^† ∘ W = \mathrm{id}_{V1}``, while
`V1 ≅ V2` implies that there exist unitaries ``U:V1→V2`` such that
``U^† ∘ U = \mathrm{id}_{V1}`` and ``U ∘ U^† = \mathrm{id}_{V2}``.

Note that spaces that are isomorphic are not necessarily equal. One can be a dual space, and
the other a normal space, or one can be an instance of `ProductSpace`, while the other is an
`ElementarySpace`. There will exist (infinitely) many isomorphisms between the corresponding
spaces, but in general none of those will be canonical.

There are also a number of convenience functions to create isomorphic spaces. The function
`fuse(V1, V2, ...)` or `fuse(V1 ⊗ V2 ⊗ ...)` returns an elementary space that is isomorphic
to `V1 ⊗ V2 ⊗ ...`.

## [Space of morphisms](@id ss_homspaces)
As mentioned in the introduction, we define tensor maps as linear maps from a `ProductSpace`
domain to a `ProductSpace` codomain. The set of all tensor maps with a fixed domain and
codomain constitutes a vector space, which we represent with the `HomSpace` type.
```julia
struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
    codomain::P1
    domain::P2
end
```
Aside from the standard constructor, a `HomSpace` instance can be created as either
`domain → codomain` or `codomain ← domain` (where the arrows are obtained as `\to+TAB` or
`\leftarrow+TAB`, and as `\rightarrow+TAB` respectively). The reason for first listing the
codomain and than the domain will become clear in the [section on tensor maps](@ref
s_tensors).

Note that `HomSpace` is not a subtype of `VectorSpace`, i.e. we restrict the latter to
encode all spaces and generalizations thereof (i.e. objects in linear monoidal categories)
that are associated with the indices and the domain and codomain of a tensor map. Even when
these generalizations are no longer strictly vector spaces and have unconventional
properties (such as non-integer dimensions), the space of tensor maps (homomorphisms)
between a given domain and codomain, represented by a `HomSpace` instance, is always a
vector space in the strict mathematical sense (with in particular an integer dimension).
Because `HomSpace` and the different subtypes of `VectorSpace` represent very different
mathematical concepts that do not directly interact, we have chosen to keep them separate in
the type hierarchy.

Furthermore, on these `HomSpace` instances, we define a number of useful methods that are a
precursor to the corresponding methods that we will define to manipulate the actual tensors,
as illustrated in the following example:
```@repl tensorkit
W = ℂ^2 ⊗ ℂ^3 → ℂ^3 ⊗ dual(ℂ^4)
field(W)
dual(W)
adjoint(W)
spacetype(W)
spacetype(typeof(W))
W[1]
W[2]
W[3]
W[4]
dim(W)
domain(W)
codomain(W)
numin(W)
numout(W)
numind(W)
numind(W) == numin(W) + numout(W)
permute(W, ((2, 3), (1, 4)))
flip(W, 3)
insertleftunit(W, 3)
insertrightunit(W, 2)
removeunit(insertrightunit(W, 2), 3)
TensorKit.compose(W, adjoint(W))
```
Note that indexing `W` follows an order that first targets the spaces in the codomain,
followed by the dual of the spaces in the domain. This particular convention is useful in
combination with the instances of type [`TensorMap`](@ref), which represent the actual
morphisms living in such a `HomSpace`. Also note that `dim(W)` is here given by the product
of the dimensions of the individual spaces, but that this is no longer true once symmetries
are involved. At any time will `dim(::HomSpace)` represent the number of linearly
independent morphisms in this space, or thus, the number of independent components that a
corresponding `TensorMap` object will have.

A complete list of methods defined on `HomSpace` instances together with the corresponding
documentation is provided in the [library section on Vector spaces](@ref s_libvectorspaces).
