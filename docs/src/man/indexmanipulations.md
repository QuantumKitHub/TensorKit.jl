# [Index manipulations](@id s_indexmanipulations)

```@meta
CollapsedDocStrings = true
```

```@setup indexmanip
using TensorKit
using LinearAlgebra
```

A `TensorMap{T, S, N₁, N₂}` is a linear map from a domain (a `ProductSpace{S, N₂}`) to a codomain (a `ProductSpace{S, N₁}`).
In practice, the bipartition of the `N₁ + N₂` indices between domain and codomain rarely remains fixed: algorithms typically need to reshuffle indices between the two sides, reorder them, or change the arrow direction on individual indices before passing a tensor to a factorization or contraction.

Index manipulations cover all such operations.
They act on the structure of the tensor data in a way that is fully determined by the categorical data of the `sectortype`, such that TensorKit automatically manipulates the tensor entries accordingly.
The operations fall into three groups, which mirror the structure of the source file:

*   **Reweighting**: [`flip`](@ref) and [`twist`](@ref) apply local isomorphisms to individual indices without changing the index structure.
*   **Space insertion/removal**: [`insertleftunit`](@ref), [`insertrightunit`](@ref) and [`removeunit`](@ref) add or remove trivial (scalar) index factors.
*   **Index rearrangements**: [`permute`](@ref), [`braid`](@ref), [`transpose`](@ref) and [`repartition`](@ref) reorder indices and/or move them between domain and codomain.

Throughout this page, new index positions are specified using `Index2Tuple{N₁, N₂}`, i.e. a pair `(p₁, p₂)` of index tuples.
The indices listed in `p₁` form the new codomain and those in `p₂` form the new domain.
The following helpers retrieve the current index structure of a tensor:

```@docs; canonical=false
numout
numin
numind
codomainind
domainind
allind
```

## Reweighting

Reweighting operations modify the entries of a tensor by applying local isomorphisms to individual indices, without changing the number of indices or their partition between domain and codomain.
In particular, [`twist`](@ref) applies the topological spin (monoidal twist) to selected indices; this operation preserves the space of the indices and is completely trivial for `BraidingStyle(I) == Bosonic()`.
In contrast, [`flip`](@ref) changes the arrow direction on selected indices by applying a (non-canonical!) isomorphism between the index space and its dual.

```@docs; canonical=false
twist(::AbstractTensorMap, ::Int)
twist!
flip(t::AbstractTensorMap, I)
```

## Inserting and removing unit spaces

The next set of functions add or remove a trivial tensor product factor at a specified index position, without affecting any other indices.
We distinguish between [`insertleftunit`](@ref), which inserts a unit space before index `i` (the unit space becoming index `i`),
and [`insertrightunit`](@ref), which inserts after index `i` (the unit space becoming index `i + 1`);
[`removeunit`](@ref) undoes either insertion.

For tensors `t` with `UnitStyle(sectortype(t)) = SimpleUnit()`, the only relevant difference between `insertleftunit(t, i + 1)` and `insertrightunit(t, i)` is that `insertleftunit(t, numout(t) + 1)` inserts the unit space as first index in the domain, whereas `insertrightunit(t, numout(t))` will insert the unit space as last index in the codomain. 

Passing `Val(i)` instead of an integer `i` for the position may improve type stability.

```@docs; canonical=false
insertleftunit(::AbstractTensorMap, ::Val{i}) where {i}
insertrightunit(::AbstractTensorMap, ::Val{i}) where {i}
removeunit(::AbstractTensorMap, ::Val{i}) where {i}
```

## Index rearrangements

These operations reorder indices and/or move them between domain and codomain by applying the transposing or braiding isomorphisms of the underlying category.
They form a hierarchy from most general to most restricted:

- [`braid`](@ref) is the most general: it accepts any permutation and requires a `levels` argument — a tuple of heights, one per index — that determines whether each index crosses over or under the others it has to pass.
- [`permute`](@ref) is a simpler interface for sector types with a symmetric braiding (`BraidingStyle(I) isa SymmetricBraiding`), where over- and under-crossings are equivalent and `levels` is therefore not needed.
- [`transpose`](@ref) is restricted to *cyclic* permutations (indices do not cross).
- [`repartition`](@ref) only moves the codomain/domain boundary without reordering the indices at all.

For plain tensors (`sectortype(t) == Trivial`), `permute` and `braid` act like `permutedims` on the underlying array:

```@repl indexmanip
V = ℂ^2;
t = randn(V ⊗ V ← V ⊗ V);
ta = convert(Array, t);
t′ = permute(t, ((4, 2, 3), (1,)));
convert(Array, t′) ≈ permutedims(ta, (4, 2, 3, 1))
```

```@docs; canonical=false
braid(::AbstractTensorMap, ::Index2Tuple, ::IndexTuple)
braid!
permute(::AbstractTensorMap, ::Index2Tuple)
permute!(::AbstractTensorMap, ::AbstractTensorMap, ::Index2Tuple)
transpose(::AbstractTensorMap, ::Index2Tuple)
transpose!
repartition(::AbstractTensorMap, ::Int, ::Int)
repartition!
```

## Fusing and splitting indices

There is no dedicated functionality for fusing or splitting indices.
In the general case there is no canonical embedding of `V1 ⊗ V2` into the fused space `V = fuse(V1 ⊗ V2)`: any two such embeddings differ by a basis transform, i.e. there is a gauge freedom.
TensorKit resolves this by requiring the user to construct an explicit isomorphism — the *fuser* — and contract it with the tensor.
One particular isomorphism can be constructed using the [`unitary`](@ref) function.
It preserves norms and inner products, and has an inverse given by its adjoint. 
For a plain tensor (`sectortype(t) == Trivial`), applying this particular `unitary` is equivalent to `reshape` on the underlying array.

Fusing index `i` and `j = i+1` of a tensor `t` is then accomplished as

```@repl indexmanip
t = randn(ℂ^2 ⊗ ℂ^3 ⊗ ℂ^4);
F = unitary(fuse(space(t, 2) ⊗ space(t, 3)), space(t, 2) ⊗ space(t, 3));
@tensor t_fused[a, c] := F[c, b₁, b₂] * t[a, b₁, b₂]
```

The fusion is undone by contracting with the adjoint fuser, which is its inverse:

```@repl indexmanip
@tensor t_split[a, b₁, b₂] := t_fused[a, c] * conj(F[c, b₁, b₂]);
t_split ≈ t
```

The resulting `unitary` is a dense `TensorMap`, and this fusion and splitting approach is not optimized for maximal performance.
However, because most tensor operations including tensor factorizations (SVD, QR, etc.) can be applied without needing any fusion, we do not expect fusion and splitting to be an essential part of performance critical parts of typical tensor algorithms.

!!! warning
    For `BraidingStyle(I) == Fermionic()`, special care has to be taken with these operations.
    The definition of `unitary` is such that `F * F' ≈ I`, which may differ from the equivalent `@tensor` call depending on the duality of the spaces.
    See also the section on [Fermionic tensor contractions](@ref).
