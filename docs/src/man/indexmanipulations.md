# [Index manipulations](@id s_indexmanipulations)

```@meta
CollapsedDocStrings = true
```

```@setup indexmanip
using TensorKit
using LinearAlgebra
```

A `TensorMap{T, S, N₁, N₂}` is a linear map from a domain (a `ProductSpace{S, N₂}`) to a codomain (a `ProductSpace{S, N₁}`).
In practice, the bipartition of the `N₁ + N₂` indices between domain and codomain is often not fixed: algorithms typically need to reshuffle indices between the two sides, reorder them, or change the arrow direction on individual indices before passing a tensor to a factorization or contraction.

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

[`flip`](@ref) changes the arrow direction on selected indices by applying the corresponding isomorphism between a space and its dual.
[`twist`](@ref) applies the topological spin (monoidal twist) to selected indices; for `BraidingStyle(I) == Bosonic()` this is always trivial.

```@docs; canonical=false
flip(t::AbstractTensorMap, I)
twist(::AbstractTensorMap, ::Int)
twist!
```

## Inserting and removing unit spaces

These functions add or remove a trivial tensor product factor at a specified index position, without affecting any other indices.
[`insertleftunit`](@ref) inserts before position `i` and [`insertrightunit`](@ref) inserts after position `i`; [`removeunit`](@ref) undoes either insertion.
Passing `Val(i)` instead of an `Int` for the position may improve type stability.

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

There is no dedicated function for fusing or splitting indices.
For a plain tensor (`sectortype(t) == Trivial`), this is equivalent to `reshape` on the underlying array.

In the general case there is no canonical embedding of `V1 ⊗ V2` into the fused space `V = fuse(V1 ⊗ V2)`: any two such embeddings differ by a basis transform, i.e. there is a gauge freedom.
TensorKit resolves this by requiring the user to construct an explicit isomorphism — the *fuser* — and contract it with the tensor:

```julia
f = unitary(fuse(space(t, i) ⊗ space(t, j)), space(t, i) ⊗ space(t, j))
@tensor t_fused[…, a, …] := f[a, i, j] * t[…, i, j, …]
```

Splitting is then the adjoint of the same map:

```julia
@tensor t_split[…, i, j, …] := f'[i, j, a] * t_fused[…, a, …]
```

Using `f'` as the splitter guarantees that the round-trip is the identity, i.e. `t_split == t`.
Using a *different* isomorphism to split would give a physically equivalent but numerically different tensor, so it is important to keep `f` and its adjoint consistent throughout a calculation.

Note that tensor factorizations (SVD, QR, etc.) can be applied directly to any index bipartition without needing to fuse indices first; see [Tensor factorizations](@ref ss_tensor_factorization).
