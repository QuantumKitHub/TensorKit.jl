# [Index manipulations](@id s_indexmanipulations)

```@setup indexmanip
using TensorKit
using LinearAlgebra
```

Tensor maps have a bipartition of their indices into a codomain and a domain.
Index manipulations are operations that reorganize this structure: reordering indices, moving them between domain and codomain, flipping arrows, applying twists, or inserting and removing trivial factors.

Throughout this page, index positions are specified using `Index2Tuple{N₁,N₂}`, i.e. a pair `(p₁, p₂)` of tuples.
The indices in `p₁` form the new codomain, and those in `p₂` form the new domain.
The helper functions [`codomainind`](@ref), [`domainind`](@ref), [`allind`](@ref), [`numout`](@ref) and [`numin`](@ref) are available to retrieve the current index structure of a tensor.

## Permuting and braiding

For sector types with a symmetric braiding (`BraidingStyle(I) isa SymmetricBraiding`), use [`permute`](@ref):

```@docs; canonical=false
permute(::AbstractTensorMap, ::Index2Tuple)
permute!(::AbstractTensorMap, ::AbstractTensorMap, ::Index2Tuple)
```

For general braiding, use [`braid`](@ref), which requires an additional `levels` argument that assigns a height to each index.
When two indices need to exchange places, the index with the higher level crosses over the index with the lower level.

```@docs; canonical=false
braid(::AbstractTensorMap, ::Index2Tuple, ::IndexTuple)
braid!
```

For plain tensors (`sectortype(t) == Trivial`), `permute` acts exactly like `permutedims` on the underlying array data:

```@repl indexmanip
V = ℂ^2;
t = randn(V ⊗ V ← V ⊗ V);
ta = convert(Array, t);
t′ = permute(t, ((4, 2, 3), (1,)));
convert(Array, t′) ≈ permutedims(ta, (4, 2, 3, 1))
```

## Transposing and repartitioning

[`transpose`](@ref) is a special case of braiding restricted to *cyclic permutations*, i.e. permutations where indices do not cross.
Unlike a generic `braid`, it introduces a compensating (inverse) twist, which is necessary to satisfy the categorical definition of transpose.

```@raw html
<img src="../img/tensor-transpose.svg" alt="transpose" class="color-invertible"/>
```

```@docs; canonical=false
transpose(::AbstractTensorMap, ::Index2Tuple)
transpose!
```

[`repartition`](@ref) is a further special case that only changes the codomain/domain split while preserving cyclic order:

```@docs; canonical=false
repartition(::AbstractTensorMap, ::Int, ::Int)
repartition!
```

## Flipping arrows

[`flip`](@ref) applies an isomorphism to change the arrow direction on selected indices:

```@docs; canonical=false
flip(t::AbstractTensorMap, I)
```

!!! note
    `flip` is not involutory: `flip(flip(t, I), I) ≠ t` in general.
    Use `flip(flip(t, I), I; inv=true)` to recover the original tensor.

## Twisting

[`twist`](@ref) applies the monoidal twist to one or more indices.
For `BraidingStyle(I) == Bosonic()`, all twists are trivial and `twist` returns the tensor unchanged.

```@docs; canonical=false
twist(::AbstractTensorMap, ::Int)
twist!
```

## Inserting and removing unit spaces

The following functions insert or remove a trivial tensor product factor (a space isomorphic to the scalar field) at a given position.
Passing `Val(i)` instead of an integer `i` improves type stability.

```@docs; canonical=false
insertleftunit(::AbstractTensorMap, ::Val{i}) where {i}
insertrightunit(::AbstractTensorMap, ::Val{i}) where {i}
removeunit(::AbstractTensorMap, ::Val{i}) where {i}
```

## Fusing and splitting indices

There is no dedicated function for fusing or splitting indices.
For a plain tensor (`sectortype(t) == Trivial`), this is equivalent to `reshape` on the underlying array.
In the general case, one can construct an explicit isomorphism using [`isomorphism`](@ref) (or [`unitary`](@ref) for Euclidean spaces) and contract it with the tensor:

```julia
u = unitary(fuse(space(t, i) ⊗ space(t, j)), space(t, i) ⊗ space(t, j))
# then contract u with indices i and j of t via @tensor
```

Note that tensor factorizations (SVD, QR, etc.) can be applied directly to any index bipartition without needing to fuse indices first; see [Tensor factorizations](@ref ss_tensor_factorization).
