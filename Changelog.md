# Planned changes for v1.0

Features that are planned to be implemented before the release of v1.0.0, in no particular order.

- [x] Separate `Sectors` module
- [x] Make `TrivialTensorMap` and `TensorMap` be the same
- [x] Simplify `TensorMap` type to hide `rowr` and `colr`
- [x] Change block order in `rowr` / `colr` to speed up particular contractions
- [x] Make `AdjointTensorMap` generic
- [ ] Rewrite planar operations in order to be AD-compatible
- [x] Fix rrules for fermionic tensors
- [ ] Fix GPU support
- [ ] Proper threading support
- [ ] Rewrite documentation

# Changelog

## v0.15

### MatrixAlgebraKit backend for factorizations

This version introduces [MatrixAlgebraKit](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl) as the new backend for tensor factorizations, replacing the previous internal `TensorKit.MatrixAlgebra` implementation. See [#230](https://github.com/QuantumKitHub/TensorKit.jl/pull/230) for details.

#### Added
- `foreachblock(f, t::AbstractTensorMap...)` - uniform interface to iterate through tensor blocks (prepares for future multithreading support)
- `eig_trunc` and `eigh_trunc` - truncated eigenvalue decompositions
- `ominus` (and unicode `⊖`) - compute orthogonal complement of a space
- Backend selection for factorizations - swap algorithms or implementations

#### Changed
- Factorization functions `leftorth`, `rightorth`, `tsvd`, `eig`, `eigh` deprecated in favor of MatrixAlgebraKit variants (`left_orth`, `right_orth`, `tsvd`, `eig`, `eigh`)
- Truncation strategies now use MatrixAlgebraKit names: `truncrank` (replaces `truncdim`), `truncbelow` (replaces `trunctol`)
- `left_orth` and `right_orth` now always output tensors with a single connecting space
- `left_orth` and `right_orth` now always have connecting space with `isdual=false`
- code formatter is now [Runic.jl](https://github.com/fredrikekre/Runic.jl)

#### Removed
- `OrthogonalFactorization` structs (constructors deprecated to return equivalent MatrixAlgebraKit algorithm structs)
- Direct permute-and-factorize operations (incompatible with `permute` vs `braid` distinction)
- `Polar` decomposition behavior for `left_orth`/`right_orth` (use `left_polar`/`right_polar` instead for `isposdef` R factors)

## v0.14

### Use `DiagonalTensorMap` for singular values and eigenvalues

The diagonal (1,1) tensor that contains the singular values or eigenvalues of a tensor
are now explicitly represented as `DiagonalTensorMap` instances.

### New index functionality
There are is new functionality for manipulating the spaces associated with a tensor:
* `flip(t, i)` changes the duality flag of the `i`th index of `t`, in such a way that flipping
  a pair of contracted indices in an `@tensor` contraction does not affect the result.
* `insertleftunit(t, i)` and `insertrightunit(t, i)` insert a trivial unit space to the left
  or to right of index `i`, whereas `removeunit(t, i)` removes such a trivial unit space.

### SVD truncation change (breaking)
There is a subtle but breaking change in the truncation mechanism in SVD, where now it is
guaranteed that smaller singular values are removed first, irrespective of the (quantum)
dimension of the sector to which they belong

### `DiagonalTensorMap` and `reduceddim`

This adds a `DiagonalTensorMap` type for representing tensor maps in which all of the
blocks are diagonal. This only makes sense for maps between a single index in the domain and
the codomain (which are furthermore required to be the same), as otherwise the fusion and
splitting trees from the domain and codomain to the blocked sectors would itself be
nondiagonal. This new type will be used to capture the singular values and eigenvalues as
tensor maps in the corresponding decompositions. The number of free parameters in a
`DiagonalTensorMap` instance on vector space `V` is equal to `reduceddim(V)`, a new function
that sums up the degeneracy dimension `dim(V, c)` for each of the sectors `c` in `V`. This
function can be useful by itself and is also exported from the `TensorKit` module. An
instance of `DiagonalTensorMap` can then be created as `DiagonalTensorMap(data, V)` where
`data` is a vector of length `reduceddim(V)`.

## v0.13

### `AbstractTensorMap{E,S,N₁,N₂}`

This adds the scalar type as a parameter to the `AbstractTensorMap` type. This is useful in
the contexts where different types of tensors are used (`AdjointTensor`, `BraidingTensor`,
...), which still have the same scalartype. Additionally, this removes many specializations
for methods in order to reduce ambiguity errors.

### `copy(BraidingTensor)`

This PR changes the behaviour of `copy` as an instantiator for creating a `TensorMap` from a
`BraidingTensor`. The rationale is that while this does sometimes happen in Julia `Base`,
this is always in the context of lazy wrapper types, for which it makes sense to not copy
the parent and then create a new wrapper. `BraidingTensor` does not wrap anything, so this
definition makes less sense.

### Refactor tensormap constructors

This PR refactors the constructors for `TensorMap` to be more in line with `Array`
constructors in Julia. In particular, this means that the default way to create
uninitialized tensors is now `TensorMap{E}(undef, codomain ← domain)`, reminiscent of
`Array{E}(undef, dims)`. Several convenience constructors are also added: `ones`, `zeros`,
`rand` and `randn` construct tensors when `dims` is replaced by `domain ← codomain`.

### TensorOperations v5

This PR bumps the compatibility of `TensorOperations` to v5. This is a breaking change
as there are some changes in the API.

### TensorKitSectors

This promotes TensorKitSectors to its own package, in order to make the dependencies
lighter and to separate the concerns of the two packages.

### FusionTree vertices

In order to simplify the `FusionTree` struct, we removed the usage of anything other than
`Int` as a vertex-label when working with `GenericFusion` sectors.

### TensorStructure

This PR changed the data structure of `TensorMap` to consist of a single vector of data,
where all blocks and subblocks are various views into this object. This entails major
simplifications for the outwards-facing interface, as now the `TensorMap` type requires less
parameters, and the fusion-tree structure is more clearly considered as an implementation
detail that can be left hidden to the user. Various other improvements and documentation
work was also carried out.
