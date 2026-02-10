# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Guidelines for updating this changelog

When making changes to this project, please update the "Unreleased" section with your changes under the appropriate category:

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Performance** for performance improvements.

When releasing a new version, move the "Unreleased" changes to a new version section with the release date.

## [Unreleased](https://github.com/QuantumKitHub/TensorKit.jl/compare/v0.16.1...HEAD)

### Added

- A more robust promotion system for `storagetype`s to better handle working with unions and other abstract tensor map types ([#370](https://github.com/QuantumKitHub/TensorKit.jl/pull/370)).

### Changed


### Deprecated


### Removed


### Fixed

- Fix `findtruncated` with `truncspace` ([#369](https://github.com/QuantumKitHub/TensorKit.jl/pull/369))
- Fix `truncrank` when kept rank is larger than input ([#368](https://github.com/QuantumKitHub/TensorKit.jl/pull/368))
- Added missing `similar` definition for `SectorVector` ([#367](https://github.com/QuantumKitHub/TensorKit.jl/pull/367))
- Small fixes for CUDA support ([#366](https://github.com/QuantumKitHub/TensorKit.jl/pull/366))


## [0.16.1](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.16.1) - 2026-02-05

### Added

- Extended support for selecting storage types in the `TensorMap` constructors ([#327](https://github.com/QuantumKitHub/TensorKit.jl/pull/327))
- `similar_diagonal` to handle storage types when constructing diagonals ([#330](https://github.com/QuantumKitHub/TensorKit.jl/pull/330))
- Support for CUDA.jl ([#336](https://github.com/QuantumKitHub/TensorKit.jl/pull/336),[#325](https://github.com/QuantumKitHub/TensorKit.jl/pull/325))
- Support for Adapt.jl ([#344](https://github.com/QuantumKitHub/TensorKit.jl/pull/344))
- Preliminary support for Mooncake ([#352](https://github.com/QuantumKitHub/TensorKit.jl/pull/352))
- Export `TimeReversed` symbol ([#337](https://github.com/QuantumKitHub/TensorKit.jl/pull/337))

### Fixed

- Issue with using relative tolerances in truncation schemes ([#314](https://github.com/QuantumKitHub/TensorKit.jl/issues/314))
- Using `scalartype` instead of `eltype` in BLAS contraction ([#326](https://github.com/QuantumKitHub/TensorKit.jl/pull/326))
- Divide by zero error in `show` for empty tensors ([#329](https://github.com/QuantumKitHub/TensorKit.jl/pull/329))
- `svd_vals(::DiagonalTensorMap)` correctly outputs `SectorVector` and implementation fix. ([#333](https://github.com/QuantumKitHub/TensorKit.jl/pull/329))
- Fix handling of real tensors with complex scalartype ([#360](https://github.com/QuantumKitHub/TensorKit.jl/pull/360))
- Sorted diagonal eigenvalues to ensure consistent ordering ([#350](https://github.com/QuantumKitHub/TensorKit.jl/pull/350))
- Adding tensors of different types now correctly promotes ([#364](https://github.com/QuantumKitHub/TensorKit.jl/pull/364))

### Changed

- `convert(TensorMap, t)` now retains `storagetype` when converting ([#357](https://github.com/QuantumKitHub/TensorKit.jl/pull/357))
- `transpose` specialization for `DiagonalTensorMap` for improved correctness/performance ([#335](https://github.com/QuantumKitHub/TensorKit.jl/pull/335))
- Uniformized `CartesianSpace` and `ComplexSpace` constructors ([#334](https://github.com/QuantumKitHub/TensorKit.jl/pull/334))

### Performance

- GPU-friendly truncation implementations ([#349](https://github.com/QuantumKitHub/TensorKit.jl/pull/349))
- `norm` performance optimizations ([#351](https://github.com/QuantumKitHub/TensorKit.jl/pull/351))
- TensorOperations ChainRules performance improvements ([#343](https://github.com/QuantumKitHub/TensorKit.jl/pull/343))
- Type-stability and small test fixes (various commits)

## [0.16.0](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.16.0) - 2025-12-08

### Added

- `rrule` for `transpose` operation ([#319](https://github.com/QuantumKitHub/TensorKit.jl/pull/319))
- New functions for multifusion support: `unitspace`, `zerospace`, `leftunitspace`, `rightunitspace`, `isunitspace` ([#291](https://github.com/QuantumKitHub/TensorKit.jl/pull/291))
- Support for projections and orthogonal complements ([#312](https://github.com/QuantumKitHub/TensorKit.jl/pull/312))

### Changed

- Improvements to the default printing of tensors, where only a (possibly compressed) representation of the (possibly truncated) list of diagonal blocks is printed. Use `blocks(t)` and `subblocks(t)` for a full inspection of the tensor data ([#304](https://github.com/QuantumKitHub/TensorKit.jl/pull/304), [#322](https://github.com/QuantumKitHub/TensorKit.jl/pull/322)))
- Updated `left_orth`, `right_orth`, `left_null` and `right_null` interfaces for MatrixAlgebraKit v0.6 ([#312](https://github.com/QuantumKitHub/TensorKit.jl/pull/312))
- Updated `ishermitian` and `isisometric` implementations ([#312](https://github.com/QuantumKitHub/TensorKit.jl/pull/312))
- Sector functions now by default use `unit` instead of `one`, `isunit` instead of `isone`, and `dual` instead of `conj` ([#291](https://github.com/QuantumKitHub/TensorKit.jl/pull/291))
- Reworked TensorOperations implementation to use backend and allocator system ([#311](https://github.com/QuantumKitHub/TensorKit.jl/pull/311))
- Major documentation update/overhaul ([#289](https://github.com/QuantumKitHub/TensorKit.jl/pull/289))
- Added symmetric tensor tutorial as appendix ([#316](https://github.com/QuantumKitHub/TensorKit.jl/pull/316))
- Improved error messages throughout codebase ([#309](https://github.com/QuantumKitHub/TensorKit.jl/pull/309))
- `eigvals` and `svdvals` now output `SectorVector` objects, which do behave as `AbstractVector` but also have the option of iterating the blocks through `Base.pairs`. ([#324](https://github.com/QuantumKitHub/TensorKit.jl/pull/309)

### Deprecated

### Removed

- All deprecations from v0.15: old factorization function names (`leftorth`, `rightorth`, `tsvd`, `eig`, `eigh`)
- Old truncation strategy names (`truncdim`, `truncbelow`)
- Old factorization struct types (`OrthogonalFactorization`)
- Old constructor syntaxes and deprecated `rand*` function names

### Fixed

- Avoid unnecessary copy in `twist` for tensors with bosonic braiding ([#305](https://github.com/QuantumKitHub/TensorKit.jl/pull/305))
- Small fixes and typos ([#295](https://github.com/QuantumKitHub/TensorKit.jl/pull/295))
- `eig_vals`, `svd_vals`, etc now all output `SectorVector` objects instead of `DiagonalTensorMap`s, in line with how MatrixAlgebraKit returns `Vector`s instead of `Diagonal`s ([#324](https://github.com/QuantumKitHub/TensorKit.jl/pull/309)

## [0.15.3](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.15.3) - 2025-10-30

### Fixed

- Fixed typo in `show(::GradedSpace)` ([#308](https://github.com/QuantumKitHub/TensorKit.jl/pull/308))
- Updated printing of `ProductSpace{<:Any,0}`
- Added tests for show methods

## [0.15.2](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.15.2) - 2025-10-28

### Added

- `subblocks` iterator for easier inspection of tensor data ([#304](https://github.com/QuantumKitHub/TensorKit.jl/pull/304))

### Changed

- Tensors no longer print their data by default, only their spaces. Use `blocks(t)` or `subblocks(t)` to inspect data ([#304](https://github.com/QuantumKitHub/TensorKit.jl/pull/304))
- Updated compatibility to TensorKitSectors v0.3 ([#290](https://github.com/QuantumKitHub/TensorKit.jl/pull/290))
- Refactored test suite and split into groups ([#298](https://github.com/QuantumKitHub/TensorKit.jl/pull/298))

### Fixed

- Fixed `TruncationIntersection` implementation and test ([#300](https://github.com/QuantumKitHub/TensorKit.jl/pull/300))
- Avoided unnecessary allocations in rrules for contractions and tensor products ([#306](https://github.com/QuantumKitHub/TensorKit.jl/pull/306))

## [0.15.1](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.15.1) - 2025-10-09

### Fixed

- Small fixes and typo corrections ([#295](https://github.com/QuantumKitHub/TensorKit.jl/pull/295))

## [0.15.0](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.15.0) - 2025-10-03

### Added

- [MatrixAlgebraKit](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl) as new backend for tensor factorizations ([#230](https://github.com/QuantumKitHub/TensorKit.jl/pull/230))
- `foreachblock(f, t::AbstractTensorMap...)` - uniform interface to iterate through tensor blocks
- `eig_trunc` and `eigh_trunc` - truncated eigenvalue decompositions
- `ominus` (and unicode `⊖`) - compute orthogonal complement of a space
- Backend selection for factorizations - swap algorithms or implementations

### Changed

- `left_orth` and `right_orth` now always output tensors with a single connecting space
- `left_orth` and `right_orth` now always have connecting space with `isdual=false`
- Code formatter is now [Runic.jl](https://github.com/fredrikekre/Runic.jl)

### Deprecated

- Factorization functions `leftorth`, `rightorth`, `tsvd`, `eig`, `eigh` in favor of MatrixAlgebraKit variants (`left_orth`, `right_orth`, `svd_compact`, `eig_full`, `eigh_full`)
- Truncation strategies: `truncdim` (use `truncrank`) and `truncbelow` (use `trunctol`)
- `OrthogonalFactorization` structs (constructors deprecated to return equivalent MatrixAlgebraKit algorithm structs)

### Removed

- Direct permute-and-factorize operations (incompatible with `permute` vs `braid` distinction)
- `Polar` decomposition behavior for `left_orth`/`right_orth` (use `left_polar`/`right_polar` instead for `isposdef` R factors)

## [0.14.0](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.14.0) - 2024-12-19

### Added

- `DiagonalTensorMap` type for representing tensor maps with diagonal blocks
- `reduceddim(V)` function that sums up degeneracy dimensions for each sector
- New index manipulation functions:
  - `flip(t, i)`
  - `insertleftunit(t, i)`
  - `insertrightunit(t, i)`
  - `removeunit(t, i)`

### Changed

- Singular values and eigenvalues now explicitly represented as `DiagonalTensorMap` instances
- SVD truncation now guarantees smaller singular values are removed first, irrespective of sector quantum dimension

## [0.13.0](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.13.0) - 2024-11-24

### Added

- Refactored `TensorMap` constructors to align with Julia `Array` constructors
- Convenience constructors: `ones`, `zeros`, `rand`, `randn` for tensors
- TensorOperations v5 support

### Changed

- Scalar type as parameter to `AbstractTensorMap` type: `AbstractTensorMap{E, S, N₁, N₂}`
- Default way to create uninitialized tensors is now `TensorMap{E}(undef, codomain ← domain)`
- Behavior of `copy` for `BraidingTensor` to properly instantiate a `TensorMap`
- TensorKitSectors promoted to separate package
- `TensorMap` data structure now consists of single vector with blocks as views
- `FusionTree` vertices now only use `Int` labels for `GenericFusion` sectors
