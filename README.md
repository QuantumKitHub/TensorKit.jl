<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QuantumKitHub/TensorKit.jl/blob/main/docs/src/assets/logo-dark.svg">
    <img alt="TensorKit.jl logo" src="https://github.com/QuantumKitHub/TensorKit.jl/blob/main/docs/src/assets/logo.svg" width="150">
</picture>

# TensorKit.jl

A Julia package for large-scale tensor computations, with a hint of category theory.

| **Documentation** | **Digital Object Identifier** | **Downloads** |
|:-----------------:|:-----------------------------:|:-------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] |
<!-- | [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] | -->

| **Build Status** | **Coverage** | **Quality assurance** |
|:----------------:|:------------:|:---------------------:|
| [![CI][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] |


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://QuantumKitHub.github.io/TensorKit.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://QuantumKitHub.github.io/TensorKit.jl/latest

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.8421339.svg
[doi-url]: https://doi.org/10.5281/zenodo.8421339

[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FTensorKit&query=total_requests&label=Downloads
[downloads-url]: http://juliapkgstats.com/pkg/TensorKit

[ci-img]: https://github.com/QuantumKitHub/TensorKit.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/QuantumKitHub/TensorKit.jl/actions/workflows/CI.yml

[codecov-img]: https://codecov.io/gh/QuantumKitHub/TensorKit.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuantumKitHub/TensorKit.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

## Release notes for v0.13

TensorKit v0.13 brings a number of performance improvements, but also comes with a number of
breaking changes:

1. The scalar type (the `eltype` of the tensor data) is now an explicit parameter of the
   the `TensorMap` type, and appears in the first position. As a consequence,
   `TensorMap{T}(undef, codomain ← domain)` can and should now be used to create a
   `TensorMap` with uninitialised data with scalar type `T`.

3. The constructors for creating tensors with randomly initialised data, of the form
   `TensorMap(randn, T, codomain ← domain)`, are being replaced with
   `randn(T, codomain ← domain)`. Hence, we overload the methods `rand` and `randn` from
   Base (actually, Random, and also `Random.randexp`) and mimick the `Array` constructors,
   relying on the fact that we use spaces instead of integers to characterise the tensor
   structure. As with integer-based `rand` and `randn`, a custom random number generator
   from the `Random` module can be passed as the first argument, and the scalar type `T` is
   optional, defaulting to `Float64`. The old constructors
   `TensorMap(randn, T, codomain ← domain)` still exist in deprecation mode, and will be
   removed in the 1.0 release.

3. The `TensorMap` data structure has been changed (simplified), so that all tensor data now
   resides in a single array of type `<:DenseVector`. While this does not lead to breaking
   changes in the interface, it does mean that `TensorMap` objects from TensorKit.jl
   v0.12.7 or earlier that were saved to disk using e.g. JLD2.jl, cannot simply be read back
   in using the new version of TensorKit.jl. We provide a script [below](https://github.com/QuantumKitHub/TensorKit.jl?tab=readme-ov-file#transferring-tensormap-data-from-older-versions-to-v013)
   to export data in a format that can be read back in by TensorKit.jl v0.13.

Major non-breaking changes include:

* Support for [TensorOperations.jl v5](https://github.com/QuantumKitHub/TensorOperations.jl), and
  with this the new backend and allocator support within the `@tensor` macro.
* The part of TensorKit.jl that defines the `Sector` type hierarchy and its corresponding
  interface, as well as the implementation of all of the standard symmetries, has been
  moved to a separate package called [TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl),
  so that it can also be used by other packages and is a more lightweight dependency.
  TensorKitSectors.jl is a direct dependency and is automatically installed when installing
  TensorKit.jl. Furthermore, its public interface is re-exported by TensorKit.jl, so that
  this should not have any observable consequences.
* The `fusiontrees` iterator now iterates over `FusionTree` objects in a different order,
  which will facilitate speeding up certain operations in the future. Furthermore, it now
  also accepts a `ProductSpace` object as first input, instead of simply a tuple of `Sector`
  objects. This also affects the data ordering in the `TensorMap` objects.
* The structural information associated with a `TensorMap` object (or rather with the
  `HomSpace` instance that represents the space to which the tensor belongs) is no longer
  stored within the tensor, but is cached in a global (or task local) dictionary. As a
  result, this information does not need to be recomputed when new `TensorMap` objects are
  created, thus eliminating some overhead that can be significant in certain applications.

### Transferring `TensorMap` data from older versions to v0.13:

To export `TensorMap` data from TensorKit.jl v0.12.7 or earlier, you should first export the
data there in a format that is explicit about how tensor data is associated with the
structural part of the tensor, i.e. the splitting and fusion tree pairs. Therefore, on the 
older version of TensorKit.jl, use the following code to save the data

```julia
using JLD2
filename = "choose_some_filename.jld2"
t_dict = Dict(:space => space(t), :data => Dict((f₁, f₂) => t[f₁, f₂] for (f₁, f₂) in fusiontrees(t)))
jldsave(filename; t_dict)
```

If you have already upgraded to TensorKit.jl v0.13, you can still install the old version in
a separate environment, for example a temporary environment. To do this, run

```julia
]activate --temp
]add TensorKit@0.12.7
```

or

```julia
import Pkg
Pkg.activate(; temp = true)
Pkg.add("TensorKit@0.12.7")
```

Then, in the environment where you have TensorKit.jl v0.13 installed, you can read in the
data and reconstruct the tensor as follows:

```julia
using JLD2
filename = "choose_some_filename.jld2"
t_dict = jldload(filename)
T = eltype(valtype(t_dict[:data]))
t = TensorMap{T}(undef, t_dict[:space])
for ((f₁, f₂), val) in t_dict[:data]
    t[f₁, f₂] .= val
end
```

## Overview

TensorKit.jl is a package that provides types and methods to represent and manipulate
tensors with symmetries. The emphasis is on the structure and functionality needed to build
tensor network algorithms for the simulation of quantum many-body systems. Such tensors are
typically invariant under a symmetry group which acts via specific representations on each of
the indices of the tensor. TensorKit.jl provides the functionality for constructing such
tensors and performing typical operations such as tensor contractions and decompositions,
thereby preserving the symmetries and exploiting them for optimal performance.

While most common symmetries are already shipped with TensorKit.jl, there exist several
extensions: [SUNRepresentations.jl](https://github.com/QuantumKitHub/SUNRepresentations.jl)
provides support for SU(N), while [CategoryData.jl](https://github.com/lkdvos/CategoryData.jl)
incorporates a large collection of *small* fusion categories.
Additionally, for libraries that implement tensor network algorithms on top of
TensorKit.jl, check out [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl),
[MERAKit.jl](https://github.com/mhauru/MERAKit.jl) and [PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl).

Check out the [tutorial](https://QuantumKitHub.github.io/TensorKit.jl/stable/man/tutorial/) and the
full [documentation](https://QuantumKitHub.github.io/TensorKit.jl/stable).

## Installation

`TensorKit.jl` can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
pkg> add TensorKit
```

Or, equivalently, via the `Pkg` API:
```julia
julia> import Pkg; Pkg.add("TensorKit.jl")
```

## Documentation

-   [**STABLE**][docs-stable-url] - **documentation of the most recently tagged version.**
-   [**DEV**][docs-dev-url] - *documentation of the in-development version.*

## Project Status

The package is tested against Julia versions `1.10` and the latest `1.x` release, as
well as against the nightly builds of the Julia `master` branch on Linux, macOS, and Windows
platforms with a 64-bit architecture.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

[issues-url]: https://github.com/QuantumKitHub/TensorKit.jl/issues

## Acknowledgements

The design and development of the TensorKit.jl package have benefited from countless discussions with many people, including most current and former members of the Quantum Group at Ghent University.
Being an open-source software project developed over the course of many years, we also thank all past, current and future contributors, including the bug reports and feature requests that have shaped this package.
In particular, we like to thank [Maarten Van Damme](@maartenvd), who initiated the MPSKit.jl package on top of TensorKit.jl early-on, and has as such had a strong influence on the development and design decisions of the TensorKit.jl package.
