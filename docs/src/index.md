# TensorKit.jl

*A Julia package for large-scale tensor computations, with a hint of category theory.*

```@meta
CurrentModule = TensorKit
```

## Package summary

TensorKit.jl aims to be a generic package for working with tensors as they appear throughout
the physical sciences. TensorKit implements a parametric type [`Tensor`](@ref) (which is
actually a specific case of the type [`TensorMap`](@ref)) and defines for these types a
number of vector space operations (scalar multiplication, addition, norms and inner
products), index operations (permutations) and linear algebra operations (multiplication,
factorizations). Finally, tensor contractions can be performed using the `@tensor` macro
from [TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl).

Currently, most effort is oriented towards tensors as they appear in the context of quantum
many body physics and in particular the field of tensor networks. Such tensors often have
large dimensions and take on a specific structure when symmetries are present. To deal with
generic symmetries, we employ notations and concepts from category theory all the way down
to the definition of a tensor.

At the same time, TensorKit.jl focusses on computational efficiency and performance. The
underlying storage of a tensor's data can be any `DenseArray`. Currently, certain operations
are already multithreaded, either by distributing the different blocks in case of a
structured tensor (i.e. with symmetries) or by using multithreading provided by the package
[Strided.jl](https://github.com/Jutho/Strided.jl). In the future, we also plan to
investigate using `CuArray`s as underlying storage for the tensors data, so as to leverage
GPUs for the different operations defined on tensors.

## Contents of the manual

```@contents
Pages = ["man/intro.md", "man/categories.md", "man/spaces.md", "man/sectors.md", "man/tensors.md"]
Depth = 3
```

## Library outline

```@contents
Pages = ["lib/sectors.md","lib/spaces.md","lib/tensors.md"]
Depth = 2
```
