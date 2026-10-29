# [Tensor factorizations](@id ss_tensor_factorization)

```@setup tensors
using TensorKit
using LinearAlgebra
```

As tensors are linear maps, they support various kinds of factorizations.
These functions all interpret the provided `AbstractTensorMap` instances as a map from `domain` to `codomain`, which can be thought of as reshaping the tensor into a matrix according to the current bipartition of the indices.

TensorKit's factorizations are provided by [MatrixAlgebraKit.jl](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl), which is used to supply both the interface, as well as the implementation of the various operations on the blocks of data.
For specific details on the provided functionality, we refer to its [documentation page](https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/user_interface/decompositions/).

Finally, note that each of the factorizations takes the current partition of `domain` and `codomain` as the *axis* along which to matricize and perform the factorization.
In order to obtain factorizations according to a different bipartition of the indices, we can use any of the previously mentioned [index manipulations](@ref s_indexmanipulations) before the factorization.

Some examples to conclude this section:
```@repl tensors
V1 = SU₂Space(0 => 2, 1/2 => 1)
V2 = SU₂Space(0 => 1, 1/2 => 1, 1 => 1)

t = randn(V1 ⊗ V1, V2);
U, S, Vh = svd_compact(t);
t ≈ U * S * Vh
D, V = eigh_full(t' * t);
D ≈ S * S
U' * U ≈ id(domain(U))
S

Q, R = left_orth(t; alg = :svd);
Q' * Q ≈ id(domain(Q))
t ≈ Q * R

U2, S2, Vh2, ε = svd_trunc(t; trunc = truncspace(V1));
Vh2 * Vh2' ≈ id(codomain(Vh2))
S2
ε ≈ norm(block(S, Irrep[SU₂](1))) * sqrt(dim(Irrep[SU₂](1)))

L, Q = right_orth(permute(t, ((1,), (2, 3))));
codomain(L), domain(L), domain(Q)
Q * Q'
P = Q' * Q;
P ≈ P * P
t′ = permute(t, ((1,), (2, 3)));
t′ ≈ t′ * P
```

To compute only the `R` or `L` factor with QR/LQ-based `left_orth!` and `right_orth!`, pass a zero-dimensional tensor map as the basis output.
Its blocks are empty matrices, so MatrixAlgebraKit skips constructing the orthogonal factor.
The corestriction output must still have the usual intermediate space:

```julia
V = infimum(fuse(codomain(t)), fuse(domain(t)))
V0 = zerospace(spacetype(t))
Q0 = similar(t, V0 ← V0)

R = similar(t, V ← domain(t))
_, R = left_orth!(copy(t), (Q0, R))

L = similar(t, codomain(t) ← V)
L, _ = right_orth!(copy(t), (L, Q0))
```

For a `DiagonalTensorMap`, use an empty `DiagonalTensorMap` as the basis output instead.
Differentiation of these R/L-only paths is not yet supported.
