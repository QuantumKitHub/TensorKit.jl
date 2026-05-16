# [Basic linear algebra](@id ss_tensor_linalg)

```@setup tensors
using TensorKit
using LinearAlgebra
```

`AbstractTensorMap` instances `t` represent linear maps, i.e. homomorphisms in a `𝕜`-linear category, just like matrices.
To a large extent, they follow the interface of `Matrix` in Julia's `LinearAlgebra` standard library.
Many methods from `LinearAlgebra` are (re)exported by TensorKit.jl, and can then us be used without `using LinearAlgebra` explicitly.
In all of the following methods, the implementation acts directly on the underlying matrix blocks (typically using the same method) and never needs to perform any basis transforms.

In particular, `AbstractTensorMap` instances can be composed, provided the domain of the first object coincides with the codomain of the second.
Composing tensor maps uses the regular multiplication symbol as in `t = t1 * t2`, which is also used for matrix multiplication.
TensorKit.jl also supports (and exports) the mutating method `mul!(t, t1, t2)`.
We can then also try to invert a tensor map using `inv(t)`, though this can only exist if the domain and codomain are isomorphic, which can e.g. be checked as `fuse(codomain(t)) == fuse(domain(t))`.
If the inverse is composed with another tensor `t2`, we can use the syntax `t1 \ t2` or `t2 / t1`.
However, this syntax also accepts instances `t1` whose domain and codomain are not isomorphic, and then amounts to `pinv(t1)`, the Moore-Penrose pseudoinverse.
This, however, is only really justified as minimizing the least squares problem if `InnerProductStyle(t) <: EuclideanProduct`.

`AbstractTensorMap` instances behave themselves as vectors (i.e. they are `𝕜`-linear) and so they can be multiplied by scalars and, if they live in the same space, i.e. have the same domain and codomain, they can be added to each other.
There is also a `zero(t)`, the additive identity, which produces a zero tensor with the same domain and codomain as `t`.
In addition, `TensorMap` supports basic Julia methods such as `fill!` and `copy!`, as well as `copy(t)` to create a copy with independent data.
Aside from basic `+` and `*` operations, TensorKit.jl reexports a number of efficient in-place methods from `LinearAlgebra`, such as `axpy!` (for `y ← α * x + y`), `axpby!` (for `y ← α * x + β * y`), `lmul!` and `rmul!` (for `y ← α * y` and `y ← y * α`, which is typically the same) and `mul!`, which can also be used for out-of-place scalar multiplication `y ← α * x`.

For `S = spacetype(t)` where `InnerProductStyle(S) <: EuclideanProduct`, we can compute `norm(t)`, and for two such instances, the inner product `dot(t1, t2)`, provided `t1` and `t2` have the same domain and codomain.
Furthermore, there is `normalize(t)` and `normalize!(t)` to return a scaled version of `t` with unit norm.
These operations should also exist for `InnerProductStyle(S) <: HasInnerProduct`, but require an interface for defining a custom inner product in these spaces.
Currently, there is no concrete subtype of `HasInnerProduct` that is not an `EuclideanProduct`.
In particular, `CartesianSpace`, `ComplexSpace` and `GradedSpace` all have `InnerProductStyle(S) <: EuclideanProduct`.

With tensors that have `InnerProductStyle(t) <: EuclideanProduct` there is associated an adjoint operation, given by `adjoint(t)` or simply `t'`, such that `domain(t') == codomain(t)` and `codomain(t') == domain(t)`.
Note that for an instance `t::TensorMap{S, N₁, N₂}`, `t'` is simply stored in a wrapper called `AdjointTensorMap{S, N₂, N₁}`, which is another subtype of `AbstractTensorMap`.
This should be mostly invisible to the user, as all methods should work for this type as well.
It can be hard to reason about the index order of `t'`, i.e. index `i` of `t` appears in `t'` at index position `j = TensorKit.adjointtensorindex(t, i)`, where the latter method is typically not necessary and hence unexported.
There is also a plural `TensorKit.adjointtensorindices` to convert multiple indices at once.
Note that, because the adjoint interchanges domain and codomain, we have `space(t', j) == space(t, i)'`.

`AbstractTensorMap` instances can furthermore be tested for exact (`t1 == t2`) or approximate (`t1 ≈ t2`) equality, though the latter requires that `norm` can be computed.

When tensor map instances are endomorphisms, i.e. they have the same domain and codomain, there is a multiplicative identity which can be obtained as `one(t)` or `one!(t)`, where the latter overwrites the contents of `t`.
The multiplicative identity on a space `V` can also be obtained using `id(A, V)` as discussed [above](@ref ss_tensor_construction), such that for a general homomorphism `t′`, we have `t′ == id(codomain(t′)) * t′ == t′ * id(domain(t′))`.
Returning to the case of endomorphisms `t`, we can compute the trace via `tr(t)` and exponentiate them using `exp(t)`, or if the contents of `t` can be destroyed in the process, `exp!(t)`.
Furthermore, there are a number of tensor factorizations for both endomorphisms and general homomorphisms that we discuss on the [Tensor factorizations](@ref ss_tensor_factorization) page.

Finally, there are a number of operations that also belong in this paragraph because of their analogy to common matrix operations.
The tensor product of two `TensorMap` instances `t1` and `t2` is obtained as `t1 ⊗ t2` and results in a new `TensorMap` with `codomain(t1 ⊗ t2) = codomain(t1) ⊗ codomain(t2)` and `domain(t1 ⊗ t2) = domain(t1) ⊗ domain(t2)`.
If we have two `TensorMap{T, S, N, 1}` instances `t1` and `t2` with the same codomain, we can combine them in a way that is analogous to `hcat`, i.e. we stack them such that the new tensor `catdomain(t1, t2)` has also the same codomain, but has a domain which is `domain(t1) ⊕ domain(t2)`.
Similarly, if `t1` and `t2` are of type `TensorMap{T, S, 1, N}` and have the same domain, the operation `catcodomain(t1, t2)` results in a new tensor with the same domain and a codomain given by `codomain(t1) ⊕ codomain(t2)`, which is the analogy of `vcat`.
Note that direct sum only makes sense between `ElementarySpace` objects, i.e. there is no way to give a tensor product meaning to a direct sum of tensor product spaces.

Time for some more examples:
```@repl tensors
using TensorKit # hide
V1 = ℂ^2
t = randn(V1 ← V1 ⊗ V1 ⊗ V1)
t == t + zero(t) == t * id(domain(t)) == id(codomain(t)) * t
t2 = randn(ComplexF64, codomain(t), domain(t));
dot(t2, t)
tr(t2' * t)
dot(t2, t) ≈ dot(t', t2')
dot(t2, t2)
norm(t2)^2
t3 = copy!(similar(t, ComplexF64), t);
t3 == t
rmul!(t3, 0.8);
t3 ≈ 0.8 * t
axpby!(0.5, t2, 1.3im, t3);
t3 ≈ 0.5 * t2 + 0.8 * 1.3im * t
t4 = randn(fuse(codomain(t)), codomain(t));
t5 = TensorMap{Float64}(undef, fuse(codomain(t)), domain(t));
mul!(t5, t4, t) == t4 * t
inv(t4) * t4 ≈ id(codomain(t))
t4 * inv(t4) ≈ id(fuse(codomain(t)))
t4 \ (t4 * t) ≈ t
t6 = randn(ComplexF64, V1, codomain(t));
numout(t4) == numout(t6) == 1
t7 = catcodomain(t4, t6);
foreach(println, (codomain(t4), codomain(t6), codomain(t7)))
norm(t7) ≈ sqrt(norm(t4)^2 + norm(t6)^2)
t8 = t4 ⊗ t6;
foreach(println, (codomain(t4), codomain(t6), codomain(t8)))
foreach(println, (domain(t4), domain(t6), domain(t8)))
norm(t8) ≈ norm(t4)*norm(t6)
```
