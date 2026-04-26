# [Manipulating tensors](@id s_tensormanipulations)

## [Vector space and linear algebra operations](@id ss_tensor_linalg)

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
Furthermore, there are a number of tensor factorizations for both endomorphisms and general homomorphism that we discuss below.

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

## [Index manipulations](@id ss_indexmanipulation)

Index manipulations are operations that reorganize the bipartition of indices between the codomain and domain, possibly also reordering them or applying braiding isomorphisms.
They are covered in detail on a dedicated page: [Index manipulations](@ref s_indexmanipulations).

## [Tensor factorizations](@id ss_tensor_factorization)

As tensors are linear maps, they suport various kinds of factorizations.
These functions all interpret the provided `AbstractTensorMap` instances as a map from `domain` to `codomain`, which can be thought of as reshaping the tensor into a matrix according to the current bipartition of the indices.

TensorKit's factorizations are provided by [MatrixAlgebraKit.jl](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl), which is used to supply both the interface, as well as the implementation of the various operations on the blocks of data.
For specific details on the provided functionality, we refer to its [documentation page](https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/user_interface/decompositions/).

Finally, note that each of the factorizations takes the current partition of `domain` and `codomain` as the *axis* along which to matricize and perform the factorization.
In order to obtain factorizations according to a different bipartition of the indices, we can use any of the previously mentioned [index manipulations](@ref ss_indexmanipulation) before the factorization.

Some examples to conclude this section
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

## [Bosonic tensor contractions and tensor networks](@id ss_tensor_contraction)

One of the most important operation with tensor maps is to compose them, more generally known as contracting them.
As mentioned in the section on [category theory](@ref s_categories), a typical composition of maps in a ribbon category can graphically be represented as a planar arrangement of the morphisms (i.e. tensor maps, boxes with lines eminating from top and bottom, corresponding to source and target, i.e. domain and codomain), where the lines connecting the source and targets of the different morphisms should be thought of as ribbons, that can braid over or underneath each other, and that can twist.
Technically, we can embed this diagram in ``ℝ × [0,1]`` and attach all the unconnected line endings corresponding objects in the source at some position ``(x,0)`` for ``x∈ℝ``, and all line endings corresponding to objects in the target at some position ``(x,1)``.
The resulting morphism is then invariant under what is known as *framed three-dimensional isotopy*, i.e. three-dimensional rearrangements of the morphism that respect the rules of boxes connected by ribbons whose open endings are kept fixed.
Such a two-dimensional diagram cannot easily be encoded in a single line of code.

However, things simplify when the braiding is symmetric (such that over- and under- crossings become equivalent, i.e. just crossings), and when twists, i.e. self-crossings in this case, are trivial.
This amounts to `BraidingStyle(I) == Bosonic()` in the language of TensorKit.jl, and is true for any subcategory of ``\mathbf{Vect}``, i.e. ordinary tensors, possibly with some symmetry constraint.
The case of ``\mathbf{SVect}`` and its subcategories, and more general categories, are discussed below.

In the case of trivial twists, we can deform the diagram such that we first combine every morphism with a number of coevaluations ``η`` so as to represent it as a tensor, i.e. with a trivial domain.
We can then rearrange the morphism to be all ligned up horizontally, where the original morphism compositions are now being performed by evaluations ``ϵ``.
This process will generate a number of crossings and twists, where the latter can be omitted because they act trivially.
Similarly, double crossings can also be omitted.
As a consequence, the diagram, or the morphism it represents, is completely specified by the tensors it is composed of, and which indices between the different tensors are connect, via the evaluation ``ϵ``, and which indices make up the source and target of the resulting morphism.
If we also compose the resulting morphisms with coevaluations so that it has a trivial domain, we just have one type of unconnected lines, henceforth called open indices.
We sketch such a rearrangement in the following picture

```@raw html
<img src="../img/tensor-bosoniccontraction.svg" alt="tensor unitary" class="color-invertible"/>
```

Hence, we can now specify such a tensor diagram, henceforth called a tensor contraction or also tensor network, using a one-dimensional syntax that mimicks [abstract index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) and specifies which indices are connected by the evaluation map using Einstein's summation conventation.
Indeed, for `BraidingStyle(I) == Bosonic()`, such a tensor contraction can take the same format as if all tensors were just multi-dimensional arrays.
For this, we rely on the interface provided by the package [TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl).

The above picture would be encoded as
```julia
@tensor E[a, b, c, d, e] := A[v, w, d, x] * B[y, z, c, x] * C[v, e, y, b] * D[a, w, z]
```
or
```julia
@tensor E[:] := A[1, 2, -4, 3] * B[4, 5, -3, 3] * C[1, -5, 4, -2] * D[-1, 2, 5]
```
where the latter syntax is known as NCON-style, and labels the unconnected or outgoing indices with negative integers, and the contracted indices with positive integers.

A number of remarks are in order.
TensorOperations.jl accepts both integers and any valid variable name as dummy label for indices, and everything in between `[ ]` is not resolved in the current context but interpreted as a dummy label.
Here, we label the indices of a `TensorMap`, like `A::TensorMap{T, S, N₁, N₂}`, in a linear fashion, where the first position corresponds to the first space in `codomain(A)`, and so forth, up to position `N₁`.
Index `N₁ + 1` then corresponds to the first space in `domain(A)`.
However, because we have applied the coevaluation ``η``, it actually corresponds to the corresponding dual space, in accordance with the interface of [`space(A, i)`](@ref) that we introduced [above](@ref ss_tensor_properties), and as indiated by the dotted box around ``A`` in the above picture.
The same holds for the other tensor maps.
Note that our convention also requires that we braid indices that we brought from the domain to the codomain, and so this is only unambiguous for a symmetric braiding, where there is a unique way to permute the indices.

With the current syntax, we create a new object `E` because we use the definition operator `:=`.
Furthermore, with the current syntax, it will be a `Tensor`, i.e. it will have a trivial domain, and correspond to the dotted box in the picture above, rather than the actual morphism `E`.
We can also directly define `E` with the correct codomain and domain by rather using
```julia
@tensor E[a b c;d e] := A[v, w, d, x] * B[y, z, c, x] * C[v, e, y, b] * D[a, w, z]
```
or
```julia
@tensor E[(a, b, c);(d, e)] := A[v, w, d, x] * B[y, z, c, x] * C[v, e, y, b] * D[a, w, z]
```
where the latter syntax can also be used when the codomain is empty.
When using the assignment operator `=`, the `TensorMap` `E` is assumed to exist and the contents will be written to the currently allocated memory.
Note that for existing tensors, both on the left hand side and right hand side, trying to specify the indices in the domain and the codomain seperately using the above syntax, has no effect, as the bipartition of indices are already fixed by the existing object.
Hence, if `E` has been created by the previous line of code, all of the following lines are now equivalent
```julia
@tensor E[(a, b, c);(d, e)] = A[v, w, d, x] * B[y, z, c, x] * C[v, e, y, b] * D[a, w, z]
@tensor E[a, b, c, d, e] = A[v w d; x] * B[(y, z, c); (x, )] * C[v e y; b] * D[a, w, z]
@tensor E[a b; c d e] = A[v; w d x] * B[y, z, c, x] * C[v, e, y, b] * D[a w; z]
```
and none of those will or can change the partition of the indices of `E` into its codomain and its domain.

Two final remarks are in order.
Firstly, the order of the tensors appearing on the right hand side is irrelevant, as we can reorder them by using the allowed moves of the Penrose graphical calculus, which yields some crossings and a twist.
As the latter is trivial, it can be omitted, and we just use the same rules to evaluate the newly ordered tensor network.
For the particular case of matrix-matrix multiplication, which also captures more general settings by appropriotely combining spaces into a single line, we indeed find

```@raw html
<img src="../img/tensor-contractionreorder.svg" alt="tensor contraction reorder" class="color-invertible"/>
```

or thus, the following two lines of code yield the same result
```julia
@tensor C[i, j] := B[i, k] * A[k, j]
@tensor C[i, j] := A[k, j] * B[i, k]
```
Reordering of tensors can be used internally by the `@tensor` macro to evaluate the contraction in a more efficient manner.
In particular, the NCON-style of specifying the contraction gives the user control over the order, and there are other macros, such as `@tensoropt`, that try to automate this process.
There is also an `@ncon` macro and `ncon` function, an we recommend reading the [manual of TensorOperations.jl](https://quantumkithub.github.io/TensorOperations.jl/stable/) to learn more about the possibilities and how they work.

A final remark involves the use of adjoints of tensors.
The current framework is such that the user should not be too worried about the actual bipartition into codomain and domain of a given `TensorMap` instance.
Indeed, for tensor contractions the `@tensor` macro figures out the correct manipulations automatically.
However, when wanting to use the `adjoint` of an instance `t::TensorMap{T, S, N₁, N₂}`, the resulting `adjoint(t)` is an `AbstractTensorMap{T, S, N₂, N₁}` and one needs to know the values of `N₁` and `N₂` to know exactly where the `i`th index of `t` will end up in `adjoint(t)`, and hence the index order of `t'`.
Within the `@tensor` macro, one can instead use `conj()` on the whole index expression so as to be able to use the original index ordering of `t`.
For example, for `TensorMap{T, S, 1, 1}` instances, this yields exactly the equivalence one expects, namely one between the following two expressions:

```julia
@tensor C[i, j] := B'[i, k] * A[k, j]
@tensor C[i, j] := conj(B[k, i]) * A[k, j]
```

For e.g. an instance `A::TensorMap{T, S, 3, 2}`, the following two syntaxes have the same effect within an `@tensor` expression: `conj(A[a, b, c, d, e])` and `A'[d, e, a, b, c]`.

Some examples:

## Fermionic tensor contractions

TODO

## Anyonic tensor contractions

TODO
