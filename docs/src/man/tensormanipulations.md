# [Manipulating tensors](@id s_tensormanipulations)

## [Vector space and linear algebra operations](@id ss_tensor_linalg)

`AbstractTensorMap` instances `t` represent linear maps, i.e. homomorphisms in a `𝕜`-linear
category, just like matrices. To a large extent, they follow the interface of `Matrix` in
Julia's `LinearAlgebra` standard library. Many methods from `LinearAlgebra` are (re)exported
by TensorKit.jl, and can then us be used without `using LinearAlgebra` explicitly. In all of
the following methods, the implementation acts directly on the underlying matrix blocks
(typically using the same method) and never needs to perform any basis transforms.

In particular, `AbstractTensorMap` instances can be composed, provided the domain of the
first object coincides with the codomain of the second. Composing tensor maps uses the
regular multiplication symbol as in `t = t1*t2`, which is also used for matrix
multiplication. TensorKit.jl also supports (and exports) the mutating method
`mul!(t, t1, t2)`. We can then also try to invert a tensor map using `inv(t)`, though this
can only exist if the domain and codomain are isomorphic, which can e.g. be checked as
`fuse(codomain(t)) == fuse(domain(t))`. If the inverse is composed with another tensor `t2`,
we can use the syntax `t1\t2` or `t2/t1`. However, this syntax also accepts instances `t1`
whose domain and codomain are not isomorphic, and then amounts to `pinv(t1)`, the
Moore-Penrose pseudoinverse. This, however, is only really justified as minimizing the least
squares problem if `InnerProductStyle(t) <: EuclideanProduct`.

`AbstractTensorMap` instances behave themselves as vectors (i.e. they are `𝕜`-linear) and
so they can be multiplied by scalars and, if they live in the same space, i.e. have the same
domain and codomain, they can be added to each other. There is also a `zero(t)`, the
additive identity, which produces a zero tensor with the same domain and codomain as `t`. In
addition, `TensorMap` supports basic Julia methods such as `fill!` and `copy!`, as well as
`copy(t)` to create a copy with independent data. Aside from basic `+` and `*` operations,
TensorKit.jl reexports a number of efficient in-place methods from `LinearAlgebra`, such as
`axpy!` (for `y ← α * x + y`), `axpby!` (for `y ← α * x + β * y`), `lmul!` and `rmul!` (for
`y ← α*y` and `y ← y*α`, which is typically the same) and `mul!`, which can also be used for
out-of-place scalar multiplication `y ← α*x`.

For `S = spacetype(t)` where `InnerProductStyle(S) <: EuclideanProduct`, we can compute
`norm(t)`, and for two such instances, the inner product `dot(t1, t2)`, provided `t1` and
`t2` have the same domain and codomain. Furthermore, there is `normalize(t)` and
`normalize!(t)` to return a scaled version of `t` with unit norm. These operations should
also exist for `InnerProductStyle(S) <: HasInnerProduct`, but require an interface for
defining a custom inner product in these spaces. Currently, there is no concrete subtype of
`HasInnerProduct` that is not an `EuclideanProduct`. In particular, `CartesianSpace`,
`ComplexSpace` and `GradedSpace` all have `InnerProductStyle(S) <: EuclideanProduct`.

With tensors that have `InnerProductStyle(t) <: EuclideanProduct` there is associated an
adjoint operation, given by `adjoint(t)` or simply `t'`, such that
`domain(t') == codomain(t)` and `codomain(t') == domain(t)`. Note that for an instance
`t::TensorMap{S,N₁,N₂}`, `t'` is simply stored in a wrapper called
`AdjointTensorMap{S,N₂,N₁}`, which is another subtype of `AbstractTensorMap`. This should be
mostly invisible to the user, as all methods should work for this type as well. It can be
hard to reason about the index order of `t'`, i.e. index `i` of `t` appears in `t'` at index
position `j = TensorKit.adjointtensorindex(t, i)`, where the latter method is typically not
necessary and hence unexported. There is also a plural `TensorKit.adjointtensorindices` to
convert multiple indices at once. Note that, because the adjoint interchanges domain and
codomain, we have `space(t', j) == space(t, i)'`.

`AbstractTensorMap` instances can furthermore be tested for exact (`t1 == t2`) or
approximate (`t1 ≈ t2`) equality, though the latter requires that `norm` can be computed.

When tensor map instances are endomorphisms, i.e. they have the same domain and codomain,
there is a multiplicative identity which can be obtained as `one(t)` or `one!(t)`, where the
latter overwrites the contents of `t`. The multiplicative identity on a space `V` can also
be obtained using `id(A, V)` as discussed [above](@ref ss_tensor_construction), such that
for a general homomorphism `t′`, we have
`t′ == id(codomain(t′)) * t′ == t′ * id(domain(t′))`. Returning to the case of endomorphisms
`t`, we can compute the trace via `tr(t)` and exponentiate them using `exp(t)`, or if the
contents of `t` can be destroyed in the process, `exp!(t)`. Furthermore, there are a number
of tensor factorizations for both endomorphisms and general homomorphism that we discuss
below.

Finally, there are a number of operations that also belong in this paragraph because of
their analogy to common matrix operations. The tensor product of two `TensorMap` instances
`t1` and `t2` is obtained as `t1 ⊗ t2` and results in a new `TensorMap` with
`codomain(t1 ⊗ t2) = codomain(t1) ⊗ codomain(t2)` and
`domain(t1 ⊗ t2) = domain(t1) ⊗ domain(t2)`. If we have two `TensorMap{T, S, N, 1}`
instances `t1` and `t2` with the same codomain, we can combine them in a way that is
analogous to `hcat`, i.e. we stack them such that the new tensor `catdomain(t1, t2)` has
also the same codomain, but has a domain which is `domain(t1) ⊕ domain(t2)`. Similarly, if
`t1` and `t2` are of type `TensorMap{T, S, 1, N}` and have the same domain, the operation
`catcodomain(t1, t2)` results in a new tensor with the same domain and a codomain given by
`codomain(t1) ⊕ codomain(t2)`, which is the analogy of `vcat`. Note that direct sum only
makes sense between `ElementarySpace` objects, i.e. there is no way to give a tensor product
meaning to a direct sum of tensor product spaces.

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
t3 ≈ 0.5 * t2  +  0.8 * 1.3im * t
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

In many cases, the bipartition of tensor indices (i.e. `ElementarySpace` instances) between
the codomain and domain is not fixed throughout the different operations that need to be
performed on that tensor map, i.e. we want to use the duality to move spaces from domain to
codomain and vice versa. Furthermore, we want to use the braiding to reshuffle the order of
the indices.

For this, we use an interface that is closely related to that for manipulating splitting-
fusion tree pairs, namely [`braid`](@ref) and [`permute`](@ref), with the interface

```julia
braid(t::AbstractTensorMap{T,S,N₁,N₂}, (p1, p2)::Index2Tuple{N₁′,N₂′}, levels::IndexTuple{N₁+N₂,Int})
```

and

```julia
permute(t::AbstractTensorMap{T,S,N₁,N₂}, (p1, p2)::Index2Tuple{N₁′,N₂′}; copy = false)
```

both of which return an instance of `AbstractTensorMap{T,S,N₁′,N₂′}`.

In these methods, `p1` and `p2` specify which of the original tensor indices ranging from
`1` to `N₁ + N₂` make up the new codomain (with `N₁′` spaces) and new domain (with `N₂′`
spaces). Hence, `(p1..., p2...)` should be a valid permutation of `1:(N₁+N₂)`. Note that,
throughout TensorKit.jl, permutations are always specified using tuples of `Int`s, for
reasons of type stability. For `braid`, we also need to specify `levels` or depths for each
of the indices of the original tensor, which determine whether indices will braid over or
underneath each other (use the braiding or its inverse). We refer to the section on
[manipulating fusion trees](@ref ss_fusiontrees) for more details.

When `BraidingStyle(sectortype(t)) isa SymmetricBraiding`, we can use the simpler interface
of `permute`, which does not require the argument `levels`. `permute` accepts a keyword
argument `copy`. When `copy == true`, the result will be a tensor with newly allocated data
that can independently be modified from that of the input tensor `t`. When `copy` takes the
default value `false`, `permute` can try to return the result in a way that it shares its
data with the input tensor `t`, though this is only possible in specific cases (e.g. when
`sectortype(S) == Trivial` and `(p1..., p2...) = (1:(N₁+N₂)...)`).

Both `braid` and `permute` come in a version where the result is stored in an already
existing tensor, i.e. [`braid!(tdst, tsrc, (p1, p2), levels)`](@ref) and
[`permute!(tdst, tsrc, (p1, p2))`](@ref).

Another operation that belongs under index manipulations is taking the `transpose` of a
tensor, i.e. `LinearAlgebra.transpose(t)` and `LinearAlgebra.transpose!(tdst, tsrc)`, both
of which are reexported by TensorKit.jl. Note that `transpose(t)` is not simply equal to
reshuffling domain and codomain with
`braid(t, (1:(N₁+N₂)...), reverse(domainind(tsrc)), reverse(codomainind(tsrc))))`. Indeed,
the graphical representation (where we draw the codomain and domain as a single object),
makes clear that this introduces an additional (inverse) twist, which is then compensated in
the `transpose` implementation.

```@raw html
<img src="../img/tensor-transpose.svg" alt="transpose" class="color-invertible"/>
```

In categorical language, the reason for this extra twist is that we use the left
coevaluation ``η``, but the right evaluation ``\tilde{ϵ}``, when repartitioning the indices
between domain and codomain.

There are a number of other index related manipulations. We can apply a twist (or inverse
twist) to one of the tensor map indices via [`twist(t, i; inv = false)`](@ref) or
[`twist!(t, i; inv = false)`](@ref). Note that the latter method does not store the result
in a new destination tensor, but just modifies the tensor `t` in place. Twisting several
indices simultaneously can be obtained by using the defining property

``θ_{V⊗W} = τ_{W,V} ∘ (θ_W ⊗ θ_V) ∘ τ_{V,W} = (θ_V ⊗ θ_W) ∘ τ_{W,V} ∘ τ_{V,W},``

but is currently not implemented explicitly.

For all sector types `I` with `BraidingStyle(I) == Bosonic()`, all twists are `1` and thus
have no effect. Let us start with some examples, in which we illustrate that, albeit
`permute` might act highly non-trivial on the fusion trees and on the corresponding data,
after conversion to a regular `Array` (when possible), it just acts like `permutedims`

```@repl tensors
domain(t) → codomain(t)
ta = convert(Array, t);
t′ = permute(t, (1, 2, 3, 4));
domain(t′) → codomain(t′)
convert(Array, t′) ≈ ta
t′′ = permute(t, ((4, 2, 3), (1,)));
domain(t′′) → codomain(t′′)
convert(Array, t′′) ≈ permutedims(ta, (4, 2, 3, 1))
transpose(t)
convert(Array, transpose(t)) ≈ permutedims(ta, (4, 3, 2, 1))
dot(t2, t) ≈ dot(transpose(t2), transpose(t))
transpose(transpose(t)) ≈ t
twist(t, 3) ≈ t
```

Note that `transpose` acts like one would expect on a `TensorMap{T, S, 1, 1}`. On a
`TensorMap{T, S, N₁, N₂}`, because `transpose` replaces the codomain with the dual of the
domain, which has its tensor product operation reversed, this in the end amounts in a
complete reversal of all tensor indices when representing it as a plain multi-dimensional
`Array`. Also, note that we have not defined the conjugation of `TensorMap` instances. One
definition that one could think of is `conj(t) = adjoint(transpose(t))`. However note that
`codomain(adjoint(tranpose(t))) == domain(transpose(t)) == dual(codomain(t))` and similarly
`domain(adjoint(tranpose(t))) == dual(domain(t))`, where `dual` of a `ProductSpace` is
composed of the dual of the `ElementarySpace` instances, in reverse order of tensor product.
This might be very confusing, and as such we leave tensor conjugation undefined. However,
note that we have a conjugation syntax within the context of [tensor contractions](@ref
ss_tensor_contraction).

To show the effect of `twist`, we now consider a type of sector `I` for which
`BraidingStyle(I) != Bosonic()`. In particular, we use `FibonacciAnyon`. We cannot convert
the resulting `TensorMap` to an `Array`, so we have to rely on indirect tests to verify our
results.

```@repl tensors
V1 = GradedSpace{FibonacciAnyon}(:I => 3, :τ => 2)
V2 = GradedSpace{FibonacciAnyon}(:I => 2, :τ => 1)
m = randn(Float32, V1, V2)
transpose(m)
twist(braid(m, ((2,), (1,)), (1, 2)), 1)
t1 = randn(V1 * V2', V2 * V1);
t2 = randn(ComplexF64, V1 * V2', V2 * V1);
dot(t1, t2) ≈ dot(transpose(t1), transpose(t2))
transpose(transpose(t1)) ≈ t1
```

A final operation that one might expect in this section is to fuse or join indices, and its
inverse, to split a given index into two or more indices. For a plain tensor (i.e. with
`sectortype(t) == Trivial`) amount to the equivalent of `reshape` on the multidimensional
data. However, this represents only one possibility, as there is no canonically unique way
to embed the tensor product of two spaces `V1 ⊗ V2` in a new space `V = fuse(V1 ⊗ V2)`. Such
a mapping can always be accompagnied by a basis transform. However, one particular choice is
created by the function `isomorphism`, or for `EuclideanProduct` spaces, `unitary`. Hence,
we can join or fuse two indices of a tensor by first constructing
`u = unitary(fuse(space(t, i) ⊗ space(t, j)), space(t, i) ⊗ space(t, j))` and then
contracting this map with indices `i` and `j` of `t`, as explained in the section on
[contracting tensors](@ref ss_tensor_contraction). Note, however, that a typical algorithm
is not expected to often need to fuse and split indices, as e.g. tensor factorizations can
easily be applied without needing to `reshape` or fuse indices first, as explained in the
next section.

## [Tensor factorizations](@id ss_tensor_factorization)

As tensors are linear maps, they suport various kinds of factorizations. These functions all
interpret the provided `AbstractTensorMap` instances as a map from `domain` to `codomain`,
which can be thought of as reshaping the tensor into a matrix according to the current
bipartition of the indices.

TensorKit's factorizations are provided by
[MatrixAlgebraKit.jl](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl), which is used
to supply both the interface, as well as the implementation of the various operations on the
blocks of data. For specific details on the provided functionality, we refer to its
[documentation page](https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/user_interface/decompositions/).

Finally, note that each of the factorizations takes the current partition of `domain` and
`codomain` as the *axis* along which to matricize and perform the factorization. In order to
obtain factorizations according to a different bipartition of the indices, we can use any of
the previously mentioned [index manipulations](@ref ss_indexmanipulation) before the
factorization.

Some examples to conclude this section
```@repl tensors
V1 = SU₂Space(0=>2, 1/2=>1)
V2 = SU₂Space(0=>1, 1/2=>1, 1=>1)

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

L, Q = right_orth(permute(t, ((1,), (2,3))));
codomain(L), domain(L), domain(Q)
Q * Q'
P = Q' * Q;
P ≈ P * P
t′ = permute(t, ((1,), (2, 3)));
t′ ≈ t′ * P
```

## [Bosonic tensor contractions and tensor networks](@id ss_tensor_contraction)

One of the most important operation with tensor maps is to compose them, more generally
known as contracting them. As mentioned in the section on [category theory](@ref
s_categories), a typical composition of maps in a ribbon category can graphically be
represented as a planar arrangement of the morphisms (i.e. tensor maps, boxes with lines
eminating from top and bottom, corresponding to source and target, i.e. domain and
codomain), where the lines connecting the source and targets of the different morphisms
should be thought of as ribbons, that can braid over or underneath each other, and that can
twist. Technically, we can embed this diagram in ``ℝ × [0,1]`` and attach all the
unconnected line endings corresponding objects in the source at some position ``(x,0)`` for
``x∈ℝ``, and all line endings corresponding to objects in the target at some position
``(x,1)``. The resulting morphism is then invariant under what is known as *framed
three-dimensional isotopy*, i.e. three-dimensional rearrangements of the morphism that
respect the rules of boxes connected by ribbons whose open endings are kept fixed. Such a
two-dimensional diagram cannot easily be encoded in a single line of code.

However, things simplify when the braiding is symmetric (such that over- and under-
crossings become equivalent, i.e. just crossings), and when twists, i.e. self-crossings in
this case, are trivial. This amounts to `BraidingStyle(I) == Bosonic()` in the language of
TensorKit.jl, and is true for any subcategory of ``\mathbf{Vect}``, i.e. ordinary tensors,
possibly with some symmetry constraint. The case of ``\mathbf{SVect}`` and its
subcategories, and more general categories, are discussed below.

In the case of trivial twists, we can deform the diagram such that we first combine every
morphism with a number of coevaluations ``η`` so as to represent it as a tensor, i.e. with a
trivial domain. We can then rearrange the morphism to be all ligned up horizontally, where
the original morphism compositions are now being performed by evaluations ``ϵ``. This
process will generate a number of crossings and twists, where the latter can be omitted
because they act trivially. Similarly, double crossings can also be omitted. As a
consequence, the diagram, or the morphism it represents, is completely specified by the
tensors it is composed of, and which indices between the different tensors are connect, via
the evaluation ``ϵ``, and which indices make up the source and target of the resulting
morphism. If we also compose the resulting morphisms with coevaluations so that it has a
trivial domain, we just have one type of unconnected lines, henceforth called open indices.
We sketch such a rearrangement in the following picture

```@raw html
<img src="../img/tensor-bosoniccontraction.svg" alt="tensor unitary" class="color-invertible"/>
```

Hence, we can now specify such a tensor diagram, henceforth called a tensor contraction or
also tensor network, using a one-dimensional syntax that mimicks
[abstract index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) and
specifies which indices are connected by the evaluation map using Einstein's summation
conventation. Indeed, for `BraidingStyle(I) == Bosonic()`, such a tensor contraction can
take the same format as if all tensors were just multi-dimensional arrays. For this, we rely
on the interface provided by the package
[TensorOperations.jl](https://github.com/QuantumKitHub/TensorOperations.jl).

The above picture would be encoded as
```julia
@tensor E[a,b,c,d,e] := A[v,w,d,x] * B[y,z,c,x] * C[v,e,y,b] * D[a,w,z]
```
or
```julia
@tensor E[:] := A[1,2,-4,3] * B[4,5,-3,3] * C[1,-5,4,-2] * D[-1,2,5]
```
where the latter syntax is known as NCON-style, and labels the unconnected or outgoing
indices with negative integers, and the contracted indices with positive integers.

A number of remarks are in order. TensorOperations.jl accepts both integers and any valid
variable name as dummy label for indices, and everything in between `[ ]` is not resolved in
the current context but interpreted as a dummy label. Here, we label the indices of a
`TensorMap`, like `A::TensorMap{T, S, N₁, N₂}`, in a linear fashion, where the first position
corresponds to the first space in `codomain(A)`, and so forth, up to position `N₁`. Index
`N₁ + 1` then corresponds to the first space in `domain(A)`. However, because we have applied
the coevaluation ``η``, it actually corresponds to the corresponding dual space, in
accordance with the interface of [`space(A, i)`](@ref) that we introduced [above](@ref
ss_tensor_properties), and as indiated by the dotted box around ``A`` in the above picture.
The same holds for the other tensor maps. Note that our convention also requires that we
braid indices that we brought from the domain to the codomain, and so this is only
unambiguous for a symmetric braiding, where there is a unique way to permute the indices.

With the current syntax, we create a new object `E` because we use the definition operator
`:=`. Furthermore, with the current syntax, it will be a `Tensor`, i.e. it will have a
trivial domain, and correspond to the dotted box in the picture above, rather than the
actual morphism `E`. We can also directly define `E` with the correct codomain and domain by
rather using
```julia
@tensor E[a b c;d e] := A[v,w,d,x] * B[y,z,c,x] * C[v,e,y,b] * D[a,w,z]
```
or
```julia
@tensor E[(a,b,c);(d,e)] := A[v,w,d,x] * B[y,z,c,x] * C[v,e,y,b] * D[a,w,z]
```
where the latter syntax can also be used when the codomain is empty. When using the
assignment operator `=`, the `TensorMap` `E` is assumed to exist and the contents will be
written to the currently allocated memory. Note that for existing tensors, both on the left
hand side and right hand side, trying to specify the indices in the domain and the codomain
seperately using the above syntax, has no effect, as the bipartition of indices are already
fixed by the existing object. Hence, if `E` has been created by the previous line of code,
all of the following lines are now equivalent
```julia
@tensor E[(a,b,c);(d,e)] = A[v,w,d,x] * B[y,z,c,x] * C[v,e,y,b] * D[a,w,z]
@tensor E[a,b,c,d,e] = A[v w d;x] * B[(y,z,c);(x,)] * C[v e y; b] * D[a,w,z]
@tensor E[a b; c d e] = A[v; w d x] * B[y,z,c,x] * C[v,e,y,b] * D[a w;z]
```
and none of those will or can change the partition of the indices of `E` into its codomain
and its domain.

Two final remarks are in order. Firstly, the order of the tensors appearing on the right
hand side is irrelevant, as we can reorder them by using the allowed moves of the Penrose
graphical calculus, which yields some crossings and a twist. As the latter is trivial, it
can be omitted, and we just use the same rules to evaluate the newly ordered tensor network.
For the particular case of matrix-matrix multiplication, which also captures more general
settings by appropriotely combining spaces into a single line, we indeed find

```@raw html
<img src="../img/tensor-contractionreorder.svg" alt="tensor contraction reorder" class="color-invertible"/>
```

or thus, the following two lines of code yield the same result
```julia
@tensor C[i,j] := B[i,k] * A[k,j]
@tensor C[i,j] := A[k,j] * B[i,k]
```
Reordering of tensors can be used internally by the `@tensor` macro to evaluate the
contraction in a more efficient manner. In particular, the NCON-style of specifying the
contraction gives the user control over the order, and there are other macros, such as
`@tensoropt`, that try to automate this process. There is also an `@ncon` macro and `ncon`
function, an we recommend reading the
[manual of TensorOperations.jl](https://quantumkithub.github.io/TensorOperations.jl/stable/)
to learn more about the possibilities and how they work.

A final remark involves the use of adjoints of tensors. The current framework is such that
the user should not be too worried about the actual bipartition into codomain and domain of
a given `TensorMap` instance. Indeed, for tensor contractions the `@tensor` macro figures
out the correct manipulations automatically. However, when wanting to use the `adjoint` of
an instance `t::TensorMap{T, S, N₁, N₂}`, the resulting `adjoint(t)` is an
`AbstractTensorMap{T, S, N₂, N₁}` and one needs to know the values of `N₁` and `N₂` to know
exactly where the `i`th index of `t` will end up in `adjoint(t)`, and hence the index order of `t'`.
Within the `@tensor` macro, one can instead use `conj()`
on the whole index expression so as to be able to use the original index ordering of `t`.
For example, for `TensorMap{T, S, 1, 1}` instances, this yields exactly the equivalence one
expects, namely one between the following two expressions:

```julia
@tensor C[i,j] := B'[i,k] * A[k,j]
@tensor C[i,j] := conj(B[k,i]) * A[k,j]
```

For e.g. an instance `A::TensorMap{T, S, 3, 2}`, the following two syntaxes have the same
effect within an `@tensor` expression: `conj(A[a, b, c, d, e])` and `A'[d, e, a, b, c]`.

Some examples:

## Fermionic tensor contractions

TODO

## Anyonic tensor contractions

TODO
