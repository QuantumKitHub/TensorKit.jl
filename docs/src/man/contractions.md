# [Tensor contractions and tensor networks](@id ss_tensor_contraction)

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

## Fermionic tensor contractions

TODO

## Anyonic tensor contractions

TODO
