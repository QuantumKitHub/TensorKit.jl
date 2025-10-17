# [Introduction](@id s_intro)

Before providing a typical "user guide" and discussing the implementation of TensorKit.jl
on the next pages, let us discuss some of the rationale behind this package.

## [What is a tensor?](@id ss_whatistensor)

At the very start we should ponder about the most suitable and sufficiently general
definition of a tensor. A good starting point is the following:

*   A tensor ``t`` is an element from the
    [tensor product](https://en.wikipedia.org/wiki/Tensor_product) of ``N`` vector spaces
    ``V_1 , V_2, …, V_N``, where ``N`` is referred to as the *rank* or *order* of the
    tensor, i.e.

    ``t ∈ V_1 ⊗ V_2 ⊗ … ⊗ V_N.``

If you think of a tensor as an object with indices, a rank ``N`` tensor has ``N`` indices
where every index is associated with the corresponding vector space in that it labels a
particular basis in that space. We will return to index notation at the very end of this
manual.

As the tensor product of vector spaces is itself a vector space, this implies that a tensor
behaves as a vector, i.e. tensors from the same tensor product space can be added and
multiplied by scalars. The tensor product is only defined for vector spaces over the same
field of scalars, e.g. there is no meaning in ``ℝ^5 ⊗ ℂ^3``. When all the vector spaces in
the tensor product have an inner product, this also implies an inner product for the tensor
product space. It is hence clear that the different vector spaces in the tensor product
should have some form of homogeneity in their structure, yet they do not need to be all
equal and can e.g. have different dimensions. It goes without saying that defining the
vector spaces and their properties will be an important part of the definition of a tensor.
As a consequence, this also constitutes a significant part of the implementation, and is
discussed in the section on [Vector spaces](@ref s_spaces).

Aside from the interpretation of a tensor as a vector, we also want to interpret it as a
matrix (or more correctly, a linear map) in order to decompose tensors using linear algebra
factorisations (e.g. eigenvalue or singular value decomposition). Henceforth, we use the
term "tensor map" as follows:

*   A tensor map ``t`` is a linear map from a source or *domain*
    ``W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2}`` to a target or *codomain* ``V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}``, i.e.

    ``t:W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2} → V_1 ⊗ V_2 ⊗ … ⊗ V_{N_1}.``

A *tensor* of rank ``N`` is then just a special case of a tensor map with ``N_1 = N`` and
``N_2 = 0``. A contraction between two tensors is just a composition of linear maps (i.e.
matrix multiplication), where the contracted indices correspond to the domain of the first
tensor and the codomain of the second tensor.

In order to allow for arbitrary tensor contractions or decompositions, we need to be able to
reorganise which vector spaces appear in the domain and the codomain of the tensor map, and
in which order. This amounts to defining canonical isomorphisms between the different ways
to order and partition the tensor indices (i.e. the vector spaces). For example, a linear
map ``W → V`` is often denoted as a rank 2 tensor in ``V ⊗ W^*``, where ``W^*`` corresponds
to the dual space of ``W``. This simple example introduces two new concepts.

1.  Typical vector spaces can appear in the domain and codomain in different related forms,
    e.g. as normal space or dual space. In fact, the most generic case is that every vector
    space ``V`` has associated with it
    a [dual space](https://en.wikipedia.org/wiki/Dual_space) ``V^*``,
    a [conjugate space](https://en.wikipedia.org/wiki/Complex_conjugate_vector_space)
    ``\overline{V}`` and a conjugate dual space ``\overline{V}^*``. The four different
    vector spaces ``V``, ``V^*``, ``\overline{V}`` and ``\overline{V}^*`` correspond to the
    representation spaces of respectively the fundamental, dual or contragredient, complex
    conjugate and dual complex conjugate representation of the general linear group
    ``\mathsf{GL}(V)``. In index notation these spaces are denoted with respectively
    contravariant (upper), covariant (lower), dotted contravariant and dotted covariant
    indices.

    For real vector spaces, the conjugate (dual) space is identical to the normal (dual)
    space and we only have upper and lower indices, i.e. this is the setting of e.g.
    general relativity. For (complex) vector spaces with a sesquilinear inner product
    ``\overline{V} ⊗ V → ℂ``, the inner product allows to define an isomorphism from the
    conjugate space to the dual space (known as
    [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem)
    in the more general context of Hilbert spaces).

    In particular, in spaces with a Euclidean inner product (the setting of e.g. quantum
    mechanics), the conjugate and dual space are naturally isomorphic (because the dual and
    conjugate representation of the unitary group are the same). Again we only need upper
    and lower indices (or kets and bras).

    Finally, in ``ℝ^d`` with a Euclidean inner product, these four different spaces are all
    equivalent and we only need one type of index. The space is completely characterized by
    its dimension ``d``. This is the setting of much of classical mechanics and we refer to
    such tensors as cartesian tensors and the corresponding space as cartesian space. These
    are the tensors that can equally well be represented as multidimensional arrays (i.e.
    using some `AbstractArray{<:Real,N}` in Julia) without loss of structure.

    The implementation of all of this is discussed in [Vector spaces](@ref s_spaces).

2.  In the generic case, the identification between maps ``W → V`` and tensors in
    ``V ⊗ W^*`` is not an equivalence but an isomorphism, which needs to be defined.
    Similarly, there is an isomorphism between between ``V ⊗ W`` and ``W ⊗ V`` that can be
    non-trivial (e.g. in the case of fermions / super vector spaces). The correct formalism
    here is provided by theory of monoidal categories, the details of which are explained
    in the appendix. Nonetheless, we try to hide these canonical isomorphisms from the user
    wherever possible, and one does not need to know category theory to be able to use this
    package.

This brings us to our final (yet formal) definition

*   A tensor (map) is a homomorphism between two objects from the category ``\mathbf{Vect}``
    (or some subcategory thereof). In practice, this will be ``\mathbf{FinVect}``, the
    category of finite dimensional vector spaces. More generally even, our concept of a
    tensor makes sense, in principle, for any linear (a.k.a. ``\mathbf{Vect}``-enriched)
    monoidal category. For more details, we refer the curious reader to the appendix on
    "[Monoidal categories and their properties](@ref s_categories)".

## [Symmetries and block sparsity](@id ss_symmetries)

Physical problems often have some symmetry, i.e. the setup is invariant under the action of
a group ``\mathsf{G}`` which acts on the vector spaces ``V`` in the problem according to a
certain representation. Having quantum mechanics in mind, TensorKit.jl is so far restricted
to unitary representations. A general representation space ``V`` can be specified as the
number of times every irreducible representation (irrep) ``a`` of ``\mathsf{G}`` appears,
i.e.

``V = \bigoplus_{a} ℂ^{n_a} ⊗ R_a``

with ``R_a`` the space associated with irrep ``a`` of ``\mathsf{G}``, which itself has
dimension ``d_a`` (often called the quantum dimension), and ``n_a`` the number of times
this irrep appears in ``V``. If the unitary irrep ``a`` for ``g ∈ \mathsf{G}`` is given by
``u_a(g)``, then there exists a specific basis for ``V`` such that the group action of
``\mathsf{G}`` on ``V`` is given by the unitary representation

``u(g) = \bigoplus_{a}  𝟙_{n_a} ⊗ u_a(g)``

with ``𝟙_{n_a}`` the ``n_a × n_a`` identity matrix. The total dimension of ``V`` is given
by ``∑_a n_a d_a``.

The reason for implementing symmetries is to exploit the computation and memory gains
resulting from restricting to tensor maps ``t:W_1 ⊗ W_2 ⊗ … ⊗ W_{N_2} → V_1 ⊗ V_2 ⊗ … ⊗
V_{N_1}`` that are equivariant under the symmetry, i.e. that act as
[intertwiners](https://en.wikipedia.org/wiki/Equivariant_map#Representation_theory)
between the symmetry action on the domain and the codomain. Indeed, such tensors should be
block diagonal because of [Schur's lemma](https://en.wikipedia.org/wiki/Schur%27s_lemma),
but only after we couple the individual irreps in the spaces ``W_i`` to a joint irrep,
which is then again split into the individual irreps of the spaces ``V_i``. The basis
change from the tensor product of irreps in the (co)domain to the joint irrep is implemented
by a sequence of Clebsch–Gordan coefficients, also known as a fusion (or splitting) tree.
We implement the necessary machinery to manipulate these fusion trees under index
permutations and repartitions for arbitrary groups ``\mathsf{G}``. In particular, this fits
with the formalism of monoidal categories, and more specifically fusion categories,
and only requires the *topological* data of the group, i.e. the fusion rules of the irreps,
their quantum dimensions and the F-symbol (6j-symbol or more precisely Racah's W-symbol in
the case of ``\mathsf{SU}_2``). In particular, we don't actually need the Clebsch–Gordan
coefficients themselves (but they can be useful for checking purposes).

Hence, a second major part of TensorKit.jl is the interface and implementation for
specifying symmetries, and further details are provided in
[Sectors, representation spaces and fusion trees](@ref s_sectorsrepfusion).
