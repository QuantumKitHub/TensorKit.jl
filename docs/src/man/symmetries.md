# [Symmetries](@id s_symmetries)

```@setup sectors
using TensorKit
import LinearAlgebra
```

## Symmetries and symmetric tensors

When a physical system exhibits certain symmetries, it can often be described using tensors that transform covariantly with respect to the corresponding symmetry group, where this group acts as a tensor product of group actions on every tensor index separately.
The group action on a single index, or thus, on the corresponding vector space, can be decomposed into irreducible representations (irreps).
Here, we restrict to unitary representations, and thus assume that the corresponding vector spaces also have a natural Euclidean inner product.
In particular, the Euclidean inner product between two vectors is invariant under the group action and thus transforms according to the trivial representation of the group.

The corresponding vector spaces will be canonically represented as ``V = ⨁_a ℂ^{n_a} ⊗ R_{a}``, where ``a`` labels the different irreps, ``n_a`` is the number of times irrep ``a`` appears and ``R_a`` is the vector space associated with irrep ``a``.
Irreps are also known as spin sectors (in the case of ``\mathsf{SU}_2``) or charge sectors (in the case of ``\mathsf{U}_1``), and we henceforth refer to ``a`` as a sector.
The number of times ``n_a`` that sector ``a`` appears will be referred to as the degeneracy of sector ``a`` in the space ``V``.
In fact, the approach taken by TensorKit.jl goes beyond the case of irreps of groups, and, using the language from the Appendix on [categories](@ref s_categories), sectors correspond to (equivalence classes of) simple objects in a unitary fusion or multifusion category, whereas the "representation spaces" ``V`` correspond to general (semisimple) objects in such a category.
Nonetheless, many aspects of the construction of symmetric tensors can already be appreciated by considering the representation theory of a non-abelian group such as ``\mathsf{SU}_2`` or ``\mathsf{SU}_3`` as example.
For practical reasons, we assume that there is a canonical order of the sectors, so that the vector space ``V`` is completely specified by the values of ``n_a``.

When considering a tensor product of such representation spaces, they can again be decomposed into a direct sum of "coupled" sectors and associated degeneracy spaces.
However, a non-trivial basis transformation is required to go from the tensor product basis to the basis of coupled sectors.
The gain in efficiency (both in memory occupation and computation time) obtained from using symmetric (technically: equivariant) tensor maps is that, by Schur's lemma, they are block diagonal in the basis of coupled sectors.
Hence, to exploit this block diagonal form, it is essential that we know the basis transformation from the individual (uncoupled) sectors appearing in the tensor product form of the domain and codomain, to the totally coupled sectors that label the different blocks.
We refer to the latter as block sectors.
The transformation from the uncoupled sectors in the domain (or codomain) of the tensor map to the block sector is encoded in a fusion tree (or splitting tree).
Essentially, it is a sequential application of pairwise fusion as described by the group's [Clebsch–Gordan (CG) coefficients](https://en.wikipedia.org/wiki/Clebsch–Gordan_coefficients).
However, it turns out that we do not need to know or instantiate the actual CG coefficients that make up the fusion and splitting trees.
Instead, we only need to know how the splitting and fusion trees transform under transformations such as interchanging the order of the incoming sectors or interchanging incoming and outgoing sectors.
This information is known as the topological data of the group.
It consists out of the fusion rules and the associativity relations encoded by the F-symbols, which are also known as recoupling coefficients or [6j-symbols](https://en.wikipedia.org/wiki/6-j_symbol) (more accurately, the F-symbol is actually [Racah's W-coefficients](https://en.wikipedia.org/wiki/Racah_W-coefficient) in the case of ``\mathsf{SU}_2``).

In the next three sections of the manual, we describe how the above concepts are implemented in TensorKit.jl in greater detail.
Firstly, we describe how sectors and their associated topological data are encoded using a specialized interface and type hierarchy.
The second section describes how to build spaces ``V`` composed of a direct sum of different sectors of the same type, and which operations are supported on those spaces.
In the third section, we explain the details of constructing and manipulating fusion trees.
Finally, we elaborate on the case of general fusion categories and the possibility of having fermionic or anyonic twists.

But first, on the remainder of this page, we provide a concise theoretical summary of the required data of the representation theory of a group.
We refer to the appendix on [categories](@ref s_categories), and in particular the subsection on [topological data of a unitary fusion category](@ref ss_topologicalfusion), for more details.

!!! note
    The infrastructure for defining sectors is actually implemented in a standalone package, [TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl), that is imported and reexported by TensorKit.jl.

!!! note
    On this and the next page of the manual, we assume some familiarity with the representation theory of non-abelian groups, and the structure of a symmetric tensor.
    For a more pedagogical introduction based on physical examples, we recommend reading the first appendix, which provides a [tutorial-style introduction on the construction of symmetric tensors](@ref s_symmetric_tutorial).
              

## [Representation theory and unitary fusion categories](@id ss_representationtheory)

Let the different irreps or sectors be labeled as ``a``, ``b``, ``c``, …
First and foremost, we need to specify the *fusion rules* ``a ⊗ b = ⨁ N^{ab}_{c} c`` with ``N^{ab}_{c}`` some non-negative integers.
The meaning of the fusion rules is that the space of covariant maps ``R_a ⊗ R_b → R_c`` (or vice versa ``R_c → R_a ⊗ R_b``) has dimension ``N^{ab}_c``.
In particular, there should always exists a unique trivial sector ``u`` (called the identity object ``I`` or ``1`` in the language of categories) such that ``a ⊗ u = a = u ⊗ a`` for every other sector ``a``.
Furthermore, with respect to every sector ``a`` there should exist a unique sector ``\bar{a}`` such that ``N^{a\bar{a}}_{u} = 1``, whereas for all ``b \neq \bar{a}``, ``N^{ab}_{u} = 0``.
For irreps of groups, ``\bar{a}`` corresponds to the complex conjugate of the representation ``a``, or some representation isomorphic to it.
For example, for the representations of ``\mathsf{SU}_2``, the trivial sector corresponds to spin zero and all irreps are self-dual (i.e. ``a = \bar{a}``), meaning that the conjugate representation is isomorphic to the non-conjugated one (they are however not equal but related by a similarity transform).

In particular, we now assume the existence of a basis for the ``N^{ab}_c``-dimensional space of covariant maps ``R_c → R_a ⊗ R_b``, which consists of unitary tensor maps ``X^{ab}_{c,μ} : R_c → R_a ⊗ R_b`` with ``μ = 1, …, N^{ab}_c`` such that

```math
X^{ab}_{c,μ})^† X^{ab}_{c,ν} = δ_{μ,ν} \mathrm{id}_{R_c}
```

and

```math
\sum_{c} \sum_{μ = 1}^{N^{ab}_c} X^{ab}_{c,μ} (X^{ab}_{c,μ})^\dagger = \mathrm{id}_{R_a ⊗ R_b}
```

The tensors ``X^{ab}_{c,μ}`` are the splitting tensors, and because we restrict to unitary representations (or unitary categories), the corresponding fusion tensors are obtained by hermitian conjugation.
Different choices of orthonormal bases would be related by a unitary basis transform within the space, i.e. acting on the multiplicity label ``μ = 1, …, N^{ab}_c``.
For ``\mathsf{SU}_2``, where ``N^{ab}_c`` is zero or one and the multiplicity labels are absent, this freedom reduces to a phase factor.
In a standard convention, the entries of ``X^{ab}_{c,μ}`` are precisely given by the CG coefficients.
However, the point is that we do not need to know the tensors ``X^{ab}_{c,μ}`` explicitly, but only the topological data of (the representation category of) the group, which describes the following transformation:

*   F-move or recoupling: the transformation from ``(a ⊗ b) ⊗ c`` to ``a ⊗ (b ⊗ c)``:

```math
(X^{ab}_{e,μ} ⊗ \mathrm{id}_c) ∘ X^{ec}_{d,ν} = ∑_{f,κ,λ} [F^{abc}_{d}]_{e,μν}^{f,κλ} (\mathrm{id}_a ⊗ X^{bc}_{f,κ}) ∘ X^{af}_{d,λ}
```

*   [Braiding](@ref ss_braiding) or permuting: the transformation from ``a ⊗ b`` to ``b ⊗ a`` as defined by ``τ_{a, b}: R_a ⊗ R_b → R_b ⊗ R_a``:

```math
τ_{R_a,R_b} ∘ X^{ab}_{c,μ} = ∑_{ν} [R^{ab}_c]^ν_μ X^{ba}_{c,ν}
```

The dimensions of the spaces ``R_a`` on which representation ``a`` acts are denoted as ``d_a`` and referred to as quantum dimensions.
In particular ``d_u = 1`` and ``d_a = d_{\bar{a}}``.
This information is also encoded in the F-symbol as ``d_a = | [F^{a \bar{a} a}_a]^u_u |^{-1}``.
Note that there are no multiplicity labels in that particular F-symbol as ``N^{a\bar{a}}_u = 1``.

There is a graphical representation associated with the fusion tensors and their manipulations, which we summarize here:

```@raw html
<img src="../img/tree-summary.svg" alt="summary" class="color-invertible"/>
```

We refer to the appendix on [category theory](@ref s_categories), and in particular the section on [topological data of a unitary fusion category](@ref ss_topologicalfusion) for further details.

Finally, for the implementation, it will be useful to distinguish between a number of different possibilities regarding the fusion rules.
If, for every ``a`` and ``b``, there is a unique ``c`` such that ``a ⊗ b = c`` (i.e. ``N^{ab}_{c} = 1`` and ``N^{ab}_{c′} = 0`` for all other ``c′``), the sector type is said to have unique fusion.
The representations of a group have this property if and only if the group multiplication law is commutative, i.e. if the group is abelian.
In that case, all spaces ``R_{a}`` associated with the representation are one-dimensional and thus trivial.
In the case of representations of non-abelian groups, or in the more general categorical case, there will always be at least one pair of sectors ``a`` and ``b`` (not necessarily distinct) for which the fusion product ``a ⊗ b`` contains more than one sector ``c`` with non-zero ``N^{ab}_c``.
In those cases, we find it useful to further distinguish between sector types for which ``N^{ab}_c`` only takes the values zero or one, such that no multiplicity labels (the Greek letters ``μ``, ... are needed), e.g. the representations of ``\mathsf{SU}_2``, and those where some ``N^{ab}_c`` are larger than one, e.g. the representations of ``\mathsf{SU}_3``.
