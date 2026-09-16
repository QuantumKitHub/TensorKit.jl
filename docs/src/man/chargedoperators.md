# [Constructing physical operators](@id s_chargedoperators)

```@setup chargedoperators
using TensorKit
```

The [previous pages](@ref s_tensors) explained how the data of a `TensorMap` is stored in blocks labeled by the coupled sector, and how a tensor can be constructed by assigning this block data directly.
There, the data itself was random, read off from a given dense array, or fixed by simple eigenvalue considerations; what was left open is how the block data of a physically meaningful operator should be determined in the first place.
This page works through such a case: the construction of fermionic creation operators, the elementary building blocks of many-body Hamiltonians such as the Hubbard model, as symmetric tensor maps.

The example is instructive because a creation operator is not symmetric in the naive sense: acting with it on a state changes the quantum numbers, so it cannot be represented as a symmetric map from the local Hilbert space to itself.
Instead, it is an instance of a *charged operator*, a symmetric tensor with an additional incoming leg that supplies the quantum numbers of the added particle.
We first explain in general how the fusion rules and Clebsch–Gordan coefficients of the symmetry group, together with the physical action of the operator, completely fix the block data of such an operator, and then carry out the construction explicitly for three commonly encountered symmetry choices:

1. ``\mathrm{U}(1)`` particle number conservation combined with ``\mathrm{SU}(2)`` spin rotation symmetry,
2. ``\mathrm{U}(1)`` particle number conservation combined with a ``\mathrm{U}(1)`` spin symmetry, i.e. only a single spin component is conserved,
3. no particle number conservation, but fermion parity ``\mathbb{Z}_2`` combined with ``\mathrm{SU}(2)`` spin symmetry.

For a gentler, code-first introduction to the construction of symmetric tensors we refer to the [tutorial in the appendix](@ref s_symmetric_tutorial); the present page focuses on how the block data itself follows from the symmetry.

## [The creation operator and its charged leg](@id ss_charged_leg)

### The creation operator without symmetries

For a single site of a spin-``\frac{1}{2}`` fermionic system, the local Hilbert space is four-dimensional,

```math
\mathcal{H}_{\mathrm{loc}}
=\operatorname{span}\left\{
\ket{0},\ket{\uparrow},\ket{\downarrow},\ket{\uparrow\downarrow}
\right\},
```

with the convention

```math
\ket{\uparrow}=c_{\uparrow}^{\dagger}\ket{0},
\qquad
\ket{\downarrow}=c_{\downarrow}^{\dagger}\ket{0},
\qquad
\ket{\uparrow\downarrow}
=c_{\downarrow}^{\dagger}c_{\uparrow}^{\dagger}\ket{0}.
```

The fermionic anticommutation relations then imply that the only nonzero actions of the creation operators on the local basis states are

```math
c_{\uparrow}^{\dagger}\ket{0}=\ket{\uparrow},
\quad
c_{\uparrow}^{\dagger}\ket{\downarrow}=-\ket{\uparrow\downarrow},
\quad
c_{\downarrow}^{\dagger}\ket{0}=\ket{\downarrow},
\quad
c_{\downarrow}^{\dagger}\ket{\uparrow}=\ket{\uparrow\downarrow}.
```

In the ordered basis ``(\ket{0},\ket{\uparrow},\ket{\downarrow},\ket{\uparrow\downarrow})`` the two creation operators are thus represented by the matrices

```math
c_\uparrow^\dagger
=
\begin{pmatrix}
0&0&0&0\\
1&0&0&0\\
0&0&0&0\\
0&0&-1&0
\end{pmatrix},
\qquad
c_\downarrow^\dagger
=
\begin{pmatrix}
0&0&0&0\\
0&0&0&0\\
1&0&0&0\\
0&1&0&0
\end{pmatrix}.
```

In the absence of any symmetry constraint, each of these is simply a rank-2 tensor with one incoming and one outgoing physical leg, i.e. an ordinary linear map ``c_\sigma^\dagger:\mathcal{H}_{\mathrm{loc}}\to\mathcal{H}_{\mathrm{loc}}``.

### Imposing a symmetry: the charged leg and the degeneracy tensors

When a global symmetry is present, the local Hilbert space decomposes into irreducible representations, ``\mathcal{H}_{\mathrm{loc}}\cong\bigoplus_a\left(D_a\otimes V_a\right)``, and any symmetric tensor is constrained by Schur's lemma.
In particular, a symmetric linear map from ``\mathcal{H}_{\mathrm{loc}}`` to itself takes the form

```math
T=\bigoplus_a\left(P^a\otimes I_{V_a}\right),
```

and therefore only has nonzero matrix elements between equivalent irreducible representations: the quantum numbers of a state cannot change under the action of a symmetric operator.
The creation operator, however, explicitly changes the quantum numbers by adding an electron.
The added quantum numbers must therefore enter through an additional incoming leg, which we call the *charged leg* ``V_e``, carrying the quantum numbers ``e`` of a single electron.
The creation operator thus becomes a three-legged tensor

```math
C^\dagger:\mathcal{H}_{\mathrm{loc}}\otimes V_e\to\mathcal{H}_{\mathrm{loc}},
```

and a matrix element between an input sector ``b`` and an output sector ``a`` can only be nonzero if the fusion rules allow ``a`` to appear in the fusion product ``b\times e``.
For an abelian symmetry this amounts to the addition rule ``a=b+e`` for the quantum numbers; for a non-abelian symmetry ``a`` must be contained in the decomposition of ``b\times e``.

By the Wigner–Eckart theorem, every such channel of the symmetric tensor factorizes into a degeneracy tensor ``P``, which contains all symmetry-independent data, and a structural part built from the splitting and fusion trees, which is fixed by the symmetry.
In general, a channel with input sectors ``(b,e)``, output sector ``a`` and coupled sector ``c`` decomposes as

```math
C^{\dagger}_{a;(b,e)}
=
\left(I_{D_a}\otimes X_c^a\right)
\circ
\left(P^c_{a,(b,e)}\otimes I_{V_c}\right)
\circ
\left(I_{D_b\otimes D_e}\otimes X_c^{be}\right)^{\dagger},
```

where ``X_c^{be}:V_b\otimes V_e\to V_c`` is a fusion tree and ``X_c^a:V_c\to V_a`` a splitting tree.
Because the codomain consists of a single leg carrying a single irrep ``a``, the coupled sector must coincide with the output sector, ``c=a``, and the splitting tree reduces to the identity, ``X_a^a=I_{V_a}``.
The channel decomposition thus simplifies to

```math
C^{\dagger}_{a;(b,e)}
=
\left(P^a_{a,(b,e)}\otimes I_{V_a}\right)
\circ
\left(I_{D_b\otimes D_e}\otimes X_a^{be}\right)^{\dagger},
```

or, in terms of the individual matrix elements,

```math
\left\langle a,m_a\middle|C^\dagger\middle|b,m_b;e,m_e\right\rangle
=\sum_{\mu_a,\mu_b}
\left(P^a_{a,(b,e)}\right)_{\mu_a,(\mu_b\mu_e)}
\left\langle b,m_b;e,m_e\middle|X_a^{be}\middle|a,m_a\right\rangle^*,
```

where ``\mu_a``, ``\mu_b`` and ``\mu_e`` label the degeneracy spaces of the corresponding sectors (the charged legs used below all have trivial degeneracy spaces, so ``\mu_e`` can be dropped), and ``m_a``, ``m_b``, ``m_e`` label states within the irreducible representations.
The matrix elements of the splitting tree on the right-hand side are nothing but the Clebsch–Gordan coefficients of the symmetry group, or products thereof for a direct product of groups.

This relation is the key to the construction: the left-hand side is fixed by the physical definition of the creation operator, the Clebsch–Gordan coefficients on the right-hand side are fixed by the group, and the degeneracy tensor ``P`` follows from dividing one by the other.
In the language of TensorKit.jl, the trees are exactly the fusion and splitting trees that a `TensorMap` keeps track of implicitly, and the degeneracy tensors ``P`` are precisely the data stored in the [blocks of the tensor](@ref ss_tensor_storage), so that once ``P`` is known for every allowed channel, constructing the symmetric `TensorMap` reduces to allocating a zero tensor on the correct spaces and filling its blocks.

Below we carry out this programme explicitly for the three symmetry choices listed above.
In every case, the steps are:

1. decompose the local Hilbert space into irreducible representations and identify the charged leg,
2. enumerate the fusion channels ``b\times e\to a`` that land back in the local Hilbert space,
3. evaluate the physical matrix elements and the Clebsch–Gordan coefficients, and solve for ``P``.

## [U(1) particle number and SU(2) spin symmetry](@id ss_example_u1su2)

For a system at fixed filling, with conserved total spin and unpolarized spin orientation, the Hamiltonian typically has a ``\mathrm{U}(1)`` particle number symmetry and an ``\mathrm{SU}(2)`` spin symmetry.
We label the sectors by the pair ``q=(n,j)``, where ``n`` is the particle number and ``j`` the total spin.
The local Hilbert space decomposes as

```math
\mathcal H_{\mathrm{loc}}
\cong
V_{(0,0)}\oplus V_{(1,\frac12)}\oplus V_{(2,0)},
```

where each irrep appears only once, so that all degeneracy spaces are one-dimensional.
The irrep spaces are spanned by the states ``\ket{q,m_q}``,

```math
\begin{aligned}
V_{(0,0)}
&=\operatorname{span}\left\{\ket{(0,0),0}\right\},\\
V_{(1,\frac12)}
&=\operatorname{span}\left\{
\ket{(1,\tfrac12),\tfrac12},
\ket{(1,\tfrac12),-\tfrac12}
\right\},\\
V_{(2,0)}
&=\operatorname{span}\left\{\ket{(2,0),0}\right\},
\end{aligned}
```

which we identify with the empty state, the two singly occupied states and the doubly occupied state, respectively.

The creation operator adds the quantum numbers ``e=(1,\frac{1}{2})`` of a single electron, so the charged leg carries the irrep ``V_e=V_{(1,\frac{1}{2})}`` and the creation operator is the three-legged tensor

```math
C^\dagger:
\left(V_{(0,0)}\oplus V_{(1,\frac12)}\oplus V_{(2,0)}\right)
\otimes V_{(1,\frac{1}{2})}
\to
V_{(0,0)}\oplus V_{(1,\frac12)}\oplus V_{(2,0)}.
```

Fusing every input sector with the charge sector gives

```math
(0,0)\times (1,\tfrac12)=(1,\tfrac12),\qquad
(1,\tfrac12)\times (1,\tfrac12)=(2,0)\oplus(2,1),\qquad
(2,0)\times (1,\tfrac12)=(3,\tfrac12),
```

and since the fusion product must again be a sector of the local Hilbert space, only the two channels

```math
(0,0)\times (1,\tfrac12)\to (1,\tfrac12),
\qquad
(1,\tfrac12)\times (1,\tfrac12)\to (2,0)
```

can have nonzero matrix elements.

On each of these channels the degeneracy tensor is a single number, and the physical matrix elements are fixed by the defining action of the creation operator:

```math
\begin{aligned}
\left\langle(1,\tfrac12),\pm\tfrac12\middle|C^\dagger
\middle|(0,0),0;(1,\tfrac12),\pm\tfrac12\right\rangle &= +1,\\
\left\langle(2,0),0\middle|C^\dagger
\middle|(1,\tfrac12),\pm\tfrac12;(1,\tfrac12),\mp\tfrac12\right\rangle &= \pm 1.
\end{aligned}
```

Here we have set the overall normalization to one.
Note that the second channel a priori comprises four matrix elements, one for each combination of spin orientations, but only the two cross terms survive the Pauli exclusion principle, and the fermionic anticommutation relations require them to differ by a sign.

Because the symmetry group is the direct product ``\mathrm{U}(1)\times\mathrm{SU}(2)``, whose two factors act independently, the matrix elements of the splitting tree factorize into a ``\mathrm{U}(1)`` and an ``\mathrm{SU}(2)`` Clebsch–Gordan coefficient,

```math
\left\langle (n_b,j_b),m_b;(n_e,j_e),m_e\middle|
X_{(n_a,j_a)}^{(n_b,j_b),(n_e,j_e)}
\middle|(n_a,j_a),m_a\right\rangle
= C^{n_a}_{n_b,n_e}
\left(C^{j_a}_{j_b,j_e}\right)^{m_a}_{m_b,m_e}.
```

The ``\mathrm{U}(1)`` coefficient is just a Kronecker delta, ``C^{n_a}_{n_b,n_e}=\delta_{n_a,n_b+n_e}``, enforcing particle number conservation, while the ``\mathrm{SU}(2)`` coefficients can be looked up in standard tables.
For the two channels at hand one finds

```math
\begin{aligned}
\left\langle(0,0),0;(1,\tfrac12),\pm\tfrac12\middle|
X_{(1,\frac12)}^{(0,0),(1,\frac12)}
\middle|(1,\tfrac12),\pm\tfrac12\right\rangle
&=C^{1}_{0,1}
\left(C^{\frac12}_{0,\frac12}\right)^{\pm\frac12}_{0,\pm\frac12}
=1,\\
\left\langle(1,\tfrac12),\pm\tfrac12;(1,\tfrac12),\mp\tfrac12\middle|
X_{(2,0)}^{(1,\frac12),(1,\frac12)}
\middle|(2,0),0\right\rangle
&=C^{2}_{1,1}
\left(C^{0}_{\frac12,\frac12}\right)^{0}_{\pm\frac12,\mp\frac12}
=\pm\frac{1}{\sqrt{2}}.
\end{aligned}
```

Substituting the physical matrix elements and the Clebsch–Gordan coefficients into the Wigner–Eckart relation from the previous subsection gives

```math
1
=P^{(1,\frac12)}_{(1,\frac12),\,((0,0),(1,\frac12))}\times 1,
\qquad
\pm1
=P^{(2,0)}_{(2,0),\,((1,\frac12),(1,\frac12))}
\times\left(\pm\frac{1}{\sqrt{2}}\right),
```

so that the two components of the degeneracy tensor are

```math
P^{(1,\frac12)}_{(1,\frac12),\,((0,0),(1,\frac12))}=1,
\qquad
P^{(2,0)}_{(2,0),\,((1,\frac12),(1,\frac12))}=\sqrt{2}.
```

The complete ``\mathrm{U}(1)\times\mathrm{SU}(2)`` symmetric creation operator thus reads

```math
C^\dagger_{\mathrm{U(1)\times SU(2)}}
=1\otimes\left(X_{(1,\frac12)}^{(0,0),(1,\frac12)}\right)^\dagger
+\sqrt2\otimes\left(X_{(2,0)}^{(1,\frac12),(1,\frac12)}\right)^\dagger.
```

It is instructive to count parameters: in the dense, symmetry-ignorant representation the two creation operators are two ``4\times4`` matrices, comprising ``2\times4\times4=32`` numbers.
In the symmetric representation, the structural tensors are fixed by the symmetry and only the two degeneracy parameters ``1`` and ``\sqrt{2}`` need to be stored.

The corresponding `TensorMap` is obtained by allocating a zero tensor on the appropriate spaces and filling its two blocks with precisely these degeneracy parameters:

```@repl chargedoperators
I = U1Irrep ⊠ SU2Irrep
vspace = Vect[I]((1, 1/2) => 1)
pspace = Vect[I]((0, 0) => 1, (1, 1/2) => 1, (2, 0) => 1)
e⁺ = zeros(Float64, pspace ← pspace ⊗ vspace)
block(e⁺, I(1, 1//2)) .= 1 # (0,0) × (1,1/2) → (1,1/2)
block(e⁺, I(2, 0)) .= sqrt(2) # (1,1/2) × (1,1/2) → (2,0)
e⁺
```

Note that the tensor automatically has exactly two block sectors, one for each allowed fusion channel.

## [U(1) particle number and U(1) spin symmetry](@id ss_example_u1u1)

For a system at fixed filling where only a single spin component is conserved, such that the two spin orientations are distinguishable, the relevant symmetry is ``\mathrm{U}(1)\times\mathrm{U}(1)``.
We label the sectors by ``(n,s)``, where ``n`` is the particle number and ``s`` the quantum number of the conserved spin component.
The local Hilbert space decomposes into four one-dimensional irreps,

```math
\mathcal H_{\mathrm{loc}}
\cong
V_{(0,0)}
\oplus V_{(1,\frac12)}
\oplus V_{(1,-\frac12)}
\oplus V_{(2,0)},
```

each appearing once, spanned by the states ``\ket{n,s}``:

```math
V_{(0,0)}=\operatorname{span}\{\ket{(0,0)}\},\quad
V_{(1,\frac12)}=\operatorname{span}\{\ket{(1,\tfrac12)}\},\quad
V_{(1,-\frac12)}=\operatorname{span}\{\ket{(1,-\tfrac12)}\},\quad
V_{(2,0)}=\operatorname{span}\{\ket{(2,0)}\}.
```

Since the spin orientations are now distinguishable, the two creation operators carry different charges, ``e_\uparrow=(1,\frac12)`` and ``e_\downarrow=(1,-\frac12)``, with charged legs ``V_{e_\uparrow}=V_{(1,\frac12)}`` and ``V_{e_\downarrow}=V_{(1,-\frac12)}``, and each of them is a three-legged tensor

```math
C_\sigma^\dagger:
\mathcal{H}_{\mathrm{loc}}\otimes V_{e_\sigma}
\to
\mathcal{H}_{\mathrm{loc}},
\qquad \sigma=\uparrow,\downarrow.
```

The fusion rules of the direct product of two ``\mathrm{U}(1)`` groups are simply additive,

```math
(n_b,s_b)\times(n_e,s_e)\to(n_b+n_e,s_b+s_e),
```

and fusing every input sector with each of the two charge sectors yields

```math
\begin{aligned}
(0,0)\times(1,\tfrac12)&=(1,\tfrac12),
&(1,\tfrac12)\times(1,\tfrac12)&=(2,1),\\
(1,-\tfrac12)\times(1,\tfrac12)&=(2,0),
&(2,0)\times(1,\tfrac12)&=(3,\tfrac12),\\
(0,0)\times(1,-\tfrac12)&=(1,-\tfrac12),
&(1,-\tfrac12)\times(1,-\tfrac12)&=(2,-1),\\
(1,\tfrac12)\times(1,-\tfrac12)&=(2,0),
&(2,0)\times(1,-\tfrac12)&=(3,-\tfrac12).
\end{aligned}
```

Keeping only the channels whose fusion product lies again in the local Hilbert space leaves four allowed channels, two for each spin orientation:

```math
\begin{aligned}
&(0,0)\times(1,\tfrac12)\to(1,\tfrac12),
&&(1,-\tfrac12)\times(1,\tfrac12)\to(2,0),\\
&(0,0)\times(1,-\tfrac12)\to(1,-\tfrac12),
&&(1,\tfrac12)\times(1,-\tfrac12)\to(2,0).
\end{aligned}
```

The physical matrix elements on these channels follow directly from the defining action of the creation operators,

```math
\begin{aligned}
\left\langle(1,\tfrac12)\middle|C_\uparrow^\dagger
\middle|(0,0);(1,\tfrac12)\right\rangle&=1,
&\left\langle(2,0)\middle|C_\uparrow^\dagger
\middle|(1,-\tfrac12);(1,\tfrac12)\right\rangle&=-1,\\
\left\langle(1,-\tfrac12)\middle|C_\downarrow^\dagger
\middle|(0,0);(1,-\tfrac12)\right\rangle&=1,
&\left\langle(2,0)\middle|C_\downarrow^\dagger
\middle|(1,\tfrac12);(1,-\tfrac12)\right\rangle&=1,
\end{aligned}
```

where the signs have the same origin as in the previous subsection.
The Clebsch–Gordan coefficients of the two ``\mathrm{U}(1)`` factors are Kronecker deltas, so they evaluate to one on every allowed channel.
Solving for the degeneracy tensors is therefore immediate:

```math
\begin{aligned}
1&=P_\uparrow^{(1,\frac12)}{}_{(1,\frac12),((0,0),(1,\frac12))}\times1,
&-1&=P_\uparrow^{(2,0)}{}_{(2,0),((1,-\frac12),(1,\frac12))}\times1,\\
1&=P_\downarrow^{(1,-\frac12)}{}_{(1,-\frac12),((0,0),(1,-\frac12))}\times1,
&1&=P_\downarrow^{(2,0)}{}_{(2,0),((1,\frac12),(1,-\frac12))}\times1.
\end{aligned}
```

The complete ``\mathrm{U}(1)\times\mathrm{U}(1)`` symmetric creation operators thus read

```math
\begin{aligned}
C_{\mathrm{U(1)\times U(1)},\uparrow}^\dagger
&=1\otimes\left(X_{(1,\frac12)}^{(0,0),(1,\frac12)}\right)^\dagger
-1\otimes\left(X_{(2,0)}^{(1,-\frac12),(1,\frac12)}\right)^\dagger,\\
C_{\mathrm{U(1)\times U(1)},\downarrow}^\dagger
&=1\otimes\left(X_{(1,-\frac12)}^{(0,0),(1,-\frac12)}\right)^\dagger
+1\otimes\left(X_{(2,0)}^{(1,\frac12),(1,-\frac12)}\right)^\dagger.
\end{aligned}
```

Hence, in the ``\mathrm{U}(1)\times\mathrm{U}(1)`` case, the two creation operators together require storing only ``4`` parameters.

The construction again proceeds by filling the blocks with the degeneracy parameters, this time one ``1\times1`` block per allowed channel.
For the spin-up operator:

```@repl chargedoperators
I = U1Irrep ⊠ U1Irrep
pspace = Vect[I]((0, 0) => 1, (1, 1/2) => 1, (1, -1/2) => 1, (2, 0) => 1)
vspace = Vect[I]((1, 1/2) => 1) # charged leg of a spin-up electron
e⁺_up = zeros(Float64, pspace ← pspace ⊗ vspace)
block(e⁺_up, I(1, 1//2)) .= 1 # (0,0) × (1,1/2) → (1,1/2)
block(e⁺_up, I(2, 0)) .= -1 # (1,-1/2) × (1,1/2) → (2,0)
e⁺_up
```

The spin-down operator uses the same physical space but a different charged leg:

```@repl chargedoperators
vspace = Vect[I]((1, -1/2) => 1) # charged leg of a spin-down electron
e⁺_dn = zeros(Float64, pspace ← pspace ⊗ vspace)
block(e⁺_dn, I(1, -1//2)) .= 1 # (0,0) × (1,-1/2) → (1,-1/2)
block(e⁺_dn, I(2, 0)) .= 1 # (1,1/2) × (1,-1/2) → (2,0)
e⁺_dn
```

## [Fermion parity ℤ₂ and SU(2) spin symmetry](@id ss_example_z2su2)

When the total spin is conserved but the Hamiltonian contains terms that change the particle number by an even amount, such as pairing terms, the particle number ``\mathrm{U}(1)`` symmetry is broken down to the fermion parity ``\mathbb{Z}_2``.
(Terms that change the particle number by an odd amount would break the ``\mathbb{Z}_2`` symmetry as well.)
We label the sectors by ``(p,j)``, where ``p=+,-`` denotes even and odd parity and ``j`` is the total spin.

Within the local Hilbert space, the empty state and the doubly occupied state are both singlets of even parity: they transform as two isomorphic copies of the irrep ``(+,0)``.
The two singly occupied states form a single spin-``\frac12`` doublet of odd parity.
Hence the local space decomposes as

```math
\mathcal H_{\mathrm{loc}}
\cong
\left(D_{+,0}\otimes V_{+,0}\right)
\oplus
\left(D_{-,\frac12}\otimes V_{-,\frac12}\right),
\qquad
d_{+,0}=2,\quad d_{-,\frac12}=1,
```

with irrep and degeneracy spaces

```math
\begin{aligned}
V_{+,0}&=\operatorname{span}\{\ket{(+,0),0}\},
&
V_{-,\frac12}&=\operatorname{span}\left\{
\ket{(-,\tfrac12),\tfrac12},
\ket{(-,\tfrac12),-\tfrac12}\right\},\\
D_{+,0}&=\operatorname{span}\{\ket{\mu_{1}},\ket{\mu_{2}}\},
&
D_{-,\frac12}&=\operatorname{span}\{\ket{\nu}\},
\end{aligned}
```

where ``\ket{\mu_1}`` and ``\ket{\mu_2}`` correspond to the empty and the doubly occupied state, respectively.

The creation operator adds the quantum numbers ``e=(-,\frac12)``, so the charged leg is ``V_e=V_{(-,\frac12)}`` and the creation operator is the three-legged tensor

```math
C^\dagger:
\left[\left(D_{+,0}\otimes V_{+,0}\right)\oplus
\left(D_{-,\frac12}\otimes V_{-,\frac12}\right)\right]
\otimes V_{(-,\frac12)}
\to
\left(D_{+,0}\otimes V_{+,0}\right)\oplus
\left(D_{-,\frac12}\otimes V_{-,\frac12}\right).
```

The parity fuses by multiplication while the spin fuses by the usual angular momentum rules, giving

```math
V_{+,0}\otimes V_{-,\frac12}\cong V_{-,\frac12},
\qquad
V_{-,\frac12}\otimes V_{-,\frac12}
\cong V_{+,0}\oplus V_{+,1}.
```

Since the local Hilbert space contains no ``(+,1)`` sector, only the two channels

```math
(+,0)\times(-,\tfrac12)\to(-,\tfrac12),
\qquad
(-,\tfrac12)\times(-,\tfrac12)\to(+,0)
```

contribute.

The new ingredient compared to the previous cases is that the degeneracy space ``D_{+,0}`` is two-dimensional, so the degeneracy tensors are genuine matrices rather than scalars.
For the channel ``(+,0)\times(-,\frac12)\to(-,\frac12)``, ``P`` maps ``D_{+,0}\otimes D_e\cong\mathbb{C}^2`` to ``D_{-,\frac12}\cong\mathbb{C}`` and is thus a row vector, while for the channel ``(-,\frac12)\times(-,\frac12)\to(+,0)`` it maps ``D_{-,\frac12}\otimes D_e\cong\mathbb{C}`` to ``D_{+,0}\cong\mathbb{C}^2`` and is thus a column vector.
The physical matrix elements, following from the defining action of the creation operator, are

```math
\left(\bra{\nu}\otimes\bra{(-,\tfrac12),\pm\tfrac12}\right)C^\dagger
\left[\left(\ket{\mu_i}\otimes\ket{(+,0),0}\right)
\otimes\ket{(-,\tfrac12),\pm\tfrac12}\right]
=\delta_{i1},
\qquad i=1,2,
```

for the first channel, i.e. the row vector ``\begin{pmatrix}1&0\end{pmatrix}``, and

```math
\left(\bra{\mu_i}\otimes\bra{(+,0),0}\right)C^\dagger
\left[\left(\ket{\nu}\otimes\ket{(-,\tfrac12),\pm\tfrac12}\right)
\otimes\ket{(-,\tfrac12),\mp\tfrac12}\right]
=\pm\,\delta_{i2},
\qquad i=1,2,
```

for the second channel, i.e. the column vector ``\begin{pmatrix}0&\pm1\end{pmatrix}^\top``, where the signs have the same origin as before.

Since the symmetry is again a direct product, ``\mathbb{Z}_2\times\mathrm{SU}(2)``, the matrix elements of the splitting tree factorize as

```math
\left\langle(p_b,j_b),m_b;(p_e,j_e),m_e\middle|
X_{(p_a,j_a)}^{(p_b,j_b),(p_e,j_e)}
\middle|(p_a,j_a),m_a\right\rangle
=C^{p_a}_{p_b,p_e}
\left(C^{j_a}_{j_b,j_e}\right)^{m_a}_{m_b,m_e},
```

with ``C^{p_a}_{p_b,p_e}=\delta_{p_a,p_b p_e}`` and the ``\mathrm{SU}(2)`` Clebsch–Gordan coefficients from the standard tables.
For the two channels at hand one finds

```math
\begin{aligned}
\left\langle (+,0),0;(-,\tfrac12),\pm\tfrac12\middle|
X_{(-,\frac12)}^{(+,0),(-,\frac12)}
\middle|(-,\tfrac12),\pm\tfrac12\right\rangle
&=\left(C^{\frac12}_{0,\frac12}\right)^{\pm\frac12}_{0,\pm\frac12}=1,\\
\left\langle(-,\tfrac12),\pm\tfrac12;(-,\tfrac12),\mp\tfrac12
\middle|X_{(+,0)}^{(-,\frac12),(-,\frac12)}
\middle|(+,0),0\right\rangle
&=\left(C^{0}_{\frac12,\frac12}\right)^{0}_{\pm\frac12,\mp\frac12}
=\pm\frac{1}{\sqrt2}.
\end{aligned}
```

Substituting everything into the Wigner–Eckart relation gives

```math
\begin{pmatrix}1&0\end{pmatrix}
=P^{(-,\frac12)}_{(-,\frac12),((+,0),(-,\frac12))}\times 1,
\qquad
\begin{pmatrix}0\\ \pm1\end{pmatrix}
=P^{(+,0)}_{(+,0),((-,\frac12),(-,\frac12))}
\times\left(\pm\frac{1}{\sqrt2}\right),
```

so that the degeneracy tensors are

```math
P^{(-,\frac12)}_{(-,\frac12),((+,0),(-,\frac12))}
=\begin{pmatrix}1&0\end{pmatrix},
\qquad
P^{(+,0)}_{(+,0),((-,\frac12),(-,\frac12))}
=\begin{pmatrix}0\\ \sqrt{2}\end{pmatrix}.
```

The complete ``\mathbb{Z}_2\times\mathrm{SU}(2)`` symmetric creation operator thus reads

```math
C^\dagger_{\mathbb Z_2\times\mathrm{SU(2)}}
=\begin{pmatrix}1&0\end{pmatrix}\otimes
\left(X_{(-,\frac12)}^{(+,0),(-,\frac12)}\right)^\dagger
+\begin{pmatrix}0\\ \sqrt{2}\end{pmatrix}\otimes
\left(X_{(+,0)}^{(-,\frac12),(-,\frac12)}\right)^\dagger.
```

Hence, also in the ``\mathbb{Z}_2\times\mathrm{SU}(2)`` case, only ``4`` independent parameters need to be stored.

In the code, the even-parity sector ``(+,0)`` corresponds to `FermionParity(0)` and the odd-parity sector ``(-,\frac12)`` to `FermionParity(1)`; the multiplicity ``d_{+,0}=2`` simply appears as the degeneracy of the corresponding sector in the space.
The two blocks are now filled with the degeneracy matrices found above:

```@repl chargedoperators
I = FermionParity ⊠ SU2Irrep
vspace = Vect[I]((1, 1/2) => 1)
pspace = Vect[I]((0, 0) => 2, (1, 1/2) => 1)
e⁺ = zeros(Float64, pspace ← pspace ⊗ vspace)
block(e⁺, I(1, 1//2)) .= [1.0 0.0] # 1 row, 2 cols
block(e⁺, I(0, 0)) .= [0.0, sqrt(2)] # 2 rows, 1 col
e⁺
```

The block labeled by `I(1, 1//2)` corresponds to the channel ``(+,0)\times(-,\frac12)\to(-,\frac12)``: it has a single row from the output degeneracy ``d_{-,\frac12}=1`` and two columns from the input degeneracy ``d_{+,0}\times d_e=2``.
Likewise, the block labeled by `I(0, 0)` corresponds to the channel ``(-,\frac12)\times(-,\frac12)\to(+,0)`` and has size ``2\times1``.