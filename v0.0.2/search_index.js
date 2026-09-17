var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#TensorKit.jl-1",
    "page": "Home",
    "title": "TensorKit.jl",
    "category": "section",
    "text": "A Julia package for large-scale tensor computations, with a hint of category theory."
},

{
    "location": "#Package-summary-1",
    "page": "Home",
    "title": "Package summary",
    "category": "section",
    "text": "TensorKit.jl aims to be a generic package for working with tensors as they appear throughout the physical sciences. TensorKit implements a parametric type Tensor (which is actually a specific case of the type TensorMap) and defines for these types a number of vector space operations (scalar multiplication, addition, norms and inner products), index operations (permutations) and linear algebra operations (multiplication, factorizations). Finally, tensor contractions can be performed using the @tensor macro from TensorOperations.jl.Currently, most effort is oriented towards tensors as they appear in the context of quantum many body physics and in particular the field of tensor networks. Such tensors often have large dimensions and take on a specific structure when symmetries are present. To deal with generic symmetries, we employ notations and concepts from category all the way down to the definition of a tensor.At the same time, TensorKit.jl focusses on computational efficiency and performance. The underlying storage of a tensor\'s data can be any DenseArray. Currently, certain operations are already multithreaded, either by distributing the different blocks in case of a structured tensor (i.e. with symmetries) or by using multithreading provided by the package Strided.jl. In the future, we also plan to investigate using GPUArrays as underlying storage for the tensors data, so as to leverage GPUs for the different operations defined on tensors."
},

{
    "location": "#Contents-of-the-manual-1",
    "page": "Home",
    "title": "Contents of the manual",
    "category": "section",
    "text": "Pages = [\"man/intro.md\", \"man/spaces.md\", \"man/sectors.md\", \"man/tensors.md\"]\nDepth = 3"
},

{
    "location": "#Library-outline-1",
    "page": "Home",
    "title": "Library outline",
    "category": "section",
    "text": "Pages = [\"lib/spaces.md\"]\nDepth = 2"
},

{
    "location": "man/intro/#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "man/intro/#Introduction-1",
    "page": "Introduction",
    "title": "Introduction",
    "category": "section",
    "text": "Before discussing the implementation and how it can be used on the following pages, let us discuss some of the rationale behind TensorKit.jl."
},

{
    "location": "man/intro/#What-is-a-tensor?-1",
    "page": "Introduction",
    "title": "What is a tensor?",
    "category": "section",
    "text": "At the very start we should ponder about the most suitable and sufficiently general definition of a tensor. A good starting point is the following:A tensor t is an element from the   tensor product of N vector spaces   V_1  V_2  V_N, where N is referred to as the rank or order of the   tensor, i.e.\nt  V_1  V_2    V_NIf you think of a tensor as an object with indices, a rank N tensor has N indices where every index is associated with the corresponding vector space in that it labels a particular basis in that space. We will return to index notation below.As the tensor product of vector spaces is itself a vector space, this implies that a tensor behaves as a vector, i.e. tensors from the same tensor product space can be added and multiplied by scalars. The tensor product is only defined for vector spaces over the same field, i.e. there is no meaning in ℝ^5  ℂ^3. When all the vector spaces in the tensor product have an inner product, this also implies an inner product for the tensor product space. It is hence clear that the different vector spaces in the tensor product should have some form of homogeneity in their structure, yet they do not need to be all equal and can e.g. have different dimensions. It goes without saying that defining the vector spaces an their properties will be an important part of the definition of a tensor. As a consequence, this also constitutes a significant part of the implementation, and is discussed in the section on Vector spaces.Aside from the interpretation of a tensor as a vector, we also want to interpret it as a matrix (or more correctly, a linear map) in order to decompose tensors using linear algebra factorisations (e.g. eigenvalue or singular value decomposition). Henceforth, we use the term \"tensor map\" as follows:A tensor map t is a linear map from a domain W_1  W_2    W_N_2 to a   codomain V_1  V_2    V_N_1, i.e.\ntW_1  W_2    W_N_2  V_1  V_2    V_N_1A tensor of rank N is then just a special case of a tensor map with N_1 = N and N_2 = 0. A contraction between two tensors is just a composition of linear maps (i.e. matrix multiplication), where the contracted indices correspond to the domain of the first tensor and the codomain of the second tensor.In order to allow for arbitrary tensor contractions or decompositions, we need to be able to reorganise which vector spaces appear in the domain and the codomain of the tensor map. This amounts to defining canonical isomorphisms between the different ways to order and partition the tensor indices (i.e. the vector spaces). For example, a linear map W  V is often denoted as a rank 2 tensor in V  W^*, where W^* corresponds to the dual space of W. This simple example introduces two new concepts.Typical vector spaces can appear in the domain and codomain in different variants, e.g.  as normal space or dual space. In fact, the most generic case is that every vector  space V has associated with it  a dual space V^*,  a conjugate space  overlineV and a conjugate dual space overlineV^*. The four different  vector spaces V, V^*, overlineV and overlineV^* correspond to the  representation spaces of respectively the fundamental, dual or contragredient, complex  conjugate and dual complex conjugate representation of the general linear group  mathsfGL(V) [tung]. In index notation these spaces are denoted with  respectively contravariant (upper), covariant (lower), dotted contravariant  and dotted covariant indices.\nFor real vector spaces, the conjugate (dual) space is identical to the normal (dual)  space and we only have upper and lower indices, i.e. this is the setting of e.g.  general relativity. For (complex) vector spaces with a sesquilinear inner product  overlineV  V  ℂ, the inner product allows to define an isomorphism from the  conjugate space to the dual space (known as  Riesz representation theorem  in the more general context of Hilbert spaces).\nIn particular, in spaces with a Euclidean inner product (the setting of e.g. quantum  mechanics), the conjugate and dual space are naturally isomorphic (because the dual and  conjugate representation of the unitary group are the same). Again we only need upper  and lower indices (or kets and bras).\nFinally, in ℝ^d with a Euclidean inner product, these four different spaces are  equivalent and we only need one type of index. The space is completely characterized by  its dimension d. This is the setting of much of classical mechanics and we refer to  such tensors as cartesian tensors and the corresponding space as cartesian space. These  are the tensors that can equally well be represented as multidimensional arrays (i.e.  using some AbstractArray{<:Real,N} in Julia) without loss of structure.\nThe implementation of all of this is discussed in Vector spaces.\nIn the generic case, the identification between maps W  V and tensors in  V  W^* is not an equivalence but an isomorphism, which needs to be defined.  Similarly, there is an isomorphism between between V  W and W  V that can be  non-trivial (e.g. in the case of fermions / super vector spaces). The correct formalism  here is provided by theory of monoidal categories. Nonetheless, we try to hide these  canonical isomorphisms from the user wherever possible.This brings us to our final (yet formal) definitionA tensor (map) is a homorphism between two objects from the category mathbfVect   (or some subcategory thereof). In practice, this will be mathbfFinVect, the   category of finite dimensional vector spaces. More generally, our concept of a tensor   makes sense, in principle, for any mathbfVect-enriched monoidal category. We refer to the section \"Monoidal categories and their properties (optional)\"."
},

{
    "location": "man/intro/#Symmetries-and-block-sparsity-1",
    "page": "Introduction",
    "title": "Symmetries and block sparsity",
    "category": "section",
    "text": "Physical problems often have some symmetry, i.e. the setup is invariant under the action of a group mathsfG which acts on the vector spaces V in the problem according to a certain representation. Having quantum mechanics in mind, TensorKit.jl restricts so far to unitary representations. A general representation space V can be specified as the number of times every irreducible representation (irrep) a of mathsfG appears, i.e.V = bigoplus_a ℂ^n_a  R_awith R_a the space associated with irrep a of mathsfG, which itself has dimension d_a (often called the quantum dimension), and n_a the number of times this irrep appears in V. If the unitary irrep a for g  mathsfG is given by u_a(g), then the group action of mathsfG on V is given by the unitary representationu(g) = bigoplus_a  𝟙_n_a  u_a(g)with 𝟙_n_a the n_a  n_a identity matrix. The total dimension of V is given by _a n_a d_a.The reason of implementing symmetries is to exploit the compuation and memory gains resulting from restricting to tensor maps tW_1  W_2    W_N_2  V_1  V_2    V_N_1 that are invariant under the symmetry (i.e. that act as intertwiners between the symmetry action on the domain and the codomain). Indeed, such tensors should be block diagonal because of Schur\'s lemma, but only after we couple the individual irreps in the spaces W_i to a joint irrep. The basis change from the tensor product of irreps in the (co)domain to the joint irrep is implemented by a sequence of Clebsch-Gordan coefficients, also known as a fusion (or splitting) tree. We implement the necessary machinery to manipulate these fusion trees under index permutations and repartitions for arbitrary groups mathsfG. In particular, this fits with the formalism of monoidal categories discussed below and only requires the topological data of the group, i.e. the fusion rules of the irreps, their quantum dimensions and the F-symbol (6j-symbol or more precisely Racah\'s W-symbol in the case of mathsfSU_2). In particular, we do not need the Clebsch-Gordan coefficients.Further details are provided in Sectors, representation spaces and fusion trees."
},

{
    "location": "man/intro/#Monoidal-categories-and-their-properties-(optional)-1",
    "page": "Introduction",
    "title": "Monoidal categories and their properties (optional)",
    "category": "section",
    "text": "The purpose of this final introductory section (which can safely be skipped), is to explain how certain concepts and terminology from the theory of monoidal categories apply in the context of tensors.  In the end, identifying tensor manipulations in TensorKit.jl with concepts from category theory is to put the diagrammatic formulation of tensor networks in the most general context on a firmer footing. The following definitions are mostly based on [selinger] and nLab, to which we refer for further information. Furthermore, we recommend the nice introduction of Beer et al.To start, a category C consists ofa class C of objects V, W, …\nfor each pair of objects V and W, a set hom(WV) of morphisms fWV\nan composition of morphisms fWV and gXW into (f  g)XV that is   associative, such that for hYX we have f  (g  h) = (f  g)  h\nfor each object V, an identity morphism mathrmid_VVV such that   f  mathrmid_W = f = mathrmid_V  f.In our case, i.e. the category mathbfVect (or some subcategory thereof), the objects are vector spaces, and the morphisms are linear maps between these vector spaces with \"matrix multiplication\" as composition. We refer to these morphisms as tensor maps exactly because there is an operation ⊗, the tensor product, that allows to combine objects into new objects. This makes mathbfVect into a monoidal category, which hasa binary operation on objects  C  C  C\na binary operation on morphisms, also denoted as , such that    hom(W_1V_1)  hom(W_2V_2)  hom(W_1  W_2 V_1  V_2)\nan identity object I\nthree families of natural isomorphisms:\n V  C, a left unitor λ_V I  V  V\n V  C, a right unitor ρ_V V  I  V\n V_1 V_2 V_3  C, an associator   α_V_1V_2V_3(V_1  V_2)  V_3  V_1  (V_2  V_3)\nthat satisfy certain consistency conditions (coherence axioms), which are known as the   triangle equation and pentagon equation.For the category mathbfVect, the identity object I is just the scalar field, which can be identified with a one-dimensional vector space. Every monoidal category is equivalent to a strict monoidal category, where the left and right unitor and associator act as the identity and their domain and codomain are truly identical. Nonetheless, for tensor maps, we do actually discriminate between V, I  V and V  I because this amounts to adding or removing an extra factor I to the tensor product structure of the (co)domain, i.e. the left and right unitor are analogous to removing extra dimensions of size 1 from an array,and an actual operation is required to do so (this has in fact led to some controversy in several programming languages that provide native support for multidimensional arrays). For what concerns the associator, the distinction between (V_1  V_2)  V_3 and V_1  (V_2  V_3) is typically absent for simple tensors or multidimensional arrays. However, this grouping can be taken to indicate how to build the fusion tree for coupling irreps to a joint irrep in the case of symmetric tensors. As such, going from one to the other requires a recoupling (F-move) which has a non-trivial action on the reduced blocks. We return to this in the discussion of symmetric tensors.With these definitions, we have the minimal requirements for defining tensor maps. In principle, we could use a more general definition and define tensor maps as morphism of any monoidal category where the hom-sets are themselves vector spaces, such that we can add morphisms and multiply them with scalars. Such categories are called mathbfVect-enriched.In order to make tensor (maps) useful and to define operations with them, we can now introduce additional structure or quantifiers to the monoidal category for which they are the morphisms."
},

{
    "location": "man/intro/#Braiding-1",
    "page": "Introduction",
    "title": "Braiding",
    "category": "section",
    "text": "To reorder tensor indices, or, equivalently, to reorder objects in the tensor product V_1  V_2   V_N, we need at the very least a braided monoidal category which has,  V W  C, a braiding σ_VW VW  WV. There is a consistency condition between the braiding and the associator known as the hexagon equation. However, for general braidings, there is no unique choice to identify a tensor in VW and WV, as any of the maps σ_VW, σ_WV^-1, σ_VW  σ_WV  σ_VW, …  and are all different. In order for there to be a unique map from V_1  V_2   V_N to any permutation of the objects in this tensor product, the braiding needs to be symmetric, i.e. σ_VW = σ_WV^-1 or, equivalently σ_WV  σ_VW = mathrmid_VW.The resulting category is also referred to as a symmetric monoidal category. In a graphical representation, it means that there is no distinction between over- and under- crossings and, as such, lines can just cross.For a simple cartesian tensor, permuting the tensor indices is equivalent to applying Julia\'s function permutedims on the underlying data. Less trivial braiding implementations arise in the context of symmetric tensors (where the fusion tree needs to be reordered) or in the case of fermions (described using so-called super vector spaces).We can extend a braided category with a twist θ_V, i.e. a family of isomorphisms θ_VVV that satisfy θ_VW = σ_WV  (θ_W  θ_V)  σ_VW and the resulting category is called a balanced monoidal category. The corresponding graphical representation is that where objects are denoted by ribbons instead of lines, and a twist is consistent with the graphical representation of a twisted ribbon and how it combines with braidings."
},

{
    "location": "man/intro/#Duals-1",
    "page": "Introduction",
    "title": "Duals",
    "category": "section",
    "text": "For tensor maps, the braiding structure only allows to reorder the objects within the domain or within the codomain separately. An autonomous or rigid monoidal category is one where objects have duals, defined via an exact pairing, i.e. two families of canonical maps, the unit η_V I  V  V^* and the co-unit ϵ_V V^*  V  I that satisfy the \"snake rules\"ρ_V  (mathrmid_V  ϵ_V)  (η_V  mathrmid_V)  λ_V^-1 = mathrmid_Vλ_V^*^-1  (ϵ_V  mathrmid_V^*)  (mathrmid_V^*  η_V)  ρ_V^*^-1 = mathrmid_V^*Given a morphism tWV, we can now identify it with (t  mathrmid_W^*)  η_W to obtain a morphism IVW^*. For the category mathbfVect, this is the identification between linear maps WV and tensors in VW^*. In particular, for complex vector spaces, using a bra-ket notation and a generic basis n for V and dual basis m for V^* (such that mn = δ_mn), the unit is η_Vℂ  V  V^*λ  λ _n n  n and the co-unit is ϵ_VV^*  V  ℂ m  n  δ_mn. Note that this does not require an inner product, i.e. no mapping from n to n was defined. Furthermore, note that we used the physics convention, whereas mathematicians would typically interchange the order of V and V^* as they appear in the codomain of the unit and in the domain of the co- unit.For a general tensor map tW_1  W_2    W_N_2  V_1  V_2    V_N_1, by successively applying η_W_N_2, η_W_N_2-1, …, η_W_1 (and the left or right unitor) but no braiding, we obtain a tensor in V_1  V_2    V_N_1  W_N_2^*    W_1^*. It does makes sense to define or identify (W_1  W_2    W_N_2)^* = W_N_2^*    W_1^*.In general categories, one can distinguish between a left and right dual, but we always assume that both objects are naturally isomorphic. Equivalently, V^**  V and the category is said to be  pivotal. For every morphism fWV, there is then a well defined notion of a transpose (also called adjoint mate) f^*V^*  W^* asf^* = λ_W^*  (ϵ_V  mathrmid_W^*)  (mathrmid_V^*  f  mathrmid_W^*)  (mathrmid_V^*  η_W)  ρ_V^*^-1f^* = ρ_W^*  (mathrmid_W^*  ϵ_V^*)  (mathrmid_V^*  f  mathrmid_W^*)  (η_W^*  mathrmid_V^*)  λ_V^*^-1and both definitions coincide (which is not the case if the category is not pivotal). In a graphical representation, this means that boxes (representing tensor maps or morphisms more generally) can be rotated. The transpose corresponds to a 180˚ rotation (either way).A braiding σ_VV^* provides a particular way to construct an maps ϵ_V^* = ϵ_V  σ_VV^*  VV^*  I and η_V^* = σ_V^*V^-1 circ η_V I V^*V, but these maps are not canonical for general braidings, so that a braided autonomous category is not automatically pivotal. A category that is both braided and pivotal automatically has a twist (and is thus balanced), vice versa a balanced autonomous category is automatically pivotal. However, the graphical representation using ribbons is only consistent if we furthermore have θ_V^* = θ_V^* (i.e. the transpose), in which case the category is said to be tortile or also a ribbon category.In the case of a symmetric braiding, most of these difficulties go away and the pivotal structure follows. A symmetric monoidal category with duals is known as a compact closed category."
},

{
    "location": "man/intro/#Adjoints-1",
    "page": "Introduction",
    "title": "Adjoints",
    "category": "section",
    "text": ""
},

{
    "location": "man/intro/#Bibliography-1",
    "page": "Introduction",
    "title": "Bibliography",
    "category": "section",
    "text": "[tung]:     Tung, W. K. (1985). Group theory in physics: an introduction to symmetry\n        principles, group representations, and special functions in classical and\n        quantum physics.\n        World Scientific Publishing Company.[selinger]: Selinger, P. (2010). A survey of graphical languages for monoidal\n        categories.\n        In New structures for physics (pp. 289-355). Springer, Berlin, Heidelberg.[beer]:     From categories to anyons: a travelogue\n        Kerstin Beer, Dmytro Bondarenko, Alexander Hahn, Maria Kalabakov, Nicole\n        Knust, Laura Niermann, Tobias J. Osborne, Christin Schridde, Stefan\n        Seckmeyer, Deniz E. Stiege- mann, and Ramona Wolf\n        [https://arxiv.org/pdf/1811.06670.pdf](https://arxiv.org/pdf/1811.06670.pdf)"
},

{
    "location": "man/spaces/#",
    "page": "Vector spaces",
    "title": "Vector spaces",
    "category": "page",
    "text": ""
},

{
    "location": "man/spaces/#Vector-spaces-1",
    "page": "Vector spaces",
    "title": "Vector spaces",
    "category": "section",
    "text": "using TensorKitFrom the Introduction, it should be clear that an important aspect in the definition of a tensor (map) is specifying the vector spaces and their structure in the domain and codomain of the map. The starting point is an abstract type VectorSpaceabstract type VectorSpace endwhich serves in a sense as the category mathbfVect. All instances of subtypes of VectorSpace will represent vector spaces. In particular, we define two abstract subtypesabstract type ElementarySpace{𝕜} <: VectorSpace end\nconst IndexSpace = ElementarySpace\n\nabstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace endHere, ElementarySpace is a super type for all vector spaces that can be associated with the individual indices of a tensor, as hinted to by its alias IndexSpace. It is parametrically dependent on 𝕜, the field of scalars (see the next section on Fields).On the other hand, subtypes of CompositeSpace{S} where S<:ElementarySpace are composed of a number of elementary spaces of type S. So far, there is a single concrete type ProductSpace{S,N} that represents the homogeneous tensor product of N vector spaces of type S. Its properties are discussed in the section on Composite spaces, together with possible extensions for the future."
},

{
    "location": "man/spaces/#Fields-1",
    "page": "Vector spaces",
    "title": "Fields",
    "category": "section",
    "text": "Vector spaces are defined over a field of scalars. We define a type hierarchy to specify the scalar field, but so far only support real and complex numbers, viaabstract type Field end\n\nstruct RealNumbers <: Field end\nstruct ComplexNumbers <: Field end\n\nconst ℝ = RealNumbers()\nconst ℂ = ComplexNumbers()Note that ℝ and ℂ can be typed as \\bbR+TAB and \\bbC+TAB. One reason for defining this new type hierarchy instead of recycling the types from Julia\'s Number hierarchy is to introduce some syntactic suggar without commiting type piracy. In particular, we now have3 ∈ ℝ\n5.0 ∈ ℂ\n5.0+1.0*im ∈ ℝ\nFloat64 ⊆ ℝ\nComplexF64 ⊆ ℂ\nℝ ⊆ ℂ\nℂ ⊆ ℝand furthermore ––probably more usefully–– ℝ^n and ℂ^n create specific elementary vector spaces as described in the next section. The underlying field of a vector space or tensor a can be obtained with field(a)."
},

{
    "location": "man/spaces/#Elementary-vector-spaces-1",
    "page": "Vector spaces",
    "title": "Elementary vector spaces",
    "category": "section",
    "text": "As mentioned at the beginning of this section, vector spaces that are associated with the individual indices of a tensor should be implemented as subtypes of ElementarySpace. As the domain and codomain of a tensor map will be the tensor product of such objects which all have the same type, it is important that related vector spaces, e.g. the dual space, are objects of the same concrete type (i.e. with the same type parameters in case of a parametric type). In particular, every ElementarySpace should implement the following methodsdim(::ElementarySpace) -> ::Int returns the dimension of the space as an Int\ndual{S<:ElementarySpace}(::S) -> ::S returns the   dual space dual(V), using an instance of   the same concrete type (i.e. not via type parameters); this should satisfy   dual(dual(V)==V\nconj{S<:ElementarySpace}(::S) -> ::S returns the   complex conjugate space   conj(V), using an instance of the same concrete type (i.e. not via type parameters);   this should satisfy conj(conj(V))==V and we automatically have   conj(V::ElementarySpace{ℝ}) = V.For convenience, the dual of a space V can also be obtained as V\'.There is concrete type GeneralSpace which is completely characterized by its field 𝕜, its dimension and whether its the dual and/or complex conjugate of 𝕜^d.struct GeneralSpace{𝕜} <: ElementarySpace{𝕜}\n    d::Int\n    dual::Bool\n    conj::Bool\nendWe furthermore define the abstract typeabstract InnerProductSpace{𝕜} <: ElementarySpace{𝕜}to contain all vector spaces V which have an inner product and thus a canonical mapping from dual(V) to V (for 𝕜 ⊆ ℝ) or from dual(V) to conj(V) (otherwise). This mapping is provided by the metric, but no further support for working with metrics is currently implemented.Finally there isabstract EuclideanSpace{𝕜} <: InnerProductSpace{𝕜}to contain all spaces V with a standard Euclidean inner product (i.e. where the metric is the identity). These spaces have the natural isomorphisms dual(V) == V (for 𝕜 == ℝ) or dual(V) == conj(V) (for 𝕜 == ℂ). In particular, we have two concrete typesimmutable CartesianSpace <: EuclideanSpace{ℝ}\n    d::Int\nend\nimmutable ComplexSpace <: EuclideanSpace{ℂ}\n  d::Int\n  dual::Bool\nendto represent the Euclidean spaces ℝ^d or ℂ^d without further inner structure. They can be created using the syntax ℝ^d and ℂ^d, or (ℂ^d)\'for the dual space of the latter. Note that the brackets are required because of the precedence rules, since d\' == d for d::Integer. Some examples:dim(ℝ^10)\n(ℝ^10)\' == ℝ^10\nisdual((ℂ^5))\nisdual((ℂ^5)\')\nisdual((ℝ^5)\')\ndual(ℂ^5) == (ℂ^5)\' == conj(ℂ^5)We refer to the next section on Sectors, representation spaces and fusion trees for further information about RepresentationSpace, which is a subtype of EuclideanSpace{ℂ} with an inner structure corresponding to the irreducible representations of a group."
},

{
    "location": "man/spaces/#Composite-spaces-1",
    "page": "Vector spaces",
    "title": "Composite spaces",
    "category": "section",
    "text": "Composite spaces are vector spaces that are built up out of individual elementary vector spaces. The most prominent (and currently only) example is a tensor product of N elementary spaces of the same type S, which is implemented asstruct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}\n    spaces::NTuple{N, S}\nendGiven some V1::S, V2::S, V3::S of the same type S<:ElementarySpace, we can easily construct ProductSpace{S,3}((V1,V2,V3)) as ProductSpace(V1,V2,V3) or using V1 ⊗ V2 ⊗ V3, where ⊗ is simply obtained by typing \\otimes+TAB. In fact, for convenience, also the regular multiplication operator * acts as tensor product between vector spaces, and as a consequence so does raising a vector space to a positive integer power, i.e.V1 = ℂ^2\nV2 = ℂ^3\nV1 ⊗ V2 ⊗ V1\' == V1 * V2 * V1\' == ProductSpace(V1,V2,V1\') == ProductSpace(V1,V2) ⊗ V1\'\nV1^3\ndim(V1 ⊗ V2)\ndims(V1 ⊗ V2)\ndual(V1 ⊗ V2)Here, the new function dims maps dim to the individual spaces in a ProductSpace and returns the result as a tuple. Note that the rationale for the last result was explained in the subsection Duals of Properties of monoidal categories.Following Julia\'s Base library, the function one applied to a ProductSpace{S,N} returns the multiplicative identity, which is ProductSpace{S,0}. The same result is obtained when acting on an instance V of S::ElementarySpace directly, however note that V ⊗ one(V) will yield a ProductSpace{S,1}(V) and not V itself. Similar to Julia Base, one also works in the type domain.In the future, other CompositeSpace types could be added. For example, the wave function of an N-particle quantum system in first quantization would require the introduction of a SymmetricSpace{S,N} or a AntiSymmetricSpace{S,N} for bosons or fermions respectively, which correspond to the symmetric (permutation invariant) or antisymmetric subspace of V^N, where V::S represents the Hilbert space of the single particle system. Other domains, like general relativity, might also benefit from tensors living in a subspace with certain symmetries under specific index permutations."
},

{
    "location": "man/spaces/#Some-more-functionality-1",
    "page": "Vector spaces",
    "title": "Some more functionality",
    "category": "section",
    "text": "Some more convenience functions are provided for the euclidean spaces CartesianSpace and ComplexSpace, as well as for RepresentationSpace discussed in the next section. All functions below that act on more than a single elementary space, are only defined when the different spaces are of the same concrete subtype S<:ElementarySpaceThe function fuse(V1, V2, ...) or fuse(V1 ⊗ V2 ⊗ ...) returns an elementary space that is isomorphic to V1 ⊗ V2 ⊗ ..., in the sense that a unitary tensor map can be constructed between those spaces, e.g. from W = V1 ⊗ V2 ⊗ ... to V = fuse(V1 ⊗ V2 ⊗ ...). The function flip(V1) returns a space that is isomorphic to V1 but has isdual(flip(V1)) == isdual(V1\'), i.e. if V1 is a normal space than flip(V1) is a dual space. Again, isomorphism here implies that a unitary map (but there is no canonical choice) can be constructed between both spaces. flip(V1) is different from dual(V1) in the case of RepresentationSpace. It is useful to flip a tensor index from a ket to a bra (or vice versa), by contracting that index with a unitary map from V1 to flip(V1). We refer to [Index operations](@ref) for further information. Some examples:fuse(ℝ^5, ℝ^3)\nfuse(ℂ^3, (ℂ^5)\', ℂ^2)\nflip(ℂ^4)We also define the direct sum V1 and V2 as V1 ⊕ V2, where ⊕ is obtained by typing \\oplus+TAB. This is possible only if isdual(V1) == isdual(V2). With a little pun on Julia Base, oneunit applied to an elementary space (in the value or type domain) returns the one-dimensional space, which is isomorphic to the scalar field of the spaceitself. Some examples illustrate this betterℝ^5 ⊕ ℝ^3\nℂ^5 ⊕ ℂ^3\nℂ^5 ⊕ (ℂ^3)\'\noneunit(ℝ^3)\nℂ^5 ⊕ oneunit(ComplexSpace)\noneunit((ℂ^3)\')\n(ℂ^5) ⊕ oneunit((ℂ^5))\n(ℂ^5)\' ⊕ oneunit((ℂ^5)\')For two spaces V1 and V2, min(V1,V2) returns the space with the smallest dimension, whereas max(V1,V2) returns the space with the largest dimension, as illustrated bymin(ℝ^5, ℝ^3)\nmax(ℂ^5, ℂ^3)\nmax(ℂ^5, (ℂ^3)\')Again, we impose isdual(V1) == isdual(V2). Again, the use of these methods is to construct unitary or isometric tensors that map between different spaces, which will be elaborated upon in the section on Tensors."
},

{
    "location": "man/sectors/#",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Sectors, representation spaces and fusion trees",
    "category": "page",
    "text": ""
},

{
    "location": "man/sectors/#Sectors,-representation-spaces-and-fusion-trees-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Sectors, representation spaces and fusion trees",
    "category": "section",
    "text": "using TensorKitSymmetries in a physical system often result in tensors which are invariant under the action of the symmetry group, where this group acts as a tensor product of group actions on every tensor index separately. The group action on a single index, or thus, on the corresponding vector space, can be decomposed into irreducible representations (irreps). Here, we restrict to unitary representations, such that the corresponding vector spaces also have a natural Euclidean inner product. In particular, the Euclidean inner product between two vectors is invariant under the group action and thus transformas according to the trivial representation of the group.The corresponding vector spaces will be canonically represented as V = _a ℂ^n_a  R_a, where a labels the different irreps, n_a is the number of times irrep a appears and R_a is the vector space associated with irrep a. Irreps are also known as spin sectors (in the case of mathsfSU_2) or charge sectors (in the case of \\mathsf{U}_1), and we henceforth refer to a as a sector. As is briefly discussed below, the approach we follow does in fact go beyond the case of irreps of groups, and sectors would more generally correspond to simple objects in a (ribbon) fusion category. Nonetheless, every step can be appreciated by using the representation theory of mathsfSU_2 or mathsfSU_3 as example. The vector space V is completely specified by the values of n_a.The gain in efficiency (both in memory occupation and computation time) obtained from using symmetric tensor maps is that, by Schur\'s lemma, they are block diagonal in the basis of coupled sectors. To exploit this block diagonal form, it is however essential that we know the basis transform from the individual (uncoupled) sectors appearing in the tensor product form of the domain and codomain, to the totally coupled sectors that label the different blocks. We refer to the latter as block sectors. The transformation from the uncoupled sectors in the domain (or codomain) of the tensor map to the block sector is encoded in a fusion tree (or splitting tree). Essentially, it is a sequential application of pairwise fusion as described by the group\'s Clebsch-Gordan (CG) coefficients. However, it turns out that we do not need the actual CG coefficients, but only how they transform under transformations such as interchanging the order of the incoming irreps or interchanging incoming and outgoing irreps. This information is known as the topological data of the group, i.e. mainly the F-symbols, which are also known as recoupling coefficients or 6j-symbols (more accurately, it\'s actually Racah\'s W-coefficients) in the case of mathsfSU_2.Below, we describe how to specify a certain type of sector what information about them needs to be implemented. Then, we describe how to build a space V composed of a direct sum of different sectors. In the last section, we explain the details of fusion trees, i.e. their construction and manipulation. But first, we provide a quick theoretical overview of the required data of the representation theory of a group."
},

{
    "location": "man/sectors/#Representation-theory-and-unitary-fusion-categories-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Representation theory and unitary fusion categories",
    "category": "section",
    "text": "Let the different irreps or sectors be labeled as a, b, c, … First and foremost, we need to specify the fusion rules a  b =  N_ab^c c with N_ab^c some non-negative integers. There should always exists a unique trivial sector u such that a  u = a = u  a. Furthermore, there should exist a unique sector overlinea such that N_aoverlinea^u = 1, whereas for all b  overlinea, N_ab^u = 0. For example, for the representations of mathsfSU_2, all irreps are self-dual (i.e. a = overlinea) and the trivial sector corresponds to spin zero.The meaning of the fusion rules is that the space of transformations R_a  R_b  R_c (or vice versa) has dimension N_ab^c. In particular, we assume the existence of a basis consisting of unitary tensor maps X_ab^cμ  R_c  R_a  R_b with μ = 1  N_ab^c such that(X_ab^cμ)^ X_ab^cμ = mathrmid_R_candsum_c sum_μ = 1^N_ab^c X_ab^cμ (X_ab^cμ)^dagger = mathrmid_R_a  R_bThe tensors X_ab^cμ are the splitting tensors, their hermitian conjugate are the fusion tensors. They are only determined up to a unitary basis transform within the space, i.e. acting on the multiplicity label μ = 1  N_ab^c. For mathsfSU_2, where N_ab^c is zero or one and the multiplicity labels are absent, the entries of X_ab^c are precisely given by the CG coefficients. The point is that we do not need to know the tensors X_ab^cμ, the topological data of (the representation category of) the group describes the following transformation:F-move or recoupling: the transformation between (a  b)  c to a  (b  c):\n(X_ab^eμ  mathrmid_c)  X_ec^dν = _fκλ F^abc_d^eμν_fκλ (X_ab^eμ  mathrmid_c) X_ec^dν (mathrmid_a  X_bc^fκ)  X_af^dλ\nBraiding or permuting as defined by σ_ab R_a  R_b  R_b  R_a:\nσ_R_aR_b  X_ab^cμ = _ν R_ab^c^μ_ν X_ba^cνThe dimensions of the spaces R_a on which representation a acts are denoted as d_a and referred to as quantum dimensions. In particular d_u = 1 and d_a = d_overlinea. This information is also encoded in the F-symbol as d_a =  F^a overlinea a_a^u_u ^-1. Note that there are no multiplicity labels in that particular F-symbol as N_aoverlinea^u = 1.If, for every a and b, there is a unique c such that a  b = c (i.e. N_ab^c = 1 and N_ab^c = 0 for all other c), the category is abelian. Indeed, the representations of a group have this property if and only if the group multiplication law is commutative. In that case, all spaces R_a associated with the representation are one-dimensional and thus trivial. In all other cases, the category is nonabelian. We find it useful to further finegrain between categories which have all N_ab^c equal to zero or one (such that no multiplicity labels are needed), e.g. the representations of mathsfSU_2, and those where some N_ab^c are larger than one, e.g. the representations of mathsfSU_3."
},

{
    "location": "man/sectors/#Sectors-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Sectors",
    "category": "section",
    "text": "We introduce a new abstract type to represent different possible sectorsabstract type Sector endAny concrete subtype of Sector should be such that its instances represent a consistent set of sectors, corresponding to the irreps of some group, or, more generally, the simple objects of a (unitary) fusion category. We refer to Appendix E of [kitaev] for a good reference.The minimal data to completely specify a type of sector arethe fusion rules, i.e. a  b =  N_ab^c c; this is implemented by a function   Nsymbol(a,b,c)\nthe list of fusion outputs from a  b; while this information is contained in   N_ab^c, it might be costly or impossible to iterate over all possible values of   c and test Nsymbol(a,b,c); instead we implement for a ⊗ b to return an iterable   object (e.g. tuple, array or a custom Julia type that listens to Base.iterate) and   which generates all c for which N_ab^c  0\nthe identity object u, such that a  u = a = u  a; this is implemented by the   function one(a) (and also in type domain) from Julia Base\nthe dual or conjugate representation overlinea for which   N_{a,\\overline{a}}^{u} = 1; this is implemented by conj(a) from Julia Base;   dual(a) also works as alias, but conj(a) is the method that should be defined\nthe F-symbol or recoupling coefficients F^abc_d^e_f, implemented as the   function Fsymbol(a,b,c,d,e,f)\nthe R-symbol R_ab^c, implemented as the function Rsymbol(a,b,c)\nfor practical reasons: a hash function hash(a, h), because sectors and objects   created from them are used as keys in lookup tables (i.e. dictionaries), and a   canonical order of sectors via isless(a,b), in order to unambiguously represent   representation spaces V = _a ℂ^n_a  R_a.Further information, such as the quantum dimensions d_a and Frobenius-Schur indicator χ_a (only if a == overlinea) are encoded in the F-symbol. They are obtained as dim(a) and frobeniusschur(a). These functions have default definitions which extract the requested data from Fsymbol(a,conj(a),a,a,one(a),one(a)), but they can be overloaded in case the value can be computed more efficiently.It is useful to distinguish between three cases with respect to the fusion rules. For irreps of Abelian groups, we have that for every a and b, there exists a unique c such that a  b = c, i.e. there is only a single fusion channel. This follows simply from the fact that all irreps are one-dimensional. All other cases are referred to as non-abelian, i.e. the irreps of a non-abelian group or some more general fusion category. We still distinguish between the case where all entries of N_ab^c  1, i.e. they are zero or one. In that case, F^abc_d^e_f and R_ab^c are scalars. If some N_ab^c  1, it means that the same sector c can appear more than once in the fusion product of a and b, and we need to introduce some multiplicity label μ for the different copies. We implement a \"trait\" (similar to IndexStyle for AbstractArrays in Julia Base), i.e. a type hierarchyabstract type FusionStyle end\nstruct Abelian <: FusionStyle\nend\nabstract type NonAbelian <: FusionStyle end\nstruct SimpleNonAbelian <: NonAbelian # non-abelian fusion but multiplicity free\nend\nstruct DegenerateNonAbelian <: NonAbelian # non-abelian fusion with multiplicities\nendNew sector types G<:Sector should then indicate which fusion style they have by defining FusionStyle(::Type{G}).In the representation and manipulation of symmetric tensors, it will be important to couple or fuse different sectors together into a single block sector. The section on Fusion trees describes the details of this process, which consists of pairwise fusing two sectors into a single coupled sector, which is then fused with the next uncoupled sector. For this, we assume the existence of a basis of unitary tensor maps X_ab^cμ  R_c  R_a  R_b such that(X_ab^cμ)^ X_ab^cμ = mathrmid_R_c and\nsum_c sum_μ = 1^N_ab^c X_ab^cμ (X_ab^cμ)^dagger = mathrmid_R_a  R_bThe tensors X_ab^cμ are the splitting tensors, their hermitian conjugate are the fusion tensors. For mathsfSU_2, their entries are precisely given by the CG coefficients. The point is that we do not need to know the tensors X_ab^cμ, the topological data of (the representation category of) the group describes the following transformation:F-move or recoupling: the transformation between (a  b)  c to a  (b  c):\n(X_ab^eμ  mathrmid_c)  X_ec^dν = _fκλ F^abc_d^eμν_fκλ (X_ab^eμ  mathrmid_c) X_ec^dν (mathrmid_a  X_bc^fκ)  X_af^dλ\nBraiding or permuting as defined by σ_ab R_a  R_b  R_b  R_a:\n``σ{a,b} ∘ X{a,b}^{c,μ} = ∑{ν} [R{a,b}^c]^μν X{b,a}^{c,ν}Furthermore, there is a relation between splitting vertices and fusion vertices given by the B-symbol, but we refer to the section on Fusion trees for the precise definition and further information. The required data is completely encoded in the the F-symbol, and corresponding Julia function Bsymbol(a,b,c) is implemented asfunction Bsymbol(a::G, b::G, c::G) where {G<:Sector}\n    if FusionStyle(G) isa Abelian || FusionStyle(G) isa SimpleNonAbelian\n        Fsymbol(a, b, dual(b), a, c, one(a))\n    else\n        reshape(Fsymbol(a,b,dual(b),a,c,one(a)), (Nsymbol(a,b,c), Nsymbol(c,dual(b),a)))\n    end\nendbut a more efficient implementation may be provided.Before discussing in more detail how a new sector type should be implemented, let us study the cases which have already been implemented. Currently, they all correspond to the irreps of groups."
},

{
    "location": "man/sectors/#Existing-group-representations-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Existing group representations",
    "category": "section",
    "text": "The first sector type is called Trivial, and corresponds to the case where there is actually no symmetry, or thus, the symmetry is the trivial group with only an identity operation and a trivial representation. Its representation theory is particularly simple:struct Trivial <: Sector\nend\nBase.one(a::Sector) = one(typeof(a))\nBase.one(::Type{Trivial}) = Trivial()\nBase.conj(::Trivial) = Trivial()\n⊗(::Trivial, ::Trivial) = (Trivial(),)\nNsymbol(::Trivial, ::Trivial, ::Trivial) = true\nFusionStyle(::Type{Trivial}) = Abelian()\nFsymbol(::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial, ::Trivial) = 1\nRsymbol(::Trivial, ::Trivial, ::Trivial) = 1The Trivial sector type is special cased in the construction of tensors, so that most of these definitions are not actually used.For all abelian groups, we gather a number of common definitionsabstract type AbelianIrrep <: Sector end\n\nBase.@pure FusionStyle(::Type{<:AbelianIrrep}) = Abelian()\nBase.@pure BraidingStyle(::Type{<:AbelianIrrep}) = Bosonic()\n\nNsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = c == first(a ⊗ b)\nFsymbol(a::G, b::G, c::G, d::G, e::G, f::G) where {G<:AbelianIrrep} =\n    Int(Nsymbol(a,b,e)*Nsymbol(e,c,d)*Nsymbol(b,c,f)*Nsymbol(a,f,d))\nfrobeniusschur(a::AbelianIrrep) = 1\nBsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = Float64(Nsymbol(a, b, c))\nRsymbol(a::G, b::G, c::G) where {G<:AbelianIrrep} = Float64(Nsymbol(a, b, c))With these common definition, we implement the representation theory of the two most common Abelian groupsstruct ZNIrrep{N} <: AbelianIrrep\n    n::Int8\n    function ZNIrrep{N}(n::Integer) where {N}\n        new{N}(mod(n, N))\n    end\nend\nBase.one(::Type{ZNIrrep{N}}) where {N} =ZNIrrep{N}(0)\nBase.conj(c::ZNIrrep{N}) where {N} = ZNIrrep{N}(-c.n)\n⊗(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = (ZNIrrep{N}(c1.n+c2.n),)\n\nstruct U1Irrep <: AbelianIrrep\n    charge::HalfInteger\nend\nBase.one(::Type{U1Irrep}) = U1Irrep(0)\nBase.conj(c::U1Irrep) = U1Irrep(-c.charge)\n⊗(c1::U1Irrep, c2::U1Irrep) = (U1Irrep(c1.charge+c2.charge),)together with some abbreviated Unicode aliasesconst ℤ₂ = ZNIrrep{2}\nconst ℤ₃ = ZNIrrep{3}\nconst ℤ₄ = ZNIrrep{4}\nconst U₁ = U1IrrepIn the definition of U1Irrep, HalfInteger<:Number is a Julia type defined in WignerSymbols.jl, which is also used for SU2Irrep below, that stores integer or half integer numbers using twice their value. Strictly speaking, the linear representations of U₁ can only have integer charges, and fractional charges lead to a projective representation. It can be useful to allow half integers in order to describe spin 1/2 systems with an axis rotation symmetry. As a user, you should not worry about the details of HalfInteger, and additional methods for automatic conversion and pretty printing are provided, as illustrated by the following exampleU₁(0.5)\nU₁(0.4)\nU₁(1) ⊗ U₁(1//2)\nu = first(U₁(1) ⊗ U₁(1//2))\nNsymbol(u, conj(u), one(u))\nz = ℤ₃(1)\nz ⊗ z\nconj(z)\none(z)For ZNIrrep{N}, we use an Int8 for compact storage, assuming that this type will not be used with N>64 (we need 2*(N-1) <= 127 in order for a ⊗ b to work correctly).As a further remark, even in the abelian case where a ⊗ b is equivalent to a single new label c, we return it as an iterable container, in this case a one-element tuple (c,).As mentioned above, we also provide the following definitionsBase.hash(c::ZNIrrep{N}, h::UInt) where {N} = hash(c.n, h)\nBase.isless(c1::ZNIrrep{N}, c2::ZNIrrep{N}) where {N} = isless(c1.n, c2.n)\nBase.hash(c::U1Irrep, h::UInt) = hash(c.charge, h)\nBase.isless(c1::U1Irrep, c2::U1Irrep) where {N} =\n    isless(abs(c1.charge), abs(c2.charge)) || zero(HalfInteger) < c1.charge == -c2.chargeSince sectors or objects made out of tuples of sectors (see the section on Fusion Trees below) are often used as keys in look-up tables (i.e. subtypes of AbstractDictionary in Julia), it is important that they can be hashed efficiently. We just hash the sectors above based on their numerical value. Note that hashes will only be used to compare sectors of the same type. The isless function provides a canonical order for sectors of a given type G<:Sector, which is useful to uniquely and unambiguously specify a representation space V = _a ℂ^n_a  R_a, as described in the section on Representation spaces below.The first example of a non-abelian representation category is that of mathsfSU_2, the implementation of which is summarized bystruct SU2Irrep <: Sector\n    j::HalfInteger\nend\nBase.one(::Type{SU2Irrep}) = SU2Irrep(zero(HalfInteger))\nBase.conj(s::SU2Irrep) = s\n⊗(s1::SU2Irrep, s2::SU2Irrep) =\n    SectorSet{SU2Irrep}(HalfInteger, abs(s1.j.num-s2.j.num):2:(s1.j.num+s2.j.num) )\ndim(s::SU2Irrep) = s.j.num+1\nBase.@pure FusionStyle(::Type{SU2Irrep}) = SimpleNonAbelian()\nNsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep) = WignerSymbols.δ(sa.j, sb.j, sc.j)\nFsymbol(s1::SU2Irrep, s2::SU2Irrep, s3::SU2Irrep,\n        s4::SU2Irrep, s5::SU2Irrep, s6::SU2Irrep) =\n    WignerSymbols.racahW(s1.j, s2.j, s4.j, s3.j, s5.j, s6.j)*sqrt(dim(s5)*dim(s6))\nfunction Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)\n    Nsymbol(sa, sb, sc) || return 0.\n    iseven(convert(Int, sa.j+sb.j-sc.j)) ? 1.0 : -1.0\nend\nBase.hash(s::SU2Irrep, h::UInt) = hash(s.j, h)\nBase.isless(s1::SU2Irrep, s2::SU2Irrep) = isless(s1.j, s2.j)\nconst SU₂ = SU2Irrepand some methods for pretty printing and converting from real numbers to irrep labels. As one can notice, the topological data (i.e. Nsymbol and Fsymbol) are provided by the package WignerSymbols.jl. The iterable a ⊗ b is a custom type, that the user does not need to care about. Some exampless = SU₂(3//2)\nconj(s)\ndim(s)\ncollect(s ⊗ s)\nfor s′ in s ⊗ s\n    @show Nsymbol(s, s, s′)\n    @show Rsymbol(s, s, s′)\nendA final non-abelian representation theory is that of the semidirect product mathsfU₁  ℤ_2, where in the context of quantum systems, this occurs in the case of systems with particle hole symmetry and the non-trivial element of ℤ_2 acts as charge conjugation C. It has the effect of interchaning mathsfU_1 irreps n and -n, and turns them together in a joint 2-dimensional index, except for the case n=0. Irreps are therefore labeled by integers n ≧ 0, however for n=0 the ℤ₂ symmetry can be realized trivially or non-trivially, resulting in an even and odd one- dimensional irrep with mathsfU)_1 charge 0. Given mathsfU_1  mathsfSO_2, this group is also simply known as mathsfO_2, and the two representations with n = 0 are the scalar and pseudo-scalar, respectively. However, because we also allow for half integer representations, we refer to it as CU₁ or CU1Irrep in full.struct CU1Irrep <: Sector\n    j::HalfInteger # value of the U1 charge\n    s::Int # rep of charge conjugation:\n    # if j == 0, s = 0 (trivial) or s = 1 (non-trivial),\n    # else s = 2 (two-dimensional representation)\n    # Let constructor take the actual half integer value j\n    function CU1Irrep(j::HalfInteger, s::Int = ifelse(j>0, 2, 0))\n        if ((j > 0 && s == 2) || (j == 0 && (s == 0 || s == 1)))\n            new(j, s)\n        else\n            error(\"Not a valid CU₁ irrep\")\n        end\n    end\nend\nBase.one(::Type{CU1Irrep}) = CU1Irrep(zero(HalfInteger), 0)\nBase.conj(c::CU1Irrep) = c\ndim(c::CU1Irrep) = ifelse(c.j == zero(HalfInteger), 1, 2)\nBase.@pure FusionStyle(::Type{CU1Irrep}) = SimpleNonAbelian()\n...\nconst CU₁ = CU1IrrepThe rest of the implementation can be read in the source code, but is rather long due to all the different cases for the arguments of Fsymbol.So far, no sectors have been implemented with FusionStyle(G) == DegenerateNonAbelian(), though an example would be the representation theory of mathsfSU_3. Such sectors are not yet fully supported; certain operations remain to be implemented."
},

{
    "location": "man/sectors/#Combining-different-sectors-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Combining different sectors",
    "category": "section",
    "text": "It is also possible to define two or more different types of symmetries, e.g. when the total symmetry group is a direct product of individual simple groups. Such sectors are obtained using the binary operator ×, which can be entered as \\times+TAB. Some examplesa = ℤ₃(1) × U₁(1)\ntypeof(a)\nconj(a)\none(a)\ndim(a)\ncollect(a ⊗ a)\nFusionStyle(a)\nb = ℤ₃(1) × SU₂(3//2)\ntypeof(b)\nconj(b)\none(b)\ndim(b)\ncollect(b ⊗ b)\nFusionStyle(c)\nc = SU₂(1) × SU₂(3//2)\ntypeof(c)\nconj(c)\none(c)\ndim(c)\ncollect(c ⊗ c)\nFusionStyle(c)We refer to the source file of ProductSector for implementation details."
},

{
    "location": "man/sectors/#Defining-a-new-type-of-sector-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Defining a new type of sector",
    "category": "section",
    "text": "By know, it should be clear how to implement a new Sector subtype. Ideally, a new G<:Sector type is a struct G ... end (immutable) that has isbitstype(G) == true (see Julia\'s manual), and implements the following minimal set of methodsBase.one(::Type{G}) = G(...)\nBase.conj(a::G) = G(...)\nTensorKit.FusionStyle(::Type{G}) = ...\n    # choose one: Abelian(), SimpleNonAbelian(), DegenerateNonAbelian()\nTensorKit.Nsymbol(a::G, b::G, c::G) = ...\n    # Bool or Integer if FusionStyle(G) == DegenerateNonAbelian()\nBase.:⊗(a::G, b::G) = ... # some iterable object that generates all possible fusion outputs\nTensorKit.Fsymbol(a::G, b::G, c::G, d::G, e::G, f::G)\nTensorKit.Rsymbol(a::G, b::G, c::G)\nBase.hash(a::G, h::UInt)\nBase.isless(a::G, b::G)Additionally, suitable definitions can be given forTensorKit.dim(a::G) = ...\nTensorKit.frobeniusschur(a::G) = ...\nTensorKit.Bsymbol(a::G, b::G, c::G) = ...If FusionStyle(G) == DegenerateNonAbelian(), then the multiple outputs c in the tensor product of a and b will be labeled as i=1, 2, …, Nsymbol(a,b,c). Optionally, a different label can be provided by definingTensorKit.vertex_ind2label(i::Int, a::G, b::G, c::G) = ...\n# some label, e.g. a `Char` or `Symbol`The following function will then automatically determine the corresponding label type (which should not vary, i.e. vertex_ind2label should be type stable)Base.@pure vertex_labeltype(G::Type{<:Sector}) =\n    typeof(vertex_ind2label(1, one(G), one(G), one(G)))The following type, which already appeared in the implementation of SU2Irrep above, can be useful for providing the return type of a ⊗ bstruct SectorSet{G<:Sector,F,S}\n    f::F\n    set::S\nend\n...\nfunction Base.iterate(s::SectorSet{G}, args...) where {G<:Sector}\n    next = iterate(s.set, args...)\n    next === nothing && return nothing\n    val, state = next\n    return convert(G, s.f(val)), state\nendThat is, SectorSet(f, set) behaves as an iterator that applies x->convert(G, f(x)) on the elements of set; if f is not provided it is just taken as the function identity."
},

{
    "location": "man/sectors/#Generalizations-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Generalizations",
    "category": "section",
    "text": "As mentioned before, the framework for sectors outlined above depends is in one-to-one correspondence to the topological data for specifying a unitary fusion category. In fact, because we also need a braiding (corresponding to Rsymbol(a,b,c)) it is a so- called ribbon fusion category. However, the category does not need to be modular.  The category of representations of a finite group[1] corresponds to a typical example (which is not modular and which have a symmetric braiding). Other examples are the representation of quasi-triangular Hopf algebras, which are typically known as anyon theories in the physics literature, e.g. Fibonicci anyons, Ising anyons, … In those cases, quantum dimensions d_a are non-integer, and there is no vector space interpretation to objects R_a (which we can identify with just a) in the decomposition V = _a ℂ^n_a  R_a. The different sectors a, … just represent abstract objects. However, there is still a vector space associated with the homomorphisms a  b  c, whose dimension is N_ab^c. The objects X_ab^cμ for μ = 1N_ab^c serve as an abstract basis for this space and from there on the discussion is completely equivalent.So far, none of these cases have been implemented, but it is a simple exercise to do so."
},

{
    "location": "man/sectors/#Representation-spaces-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Representation spaces",
    "category": "section",
    "text": "We have introduced Sector subtypes as a way to label the irreps or sectors in the decomposition V = _a ℂ^n_a  R_a. To actually represent such spaces, we now also introduce a corresponding type RepresentationSpace, which is a subtype of EuclideanSpace{ℂ}, i.e.abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} endNote that this is still an abstract type, nonetheless it will be the type name that the user calls to create specific instances."
},

{
    "location": "man/sectors/#Types-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Types",
    "category": "section",
    "text": "The actual implementation comes in two flavorsstruct GenericRepresentationSpace{G<:Sector} <: RepresentationSpace{G}\n    dims::SectorDict{G,Int}\n    dual::Bool\nend\nstruct ZNSpace{N} <: RepresentationSpace{ZNIrrep{N}}\n    dims::NTuple{N,Int}\n    dual::Bool\nendThe GenericRepresentationSpace is the default implementation and stores the different sectors a and their corresponding degeneracy n_a as key value pairs in an Associative array, i.e. a dictionary dims::SectorDict. SectorDict is a constant type alias for a specific dictionary implementation, either Julia\'s default Dict or the type SortedVectorDict implemented in TensorKit.jl. Note that only sectors a with non-zero n_a are stored. The second implementation ZNSpace{N} is a dedicated implementation for ZNIrrep{N} symmetries, and just stores all N different values n_a in a tuple.As mentioned, creating instances of these types goes via RepresentationSpace, using a list of pairs a=>n_a, i.e. V = RepresentationSpace(a=>n_a, b=>n_b, c=>n_c). In this case, the sector type G is inferred from the sectors. However, it is often more convenient to specify the sector type explicitly, since then the sectors are automatically converted to the correct type, i.e. compareRepresentationSpace{U1Irrep}(0=>3, 1=>2, -1=>1) ==\n    RepresentationSpace(U1Irrep(0)=>3, U1Irrep(1)=>2, U1Irrep(-1)=>1)or using UnicodeRepresentationSpace{U₁}(0=>3, 1=>2, -1=>1) ==\n    RepresentationSpace(U₁(0)=>3, U₁(-1)=>1, U₁(1)=>2)However, both are still to long for the most common cases. Therefore, we provide a number of type aliases, both in plain ASCII and in Unicodeconst ℤ₂Space = ZNSpace{2}\nconst ℤ₃Space = ZNSpace{3}\nconst ℤ₄Space = ZNSpace{4}\nconst U₁Space = GenericRepresentationSpace{U₁}\nconst CU₁Space = GenericRepresentationSpace{CU₁}\nconst SU₂Space = GenericRepresentationSpace{SU₂}\n\n# non-Unicode alternatives\nconst Z2Space = ℤ₂Space\nconst Z3Space = ℤ₃Space\nconst Z4Space = ℤ₄Space\nconst U1Space = U₁Space\nconst CU1Space = CU₁Space\nconst SU2Space = SU₂Space"
},

{
    "location": "man/sectors/#Methods-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Methods",
    "category": "section",
    "text": "There are a number of methods to work with instances V of RepresentationSpace. The function sectortype returns the type of the sector labels. It also works on other vector spaces, in which case it returns Trivial. The function sectors returns an iterator over the different sectors a with non-zero n_a, for other ElementarySpace types it returns (Trivial,). The degeneracy dimensions n_a can be extracted as dim(V, a), it properly returns 0 if sector a is not present in the decomposition of V. With hassector(V, a) one can check if V contains a sector a with dim(V,a)>0. Finally, dim(V) returns the total dimension of the space V, i.e. _a n_a d_a or thus dim(V) = sum(dim(V,a) * dim(a) for a in sectors(V)).Other methods for ElementarySpace, such as dual, fuse and flip also work. In fact, RepresentationSpace is the reason flip exists, cause in this case it is different then dual. The existence of flip originates from the non-trivial isomorphism between R_overlinea and R_a^*, i.e. the representation space of the dual overlinea of sector a and the dual of the representation space of sector a.In order for flip(V) to be isomorphic to V, it is such that, if V = RepresentationSpace(a=>n_a,...) then flip(V) = dual(RepresentationSpace(dual(a)=>n_a,....)). Furthermore, for two spaces V1 = RepresentationSpace(a=>n1_a, ...) and V2 = RepresentationSpace(a=>n2_a, ...), we have min(V1,V2) = RepresentationSpace(a=>min(n1_a,n2_a), ....) and similarly for max, i.e. they act on the degeneracy dimensions of every sector separately. Therefore, it can be that the return value of min(V1,V2) or max(V1,V2) is neither equal to V1 or V2.For W a ProductSpace{<:RepresentationSpace{G},N}, sectors(W) returns an iterator that generates all possible combinations of sectors as represented as NTuple{G,N}. The function dims(W, as) returns the corresponding tuple with degeneracy dimensions, while dim(W, as) returns the product of these dimensions. hassector(W, as) is equivalent to dim(W, as)>0. Finally, there is the function blocksectors(W) which returns a list (of type Vector) with all possible \"block sectors\" or total/coupled sectors that can result from fusing the individual uncoupled sectors in W. Correspondingly, blockdim(W, a) counts the total dimension of coupled sector a in W. The machinery for computing this is the topic of the next section on Fusion trees, but first, it\'s time for some examples."
},

{
    "location": "man/sectors/#Examples-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Examples",
    "category": "section",
    "text": "Let\'s start with an example involving mathsfU_1:V1 = RepresentationSpace{U₁}(0=>3, 1=>2, -1=>1)\nV1 == U1Space(0=>3, 1=>2, -1=>1) == U₁Space(0=>3, 1=>2, -1=>1)\n(sectors(V1)...,)\ndim(V1, U₁(1))\ndim(V1)\nhassector(V1, U₁(1))\nhassector(V1, U₁(2))\ndual(V1)\nflip(V1)\nV2 = U1Space(0=>2, 1=>1, -1=>1, 2=>1, -2=>1)\nmin(V1,V2)\nmax(V1,V2)\n⊕(V1,V2)\nW = ⊗(V1,V2)\n(sectors(W)...,)\ndims(W, (U₁(0), U₁(0)))\ndim(W, (U₁(0), U₁(0)))\nhassector(W, (U₁(0), U₁(0)))\nhassector(W, (U₁(2), U₁(0)))\nfuse(W)\n(blocksectors(W)...,)\nblockdim(W, U₁(0))and then with mathsfSU_2:V1 = RepresentationSpace{SU₂}(0=>3, 1//2=>2, 1=>1)\nV1 == SU2Space(0=>3, 1//2=>2, 1=>1) == SU₂Space(0=>3, 1//2=>2, 1=>1)\n(sectors(V1)...,)\ndim(V1, SU₂(1))\ndim(V1)\nhassector(V1, SU₂(1))\nhassector(V1, SU₂(2))\ndual(V1)\nflip(V1)\nV2 = SU2Space(0=>2, 1//2=>1, 1=>1, 3//2=>1, 2=>1)\nmin(V1,V2)\nmax(V1,V2)\n⊕(V1,V2)\nW = ⊗(V1,V2)\n(sectors(W)...,)\ndims(W, (SU₂(0), SU₂(0)))\ndim(W, (SU₂(0), SU₂(0)))\nhassector(W, (SU₂(0), SU₂(0)))\nhassector(W, (SU₂(2), SU₂(0)))\nfuse(W)\n(blocksectors(W)...,)\nblockdim(W, SU₂(0))"
},

{
    "location": "man/sectors/#Fusion-trees-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Fusion trees",
    "category": "section",
    "text": "Work in progressThe gain in efficiency (both in memory occupation and computation time) obtained from using symmetric tensor maps is that, by Schur\'s lemma, they are block diagonal in the basis of coupled sectors. To exploit this block diagonal form, it is however essential that we know the basis transform from the individual (uncoupled) sectors appearing in the tensor product form of the domain and codomain, to the totally coupled sectors that label the different blocks. We refer to the latter as block sectors, as we already encountered in the previous section blocksectors and blockdim defined on the type ProductSpace.To couple or fuse the different sectors together into a single block sector, we sequentially fuse together two sectors into a single coupled sector, which is then fused with the next uncoupled sector. For this, we assume the existence of unitary tensor maps X_ab^cμ  R_c  R_a  R_b introduced in the section Sectors.such that (X_ab^cμ)^ X_ab^cμ = mathrmid_R_c andsum_c sum_μ = 1^N_ab^c X_ab^cμ (X_ab^cμ)^dagger = mathrmid_R_a  R_bThe tensors X_ab^cμ are the splitting tensors, their hermitian conjugate are the fusion tensors. For mathsfSU_2, their entries are given by the Clebsch-Gordan coefficients"
},

{
    "location": "man/sectors/#Canonical-representation-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Canonical representation",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/sectors/#Possible-manipulations-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Possible manipulations",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/sectors/#Fermions-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Fermions",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/sectors/#Bibliography-1",
    "page": "Sectors, representation spaces and fusion trees",
    "title": "Bibliography",
    "category": "section",
    "text": "[kitaev]: Kitaev, A. (2006). Anyons in an exactly solved model and beyond. Annals of Physics, 321(1), 2-111.[1]: Strictly speaking the number of sectors, i.e. simple objects, in a fusion category needs to be finite, so that RepmathsfG is only a fusion category for a finite group mathsfG. It is clear our formalism also works for compact Lie groups with an infinite number of irreps, since any finite-dimensional vector space will only have a finite number of all possible irreps in its decomposition."
},

{
    "location": "man/tensors/#",
    "page": "Tensors and the TensorMap type",
    "title": "Tensors and the TensorMap type",
    "category": "page",
    "text": ""
},

{
    "location": "man/tensors/#Tensors-and-the-TensorMap-type-1",
    "page": "Tensors and the TensorMap type",
    "title": "Tensors and the TensorMap type",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/tensors/#Defining-and-constructing-tensor-maps-1",
    "page": "Tensors and the TensorMap type",
    "title": "Defining and constructing tensor maps",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/tensors/#Linear-algebra-operations-1",
    "page": "Tensors and the TensorMap type",
    "title": "Linear algebra operations",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/tensors/#Index-manipulations-1",
    "page": "Tensors and the TensorMap type",
    "title": "Index manipulations",
    "category": "section",
    "text": "TODO"
},

{
    "location": "man/tensors/#Tensor-contractions-and-tensor-networks-1",
    "page": "Tensors and the TensorMap type",
    "title": "Tensor contractions and tensor networks",
    "category": "section",
    "text": "TODO"
},

{
    "location": "lib/spaces/#",
    "page": "Vector spaces",
    "title": "Vector spaces",
    "category": "page",
    "text": ""
},

{
    "location": "lib/spaces/#TensorKit.VectorSpace",
    "page": "Vector spaces",
    "title": "TensorKit.VectorSpace",
    "category": "type",
    "text": "abstract type VectorSpace end\n\nAbstract type at the top of the type hierarchy for denoting vector spaces.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.ElementarySpace",
    "page": "Vector spaces",
    "title": "TensorKit.ElementarySpace",
    "category": "type",
    "text": "abstract type ElementarySpace{𝕜} <: VectorSpace end\n\nElementary finite-dimensional vector space over a field 𝕜 that can be used as the index space corresponding to the indices of a tensor.\n\nEvery elementary vector space should respond to the methods conj and dual, returning the complex conjugate space and the dual space respectively. The complex conjugate of the dual space is obtained as dual(conj(V)) === conj(dual(V)). These different spaces should be of the same type, so that a tensor can be defined as an element of a homogeneous tensor product of these spaces.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.GeneralSpace",
    "page": "Vector spaces",
    "title": "TensorKit.GeneralSpace",
    "category": "type",
    "text": "struct GeneralSpace{𝕜} <: ElementarySpace{𝕜}\n\nA finite-dimensional space over an arbitrary field 𝕜 without additional structure. It is thus characterized by its dimension, and whether or not it is the dual and/or conjugate space. For a real field 𝕜, the space and its conjugate are the same.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.InnerProductSpace",
    "page": "Vector spaces",
    "title": "TensorKit.InnerProductSpace",
    "category": "type",
    "text": "abstract type InnerProductSpace{𝕜} <: ElementarySpace{𝕜} end\n\nAbstract type for denoting vector with an inner product and a corresponding metric, which can be used to raise or lower indices of tensors.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.EuclideanSpace",
    "page": "Vector spaces",
    "title": "TensorKit.EuclideanSpace",
    "category": "type",
    "text": "abstract type EuclideanSpace{𝕜} <: InnerProductSpace{𝕜} end\n\nAbstract type for denoting real or complex spaces with a standard (Euclidean) inner product (i.e. orthonormal basis), such that the dual space is naturally isomorphic to the conjugate space (in the complex case) or even to the space itself (in the real case), also known as the category of finite-dimensional Hilbert spaces FdHilb.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.CartesianSpace",
    "page": "Vector spaces",
    "title": "TensorKit.CartesianSpace",
    "category": "type",
    "text": "struct CartesianSpace <: EuclideanSpace{ℝ}\n\nA real euclidean space ℝ^d, which is therefore self-dual. CartesianSpace has no additonal structure and is completely characterised by its dimension d. This is the vector space that is implicitly assumed in most of matrix algebra.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.ComplexSpace",
    "page": "Vector spaces",
    "title": "TensorKit.ComplexSpace",
    "category": "type",
    "text": "struct ComplexSpace <: EuclideanSpace{ℂ}\n\nA standard complex vector space ℂ^d with Euclidean inner product and no additional structure. It is completely characterised by its dimension and whether its the normal space or its dual (which is canonically isomorphic to the conjugate space).\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.RepresentationSpace",
    "page": "Vector spaces",
    "title": "TensorKit.RepresentationSpace",
    "category": "type",
    "text": "abstract type RepresentationSpace{G<:Sector} <: EuclideanSpace{ℂ} end\n\nComplex Euclidean space with a direct sum structure corresponding to different superselection sectors of type G<:Sector, e.g. the elements or irreps of a compact or finite group, or the labels of a unitary fusion category.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.GenericRepresentationSpace",
    "page": "Vector spaces",
    "title": "TensorKit.GenericRepresentationSpace",
    "category": "type",
    "text": "struct GenericRepresentationSpace{G<:Sector} <: RepresentationSpace{G}\n\nGeneric implementation of a representation space, i.e. a complex Euclidean space with a direct sum structure corresponding to different superselection sectors of type G<:Sector, e.g. the irreps of a compact or finite group, or the labels of a unitary fusion category.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.ZNSpace",
    "page": "Vector spaces",
    "title": "TensorKit.ZNSpace",
    "category": "type",
    "text": "struct ZNSpace{N} <: AbstractRepresentationSpace{ZNIrrep{N}}\n\nOptimized implementation of a graded ℤ_N space, i.e. a complex Euclidean space graded by the irreps of type ZNIrrep{N}.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.CompositeSpace",
    "page": "Vector spaces",
    "title": "TensorKit.CompositeSpace",
    "category": "type",
    "text": "abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end\n\nAbstract type for composite spaces that are defined in terms of a number of elementary vector spaces of a homogeneous type S<:ElementarySpace{𝕜}.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#TensorKit.ProductSpace",
    "page": "Vector spaces",
    "title": "TensorKit.ProductSpace",
    "category": "type",
    "text": "struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}\n\nA ProductSpace is a tensor product space of N vector spaces of type S<:ElementarySpace. Only tensor products between ElementarySpace objects of the same type are allowed.\n\n\n\n\n\n"
},

{
    "location": "lib/spaces/#Vector-spaces-1",
    "page": "Vector spaces",
    "title": "Vector spaces",
    "category": "section",
    "text": "CurrentModule = TensorKitThe type hierarchy for representing vector spacesVectorSpace\nElementarySpace\nGeneralSpace\nInnerProductSpace\nEuclideanSpace\nCartesianSpace\nComplexSpace\nRepresentationSpace\nGenericRepresentationSpace\nZNSpace\nCompositeSpace\nProductSpaceThe type hierarchy for representing sectorsSector\nAbelianIrrep\nZNIrrep{N}\nU1Irrep\nSU2Irrep\nCU1Irrep\nFusionStyle\nAbelian\nNonAbelian\nSimpleNonAbelian\nDegenerateNonAbelianMethods often apply similar to e.g. spaces and corresponding tensors or tensor maps, e.g.:field\nsectortype\nsectors\nhassector\ndim\ndims\nblocksectors\nblockdimThe following methods act specifically ElementarySpace spacesisdual\ndual\nconj\nflip\n:⊕\noneunitor also on ProductSpacefuse\n:⊗\none"
},

]}
