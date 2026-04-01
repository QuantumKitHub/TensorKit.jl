"""
    struct HomSpace{S<:ElementarySpace, P1<:CompositeSpace{S}, P2<:CompositeSpace{S}}
    HomSpace(codomain::CompositeSpace{S}, domain::CompositeSpace{S}) where {S<:ElementarySpace}

Represents the linear space of morphisms with codomain of type `P1` and domain of type `P2`.
Note that `HomSpace` is not a subtype of [`VectorSpace`](@ref), i.e. we restrict the latter
to denote categories and their objects, and keep `HomSpace` distinct.
"""
struct HomSpace{S <: ElementarySpace, P1 <: CompositeSpace{S}, P2 <: CompositeSpace{S}}
    codomain::P1
    domain::P2
end

function HomSpace(codomain::S, domain::CompositeSpace{S}) where {S <: ElementarySpace}
    return HomSpace(⊗(codomain), domain)
end
function HomSpace(codomain::CompositeSpace{S}, domain::S) where {S <: ElementarySpace}
    return HomSpace(codomain, ⊗(domain))
end
function HomSpace(codomain::S, domain::S) where {S <: ElementarySpace}
    return HomSpace(⊗(codomain), ⊗(domain))
end
HomSpace(codomain::VectorSpace) = HomSpace(codomain, zerospace(codomain))

codomain(W::HomSpace) = W.codomain
domain(W::HomSpace) = W.domain

dual(W::HomSpace) = HomSpace(dual(W.domain), dual(W.codomain))
function Base.adjoint(W::HomSpace{S}) where {S}
    InnerProductStyle(S) === EuclideanInnerProduct() ||
        throw(ArgumentError("adjoint requires Euclidean inner product"))
    return HomSpace(W.domain, W.codomain)
end

Base.hash(W::HomSpace, h::UInt) = hash(domain(W), hash(codomain(W), h))
function Base.:(==)(W₁::HomSpace, W₂::HomSpace)
    return (W₁.codomain == W₂.codomain) && (W₁.domain == W₂.domain)
end

function sectorequal(W₁::HomSpace, W₂::HomSpace)
    return sectorequal(codomain(W₁), codomain(W₂)) && sectorequal(domain(W₁), domain(W₂))
end
function sectorhash(W::HomSpace, h::UInt)
    h = sectorhash(codomain(W), h)
    h = sectorhash(domain(W), h)
    return h
end

spacetype(::Type{<:HomSpace{S}}) where {S} = S

const TensorSpace{S <: ElementarySpace} = Union{S, ProductSpace{S}}
const TensorMapSpace{S <: ElementarySpace, N₁, N₂} = HomSpace{
    S, ProductSpace{S, N₁}, ProductSpace{S, N₂},
}

numout(::Type{TensorMapSpace{S, N₁, N₂}}) where {S, N₁, N₂} = N₁
numin(::Type{TensorMapSpace{S, N₁, N₂}}) where {S, N₁, N₂} = N₂

function Base.getindex(W::TensorMapSpace{<:IndexSpace, N₁, N₂}, i) where {N₁, N₂}
    return i <= N₁ ? codomain(W)[i] : dual(domain(W)[i - N₁])
end

function ←(codom::ProductSpace{S}, dom::ProductSpace{S}) where {S <: ElementarySpace}
    return HomSpace(codom, dom)
end
function ←(codom::S, dom::S) where {S <: ElementarySpace}
    return HomSpace(ProductSpace(codom), ProductSpace(dom))
end
←(codom::VectorSpace, dom::VectorSpace) = ←(promote(codom, dom)...)
→(dom::VectorSpace, codom::VectorSpace) = ←(codom, dom)

function Base.show(io::IO, W::HomSpace)
    return print(
        io,
        numout(W) == 1 ? codomain(W)[1] : codomain(W),
        " ← ",
        numin(W) == 1 ? domain(W)[1] : domain(W)
    )
end

"""
    blocksectors(W::HomSpace) -> Indices{I}

Return an `Indices` of all coupled sectors for `W`. The result is cached based on the
sector structure of `W` (ignoring degeneracy dimensions).

See also [`hasblock`](@ref), [`blockstructure`](@ref).
"""
blocksectors(W::HomSpace) = sectorstructure(W).blocksectors

function _blocksectors(W::HomSpace)
    sectortype(W) === Trivial &&
        return OneOrNoneIterator(dim(domain(W)) != 0 && dim(codomain(W)) != 0, Trivial())

    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    I = sectortype(W)
    if N₁ == N₂ == 0
        return allunits(I)
    elseif N₁ == 0
        return filter!(isunit, collect(blocksectors(dom))) # module space cannot end in empty space
    elseif N₂ == 0
        return filter!(isunit, collect(blocksectors(codom)))
    elseif N₂ <= N₁
        return filter!(c -> hasblock(codom, c), collect(blocksectors(dom)))
    else
        return filter!(c -> hasblock(dom, c), collect(blocksectors(codom)))
    end
end

"""
    hasblock(W::HomSpace, c::Sector)

Query whether a coupled sector `c` appears in both the codomain and domain of `W`.

See also [`blocksectors`](@ref).
"""
hasblock(W::HomSpace, c::Sector) = c in blocksectors(W)

"""
    dim(W::HomSpace) -> Int

Return the total dimension of a `HomSpace`, i.e. the number of linearly independent
morphisms that can be constructed within this space.
"""
dim(W::HomSpace) = degeneracystructure(W).totaldim

dims(W::HomSpace) = (dims(codomain(W))..., dims(domain(W))...)

"""
    blockstructure(W::HomSpace) -> Dictionary

Return a `Dictionary` mapping each coupled sector `c::I` to a tuple `((d₁, d₂), r)`,
where `d₁` and `d₂` are the block dimensions for the codomain and domain respectively,
and `r` is the corresponding index range in the flat data vector.

See also [`degeneracystructure`](@ref), [`subblockstructure`](@ref).
"""
blockstructure(W::HomSpace) = Dictionary(blocksectors(W), degeneracystructure(W).blockstructure)

"""
    fusiontrees(W::HomSpace) -> Indices{Tuple{F₁,F₂}}

Return an `Indices` of all valid fusion tree pairs `(f₁, f₂)` for `W`, providing a
bijection to sequential integer positions via `gettoken`/`gettokenvalue`. The result is
cached based on the sector structure of `W` (ignoring degeneracy dimensions), so
`HomSpace`s that share the same sectors, dualities, and index count will reuse the same
object.

See also [`sectorstructure`](@ref), [`subblockstructure`](@ref).
"""
fusiontrees(W::HomSpace) = sectorstructure(W).fusiontrees

"""
    subblockstructure(W::HomSpace) -> Dictionary

Return a `Dictionary` mapping each fusion tree pair `(f₁, f₂)` to its
[`StridedStructure`](@ref) `(sizes, strides, offset)`.

See also [`degeneracystructure`](@ref), [`blockstructure`](@ref).
"""
subblockstructure(W::HomSpace) = Dictionary(fusiontrees(W), degeneracystructure(W).subblockstructure)

"""
    fusionblocks(W::HomSpace)

Return the [`FusionTreeBlock`](@ref)s corresponding to all valid fusion channels of a given `HomSpace`,
grouped by their uncoupled charges.
"""
function fusionblocks(W::HomSpace)
    I = sectortype(W)
    N₁, N₂ = numout(W), numin(W)
    isdual_src = (map(isdual, codomain(W)), map(isdual, domain(W)))
    fblocks = Vector{FusionTreeBlock{I, N₁, N₂, fusiontreetype(I, N₁, N₂)}}()
    for dom_uncoupled_src in sectors(domain(W)), cod_uncoupled_src in sectors(codomain(W))
        fs_src = FusionTreeBlock{I}((cod_uncoupled_src, dom_uncoupled_src), isdual_src)
        isempty(fs_src) || push!(fblocks, fs_src)
    end
    return fblocks
end

function diagonalblockstructure(W::HomSpace)
    ((numin(W) == numout(W) == 1) && domain(W) == codomain(W)) ||
        throw(SpaceMismatch("Diagonal only support on V←V with a single space V"))
    structure = SectorDict{sectortype(W), UnitRange{Int}}() # range
    offset = 0
    dom = domain(W)[1]
    for c in blocksectors(W)
        d = dim(dom, c)
        structure[c] = offset .+ (1:d)
        offset += d
    end
    return structure
end

# Operations on HomSpaces
# -----------------------
"""
    permute(W::HomSpace, (p₁, p₂)::Index2Tuple)

Return the `HomSpace` obtained by permuting the indices of the domain and codomain of `W`
according to the permutation `p₁` and `p₂` respectively.
"""
function permute(W::HomSpace, (p₁, p₂)::Index2Tuple)
    p = (p₁..., p₂...)
    TupleTools.isperm(p) && length(p) == numind(W) ||
        throw(ArgumentError("$((p₁, p₂)) is not a valid permutation for $(W)"))
    return select(W, (p₁, p₂))
end

_transpose_indices(W::HomSpace) = (reverse(domainind(W)), reverse(codomainind(W)))

function LinearAlgebra.transpose(W::HomSpace, (p₁, p₂)::Index2Tuple = _transpose_indices(W))
    p = linearizepermutation(p₁, p₂, numout(W), numin(W))
    iscyclicpermutation(p) || throw(ArgumentError(lazy"$((p₁, p₂)) is not a cyclic permutation for $W"))
    return select(W, (p₁, p₂))
end

function braid(W::HomSpace, (p₁, p₂)::Index2Tuple, levels::IndexTuple)
    p = (p₁..., p₂...)
    TupleTools.isperm(p) && length(p) == numind(W) == length(levels) ||
        throw(ArgumentError("$((p₁, p₂)), $levels is not a valid braiding for $(W)"))
    return select(W, (p₁, p₂))
end

"""
    select(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂})

Return the `HomSpace` obtained by a selection from the domain and codomain of `W` according
to the indices in `p₁` and `p₂` respectively.
"""
function select(W::HomSpace{S}, (p₁, p₂)::Index2Tuple{N₁, N₂}) where {S, N₁, N₂}
    cod = ProductSpace{S, N₁}(map(n -> W[n], p₁))
    dom = ProductSpace{S, N₂}(map(n -> dual(W[n]), p₂))
    return cod ← dom
end

"""
    flip(W::HomSpace, I)

Return a new `HomSpace` object by applying `flip` to each of the spaces in the domain and
codomain of `W` for which the linear index `i` satisfies `i ∈ I`.
"""
function flip(W::HomSpace{S}, I) where {S}
    cod′ = let cod = codomain(W)
        ProductSpace{S}(ntuple(i -> i ∈ I ? flip(cod[i]) : cod[i], numout(W)))
    end
    dom′ = let dom = domain(W)
        ProductSpace{S}(ntuple(i -> (i + numout(W)) ∈ I ? flip(dom[i]) : dom[i], numin(W)))
    end
    return cod′ ← dom′
end

"""
    compose(W::HomSpace, V::HomSpace)

Obtain the HomSpace that is obtained from composing the morphisms in `W` and `V`. For this
to be possible, the domain of `W` must match the codomain of `V`.
"""
function compose(W::HomSpace{S}, V::HomSpace{S}) where {S}
    domain(W) == codomain(V) || throw(SpaceMismatch("$(domain(W)) ≠ $(codomain(V))"))
    return HomSpace(codomain(W), domain(V))
end

function TensorOperations.tensorcontract(
        A::HomSpace, pA::Index2Tuple, conjA::Bool,
        B::HomSpace, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple
    )
    return if conjA && conjB
        A′ = A'
        pA′ = adjointtensorindices(A, pA)
        B′ = B'
        pB′ = adjointtensorindices(B, pB)
        TensorOperations.tensorcontract(A′, pA′, false, B′, pB′, false, pAB)
    elseif conjA
        A′ = A'
        pA′ = adjointtensorindices(A, pA)
        TensorOperations.tensorcontract(A′, pA′, false, B, pB, false, pAB)
    elseif conjB
        B′ = B'
        pB′ = adjointtensorindices(B, pB)
        TensorOperations.tensorcontract(A, pA, false, B′, pB′, false, pAB)
    else
        return permute(compose(permute(A, pA), permute(B, pB)), pAB)
    end
end

"""
    insertleftunit(W::HomSpace, i = numind(W) + 1; conj = false, dual = false)

Insert a trivial vector space, isomorphic to the underlying field, at position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a left monoidal unit or its dual.

See also [`insertrightunit`](@ref insertrightunit(::HomSpace, ::Val{i}) where {i}),
[`removeunit`](@ref removeunit(::HomSpace, ::Val{i}) where {i}).
"""
function insertleftunit(
        W::HomSpace, ::Val{i} = Val(numind(W) + 1);
        conj::Bool = false, dual::Bool = false
    ) where {i}
    if i ≤ numout(W)
        return insertleftunit(codomain(W), Val(i); conj, dual) ← domain(W)
    else
        return codomain(W) ← insertleftunit(domain(W), Val(i - numout(W)); conj, dual)
    end
end

"""
    insertrightunit(W::HomSpace, i = numind(W); conj = false, dual = false)

Insert a trivial vector space, isomorphic to the underlying field, after position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a right monoidal unit or its dual.

See also [`insertleftunit`](@ref insertleftunit(::HomSpace, ::Val{i}) where {i}),
[`removeunit`](@ref removeunit(::HomSpace, ::Val{i}) where {i}).
"""
function insertrightunit(
        W::HomSpace, ::Val{i} = Val(numind(W));
        conj::Bool = false, dual::Bool = false
    ) where {i}
    if i ≤ numout(W)
        return insertrightunit(codomain(W), Val(i); conj, dual) ← domain(W)
    else
        return codomain(W) ← insertrightunit(domain(W), Val(i - numout(W)); conj, dual)
    end
end

"""
    removeunit(P::HomSpace, i)

This removes a trivial tensor product factor at position `1 ≤ i ≤ N`, where `i`
can be specified as an `Int` or as `Val(i)` for improved type stability.
For this to work, the space at position `i` has to be isomorphic to the field of scalars.

This operation undoes the work of [`insertleftunit`](@ref insertleftunit(::HomSpace, ::Val{i}) where {i}) 
and [`insertrightunit`](@ref insertrightunit(::HomSpace, ::Val{i}) where {i}).
"""
function removeunit(P::HomSpace, ::Val{i}) where {i}
    if i ≤ numout(P)
        return removeunit(codomain(P), Val(i)) ← domain(P)
    else
        return codomain(P) ← removeunit(domain(P), Val(i - numout(P)))
    end
end
