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

spacetype(::Type{<:HomSpace{S}}) where {S} = S

const TensorSpace{S <: ElementarySpace} = Union{S, ProductSpace{S}}
const TensorMapSpace{S <: ElementarySpace, N₁, N₂} = HomSpace{
    S, ProductSpace{S, N₁},
    ProductSpace{S, N₂},
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
    if length(W.codomain) == 1
        print(io, W.codomain[1])
    else
        print(io, W.codomain)
    end
    print(io, " ← ")
    return if length(W.domain) == 1
        print(io, W.domain[1])
    else
        print(io, W.domain)
    end
end

"""
    blocksectors(W::HomSpace)

Return an iterator over the different unique coupled sector labels, i.e. the intersection
of the different fusion outputs that can be obtained by fusing the sectors present in the
domain, as well as from the codomain.

See also [`hasblock`](@ref).
"""
function blocksectors(W::HomSpace)
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
hasblock(W::HomSpace, c::Sector) = hasblock(codomain(W), c) && hasblock(domain(W), c)

"""
    dim(W::HomSpace)

Return the total dimension of a `HomSpace`, i.e. the number of linearly independent
morphisms that can be constructed within this space.
"""
function dim(W::HomSpace)
    d = 0
    for c in blocksectors(W)
        d += blockdim(codomain(W), c) * blockdim(domain(W), c)
    end
    return d
end

dims(W::HomSpace) = (dims(codomain(W))..., dims(domain(W))...)

"""
    fusiontrees(W::HomSpace)

Return the fusiontrees corresponding to all valid fusion channels of a given `HomSpace`.
"""
fusiontrees(W::HomSpace) = fusiontreelist(W).fusiontreelist

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

# Block and fusion tree ranges: structure information for building tensors
#--------------------------------------------------------------------------

# sizes, strides, offset
const StridedStructure{N} = Tuple{NTuple{N, Int}, NTuple{N, Int}, Int}

function sectorequal(W₁::HomSpace, W₂::HomSpace)
    check_spacetype(W₁, W₂)
    (numout(W₁) == numout(W₂) && numin(W₁) == numin(W₂)) || return false
    for (w₁, w₂) in zip(codomain(W₁), codomain(W₂))
        isdual(w₁) == isdual(w₂) || return false
        isequal(sectors(w₁), sectors(w₂)) || return false
    end
    for (w₁, w₂) in zip(domain(W₁), domain(W₂))
        isdual(w₁) == isdual(w₂) || return false
        isequal(sectors(w₁), sectors(w₂)) || return false
    end
    return true
end
function sectorhash(W::HomSpace, h::UInt)
    for w in codomain(W)
        h = hash(sectors(w), hash(isdual(w), h))
    end
    for w in domain(W)
        h = hash(sectors(w), hash(isdual(w), h))
    end
    return h
end

"""
    FusionTreeList{F₁, F₂}

Charge-only structure encoding a bijection between the fusion tree pairs and a linear index.
This encodes the symmetry structure of a `HomSpace`, shared across all `HomSpace`s with the same `sectors` but varying degeneracies.
"""
struct FusionTreeList{F₁, F₂}
    fusiontreelist::Vector{Tuple{F₁, F₂}}
    fusiontreeindices::FusionTreeDict{Tuple{F₁, F₂}, Int}
end

struct FusionBlockStructure{I, N, F₁, F₂}
    totaldim::Int
    blockstructure::SectorDict{I, Tuple{Tuple{Int, Int}, UnitRange{Int}}}
    fusiontreestructure::Vector{StridedStructure{N}}
    treelist::FusionTreeList{F₁, F₂}
end

function fusionblockstructuretype(W::HomSpace)
    N₁ = length(codomain(W))
    N₂ = length(domain(W))
    N = N₁ + N₂
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    return FusionBlockStructure{I, N, F₁, F₂}
end

Base.@assume_effects :foldable function fusiontreelisttype(key::Hashed{S}) where {S <: HomSpace}
    I = sectortype(S)
    F₁ = fusiontreetype(I, numout(S))
    F₂ = fusiontreetype(I, numin(S))
    return FusionTreeList{F₁, F₂}
end

fusiontreelist(W::HomSpace) = fusiontreelist(Hashed(W, sectorhash, sectorequal))

@cached function fusiontreelist(key::Hashed{S})::fusiontreelisttype(key) where {S <: HomSpace}
    W = parent(key)
    codom, dom = codomain(W), domain(W)
    I = sectortype(S)
    N₁, N₂ = numout(S), numin(S)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    trees = Vector{Tuple{F₁, F₂}}()

    for c in blocksectors(W)
        codom_start = length(trees) + 1
        n₁ = 0
        for f₂ in fusiontrees(dom, c)
            if n₁ == 0
                # First f₂ for this sector: enumerate codomain trees and record how many there are.
                for f₁ in fusiontrees(codom, c)
                    push!(trees, (f₁, f₂))
                end
                n₁ = length(trees) - codom_start + 1
            else
                # Subsequent f₂s: the codomain trees are already in the list at
                # codom_start:codom_start+n₁-1, so read them back instead of recomputing.
                for j in codom_start:(codom_start + n₁ - 1)
                    push!(trees, (trees[j][1], f₂))
                end
            end
        end
    end

    treeindices = sizehint!(
        FusionTreeDict{Tuple{F₁, F₂}, Int}(), length(trees)
    )
    for (i, f₁₂) in enumerate(trees)
        treeindices[f₁₂] = i
    end

    return FusionTreeList{F₁, F₂}(trees, treeindices)
end

CacheStyle(::typeof(fusiontreelist), ::Hashed{S}) where {S <: HomSpace} = GlobalLRUCache()

@cached function fusionblockstructure(W::HomSpace)::fusionblockstructuretype(W)
    codom = codomain(W)
    dom = domain(W)
    N = length(codom) + length(dom)
    I = sectortype(W)

    treelist = fusiontreelist(W)
    trees = treelist.fusiontreelist
    L = length(trees)
    fusiontreestructure = sizehint!(Vector{StridedStructure{N}}(), L)
    blockstructure = SectorDict{I, Tuple{Tuple{Int, Int}, UnitRange{Int}}}()


    # temporary data structures
    splittingstructure = Vector{NTuple{numout(W), Int}}()

    blockoffset = 0
    tree_index = 1
    while tree_index <= L
        f₁, f₂ = trees[tree_index]
        c = f₁.coupled

        # compute subblock structure
        # splitting tree data
        empty!(splittingstructure)
        offset₁ = 0
        for (f₁′, f₂′) in view(trees, tree_index:L)
            f₂′ == f₂ || break
            s₁ = f₁′.uncoupled
            d₁s = dims(codom, s₁)
            d₁ = prod(d₁s)
            offset₁ += d₁
            push!(splittingstructure, d₁s)
        end
        blockdim₁ = offset₁
        n₁ = length(splittingstructure)
        strides = (1, blockdim₁)

        # fusion tree data and combine
        offset₂ = 0
        n₂ = 0
        for (f₁′, f₂′) in view(trees, tree_index:n₁:L)
            f₂′.coupled == c || break
            n₂ += 1
            s₂ = f₂′.uncoupled
            d₂s = dims(dom, s₂)
            d₂ = prod(d₂s)
            offset₁ = 0
            for d₁s in splittingstructure
                d₁ = prod(d₁s)
                totaloffset = blockoffset + offset₂ * blockdim₁ + offset₁
                subsz = (d₁s..., d₂s...)
                @assert !any(==(0), subsz)
                substr = _subblock_strides(subsz, (d₁, d₂), strides)
                push!(fusiontreestructure, (subsz, substr, totaloffset))
                offset₁ += d₁
            end
            offset₂ += d₂
        end

        # compute block structure
        blockdim₂ = offset₂
        blockrange = (blockoffset + 1):(blockoffset + blockdim₁ * blockdim₂)
        blockstructure[c] = ((blockdim₁, blockdim₂), blockrange)

        # reset
        blockoffset = last(blockrange)
        tree_index += n₁ * n₂
    end
    @assert length(fusiontreestructure) == L

    return FusionBlockStructure(blockoffset, blockstructure, fusiontreestructure, treelist)
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    strides = Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
    isnothing(strides) &&
        throw(ArgumentError("unexpected error in computing subblock strides"))
    return strides
end

CacheStyle(::typeof(fusionblockstructure), W::HomSpace) = GlobalLRUCache()

# Diagonal ranges
#----------------
# TODO: is this something we want to cache?
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
