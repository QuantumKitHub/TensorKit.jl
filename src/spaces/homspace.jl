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

numout(W::HomSpace) = length(codomain(W))
numin(W::HomSpace) = length(domain(W))
numind(W::HomSpace) = numin(W) + numout(W)

const TensorSpace{S <: ElementarySpace} = Union{S, ProductSpace{S}}
const TensorMapSpace{S <: ElementarySpace, N₁, N₂} = HomSpace{
    S, ProductSpace{S, N₁},
    ProductSpace{S, N₂},
}

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
    # TODO: is sort! still necessary now that blocksectors of ProductSpace is sorted?
    if N₂ <= N₁
        return sort!(filter!(c -> hasblock(codom, c), blocksectors(dom)))
    else
        return sort!(filter!(c -> hasblock(dom, c), blocksectors(codom)))
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
fusiontrees(W::HomSpace) = fusionblockstructure(W).fusiontreelist

# Operations on HomSpaces
# -----------------------
"""
    permute(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂})

Return the `HomSpace` obtained by permuting the indices of the domain and codomain of `W`
according to the permutation `p₁` and `p₂` respectively.
"""
function permute(W::HomSpace{S}, (p₁, p₂)::Index2Tuple{N₁, N₂}) where {S, N₁, N₂}
    p = (p₁..., p₂...)
    TupleTools.isperm(p) && length(p) == numind(W) ||
        throw(ArgumentError("$((p₁, p₂)) is not a valid permutation for $(W)"))
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

"""
    insertleftunit(W::HomSpace, i=numind(W) + 1; conj=false, dual=false)

Insert a trivial vector space, isomorphic to the underlying field, at position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a left monoidal unit or its dual.

See also [`insertrightunit`](@ref insertrightunit(::HomSpace, ::Val{i}) where {i}),
[`removeunit`](@ref removeunit(::HomSpace, ::Val{i}) where {i}).
"""
function insertleftunit(
        W::HomSpace, ::Val{i} = Val(numind(W) + 1); conj::Bool = false, dual::Bool = false
    ) where {i}
    if i ≤ numout(W)
        return insertleftunit(codomain(W), Val(i); conj, dual) ← domain(W)
    else
        return codomain(W) ← insertleftunit(domain(W), Val(i - numout(W)); conj, dual)
    end
end

"""
    insertrightunit(W::HomSpace, i=numind(W); conj=false, dual=false)

Insert a trivial vector space, isomorphic to the underlying field, after position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a right monoidal unit or its dual.

See also [`insertleftunit`](@ref insertleftunit(::HomSpace, ::Val{i}) where {i}),
[`removeunit`](@ref removeunit(::HomSpace, ::Val{i}) where {i}).
"""
function insertrightunit(
        W::HomSpace, ::Val{i} = Val(numind(W)); conj::Bool = false, dual::Bool = false
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

struct FusionBlockStructure{I, N, F₁, F₂}
    totaldim::Int
    blockstructure::SectorDict{I, Tuple{Tuple{Int, Int}, UnitRange{Int}}}
    fusiontreelist::Vector{Tuple{F₁, F₂}}
    fusiontreestructure::Vector{StridedStructure{N}}
    fusiontreeindices::FusionTreeDict{Tuple{F₁, F₂}, Int}
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

@cached function fusionblockstructure(W::HomSpace)::fusionblockstructuretype(W)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    I = sectortype(W)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    # output structure
    blockstructure = SectorDict{I, Tuple{Tuple{Int, Int}, UnitRange{Int}}}() # size, range
    fusiontreelist = Vector{Tuple{F₁, F₂}}()
    fusiontreestructure = Vector{Tuple{NTuple{N₁ + N₂, Int}, NTuple{N₁ + N₂, Int}, Int}}() # size, strides, offset

    # temporary data structures
    splittingtrees = Vector{F₁}()
    splittingstructure = Vector{Tuple{Int, Int}}()

    # main computational routine
    blockoffset = 0
    for c in blocksectors(W)
        empty!(splittingtrees)
        empty!(splittingstructure)

        offset₁ = 0
        for f₁ in fusiontrees(codom, c)
            push!(splittingtrees, f₁)
            d₁ = dim(codom, f₁.uncoupled)
            push!(splittingstructure, (offset₁, d₁))
            offset₁ += d₁
        end
        blockdim₁ = offset₁
        strides = (1, blockdim₁)

        offset₂ = 0
        for f₂ in fusiontrees(dom, c)
            s₂ = f₂.uncoupled
            d₂ = dim(dom, s₂)
            for (f₁, (offset₁, d₁)) in zip(splittingtrees, splittingstructure)
                push!(fusiontreelist, (f₁, f₂))
                totaloffset = blockoffset + offset₂ * blockdim₁ + offset₁
                subsz = (dims(codom, f₁.uncoupled)..., dims(dom, f₂.uncoupled)...)
                @assert !any(isequal(0), subsz)
                substr = _subblock_strides(subsz, (d₁, d₂), strides)
                push!(fusiontreestructure, (subsz, substr, totaloffset))
            end
            offset₂ += d₂
        end
        blockdim₂ = offset₂
        blocksize = (blockdim₁, blockdim₂)
        blocklength = blockdim₁ * blockdim₂
        blockrange = (blockoffset + 1):(blockoffset + blocklength)
        blockoffset = last(blockrange)
        blockstructure[c] = (blocksize, blockrange)
    end

    fusiontreeindices = sizehint!(
        FusionTreeDict{Tuple{F₁, F₂}, Int}(), length(fusiontreelist)
    )
    for (i, f₁₂) in enumerate(fusiontreelist)
        fusiontreeindices[f₁₂] = i
    end
    totaldim = blockoffset
    structure = FusionBlockStructure(
        totaldim, blockstructure, fusiontreelist, fusiontreestructure, fusiontreeindices
    )
    return structure
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    strides = Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
    isnothing(strides) &&
        throw(ArgumentError("unexpected error in computing subblock strides"))
    return strides
end

function CacheStyle(::typeof(fusionblockstructure), W::HomSpace)
    return GlobalLRUCache()
end

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
