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

    treeindices = sizehint!(FusionTreeDict{Tuple{F₁, F₂}, Int}(), length(trees))
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
