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
    FusionBlockStructure{I, N}

Full block structure of a `HomSpace`, encoding how a tensor's flat data vector is
partitioned into symmetry blocks and sub-blocks indexed by fusion tree pairs.

## Fields
- `totaldim`: total number of elements in the flat data vector.
- `blockstructure`: maps each coupled sector `c::I` to a tuple `((d₁, d₂), r)`, where
  `d₁` and `d₂` are the block dimensions for the codomain and domain respectively, and
  `r` is the corresponding index range in the flat data vector.
- `fusiontreestructure`: a `Vector` of [`StridedStructure`](@ref) `(sizes, strides, offset)`
  values, one per fusion tree pair, in the canonical enumeration order from
  [`fusiontrees`](@ref). Use `fusiontrees` to obtain the corresponding `Indices` of
  fusion tree pairs.

See also [`fusionblockstructure`](@ref), [`fusiontrees`](@ref).
"""
struct FusionBlockStructure{I, N}
    totaldim::Int
    blockstructure::SectorDict{I, Tuple{Tuple{Int, Int}, UnitRange{Int}}}
    fusiontreestructure::Vector{StridedStructure{N}}
end

function fusionblockstructuretype(W::HomSpace)
    N = length(codomain(W)) + length(domain(W))
    I = sectortype(W)
    return FusionBlockStructure{I, N}
end

Base.@assume_effects :foldable function fusiontreestype(key::Hashed{S}) where {S <: HomSpace}
    I = sectortype(S)
    F₁ = fusiontreetype(I, numout(S))
    F₂ = fusiontreetype(I, numin(S))
    return Indices{Tuple{F₁, F₂}}
end

"""
    fusiontrees(W::HomSpace) -> Indices{Tuple{F₁,F₂}}

Return an `Indices` of all valid fusion tree pairs `(f₁, f₂)` for `W`, providing a
bijection to sequential integer positions via `gettoken`/`gettokenvalue`. The result is
cached based on the sector structure of `W` (ignoring degeneracy dimensions), so
`HomSpace`s that share the same sectors, dualities, and index count will reuse the same
object.

See also [`fusionblockstructure`](@ref).
"""
fusiontrees(W::HomSpace) = fusiontrees(Hashed(W, sectorhash, sectorequal))

@cached function fusiontrees(key::Hashed{S})::fusiontreestype(key) where {S <: HomSpace}
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

    return Indices(trees)
end

CacheStyle(::typeof(fusiontrees), ::Hashed{S}) where {S <: HomSpace} = GlobalLRUCache()

@doc """
    fusionblockstructure(W::HomSpace) -> FusionBlockStructure

Compute the full [`FusionBlockStructure`](@ref) for `W`, describing how a tensor's flat
data vector is laid out in terms of symmetry blocks and fusion-tree sub-blocks. The result
is cached per `HomSpace` instance (keyed by object identity, not sector structure, since
degeneracy dimensions affect the block sizes and offsets).

See also [`FusionBlockStructure`](@ref), [`fusiontrees`](@ref).
""" fusionblockstructure(::HomSpace)

@cached function fusionblockstructure(W::HomSpace)::fusionblockstructuretype(W)
    codom = codomain(W)
    dom = domain(W)
    N = length(codom) + length(dom)
    I = sectortype(W)

    treelist = fusiontrees(W)
    L = length(treelist)
    structurevalues = sizehint!(Vector{StridedStructure{N}}(), L)
    blockstructure = SectorDict{I, Tuple{Tuple{Int, Int}, UnitRange{Int}}}()

    # temporary data structures
    splittingstructure = Vector{NTuple{numout(W), Int}}()

    blockoffset = 0
    tree_index = 1
    while tree_index <= L
        f₁, f₂ = gettokenvalue(treelist, tree_index)
        c = f₁.coupled

        # compute subblock structure
        # splitting tree data
        empty!(splittingstructure)
        offset₁ = 0
        for i in tree_index:L
            f₁′, f₂′ = gettokenvalue(treelist, i)
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
        for i in tree_index:n₁:L
            f₁′, f₂′ = gettokenvalue(treelist, i)
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
                push!(structurevalues, (subsz, substr, totaloffset))
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
    @assert length(structurevalues) == L

    return FusionBlockStructure(blockoffset, blockstructure, structurevalues)
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    strides = Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
    isnothing(strides) &&
        throw(ArgumentError("unexpected error in computing subblock strides"))
    return strides
end

CacheStyle(::typeof(fusionblockstructure), W::HomSpace) = GlobalLRUCache()

"""
    fusiontreestructure(W::HomSpace) -> Dictionary

Return a `Dictionary` mapping each fusion tree pair `(f₁, f₂)` to its
[`StridedStructure`](@ref) `(sizes, strides, offset)`. This wraps the cached
[`fusiontrees`](@ref) `Indices` together with the values stored in
[`fusionblockstructure`](@ref), with no data copying.
"""
function fusiontreestructure(W::HomSpace)
    return Dictionary(fusiontrees(W), fusionblockstructure(W).fusiontreestructure)
end

# Diagonal ranges
#----------------
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
