# Block and fusion tree ranges: structure information for building tensors
#--------------------------------------------------------------------------

# sizes, strides, offset
const StridedStructure{N} = Tuple{NTuple{N, Int}, NTuple{N, Int}, Int}

function sectorequal(W₁::HomSpace, W₂::HomSpace)
    check_spacetype(W₁, W₂)
    return sectorequal(codomain(W₁), codomain(W₂)) && sectorequal(domain(W₁), domain(W₂))
end
function sectorhash(W::HomSpace, h::UInt)
    h = sectorhash(codomain(W), h)
    h = sectorhash(domain(W), h)
    return h
end

"""
    SectorStructure{I, F₁, F₂}

Sector-only structure of a `HomSpace`: the coupled sectors and all valid fusion tree pairs,
depending only on which sectors appear (not their degeneracy dimensions). Shared across
`HomSpace`s with the same sector structure.

## Fields
- `blocksectors`: `Indices` of all coupled sectors `c::I`.
- `fusiontrees`: `Indices` of all valid fusion tree pairs `(f₁, f₂)`, in canonical order.

See also [`sectorstructure`](@ref), [`DegeneracyStructure`](@ref).
"""
struct SectorStructure{I, F₁, F₂}
    blocksectors::Indices{I}
    fusiontrees::Indices{Tuple{F₁, F₂}}
end

Base.@assume_effects :foldable function sectorstructuretype(key::Hashed{S}) where {S <: HomSpace}
    I = sectortype(S)
    F₁ = fusiontreetype(I, numout(S))
    F₂ = fusiontreetype(I, numin(S))
    return SectorStructure{I, F₁, F₂}
end

"""
    DegeneracyStructure{N}

Degeneracy-dependent structure of a `HomSpace`: the block sizes, ranges, and sub-block
strides that depend on the degeneracy (multiplicity) dimensions. Specific to a given
`HomSpace` instance.

## Fields
- `totaldim`: total number of elements in the flat data vector.
- `blockstructure`: `Vector` of `((d₁, d₂), range)` values, one per coupled sector, in the
  same order as [`sectorstructure`](@ref)`.blocksectors`.
- `subblockstructure`: `Vector` of [`StridedStructure`](@ref) `(sizes, strides, offset)`
  values, one per fusion tree pair, in the same order as
  [`sectorstructure`](@ref)`.fusiontrees`.

See also [`degeneracystructure`](@ref), [`SectorStructure`](@ref).
"""
struct DegeneracyStructure{N}
    totaldim::Int
    blockstructure::Vector{Tuple{Tuple{Int, Int}, UnitRange{Int}}}
    subblockstructure::Vector{StridedStructure{N}}
end

function degeneracystructuretype(W::HomSpace)
    N = length(codomain(W)) + length(domain(W))
    return DegeneracyStructure{N}
end

@doc """
    sectorstructure(W::HomSpace) -> SectorStructure

Return the [`SectorStructure`](@ref) for `W`, containing the coupled sectors and fusion tree
pairs as `Indices`. The result is cached based on the sector structure of `W` (ignoring
degeneracy dimensions).

See also [`degeneracystructure`](@ref), [`fusiontrees`](@ref), [`blocksectors`](@ref).
""" sectorstructure(::HomSpace)
sectorstructure(W::HomSpace) = sectorstructure(Hashed(W, sectorhash, sectorequal))

@cached function sectorstructure(key::Hashed{S})::sectorstructuretype(key) where {S <: HomSpace}
    W = parent(key)
    codom, dom = codomain(W), domain(W)
    I = sectortype(S)
    N₁, N₂ = numout(S), numin(S)
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    bs = Vector{I}()
    trees = Vector{Tuple{F₁, F₂}}()

    for c in _blocksectors(W)
        push!(bs, c)
        offset = length(trees)
        n₁ = 0
        for f₂ in fusiontrees(dom, c)
            if n₁ == 0
                # First f₂ for this sector: enumerate codomain trees and record how many there are.
                for f₁ in fusiontrees(codom, c)
                    push!(trees, (f₁, f₂))
                end
                n₁ = length(trees) - offset
            else
                # Subsequent f₂s: the codomain trees are already in the list at
                # offset .+ (1:n₁), so read them back instead of recomputing.
                for j in offset .+ (1:n₁)
                    push!(trees, (trees[j][1], f₂))
                end
            end
        end
    end

    return SectorStructure{I, F₁, F₂}(Indices(bs), Indices(trees))
end

CacheStyle(::typeof(sectorstructure), ::Hashed{S}) where {S <: HomSpace} = GlobalLRUCache()

@doc """
    degeneracystructure(W::HomSpace) -> DegeneracyStructure

Compute the [`DegeneracyStructure`](@ref) for `W`, describing block sizes, data ranges, and
sub-block strides. The result is cached per `HomSpace` instance (keyed by object identity,
since degeneracy dimensions affect the block sizes and offsets).

See also [`sectorstructure`](@ref), [`blockstructure`](@ref), [`subblockstructure`](@ref).
""" degeneracystructure(::HomSpace)
@cached function degeneracystructure(W::HomSpace)::degeneracystructuretype(W)
    codom = codomain(W)
    dom = domain(W)
    N = length(codom) + length(dom)

    ss = sectorstructure(W)
    treelist = ss.fusiontrees
    L = length(treelist)
    structurevalues = sizehint!(Vector{StridedStructure{N}}(), L)
    blockvalues = Vector{Tuple{Tuple{Int, Int}, UnitRange{Int}}}(undef, length(ss.blocksectors))

    # temporary data structures
    splittingstructure = Vector{NTuple{numout(W), Int}}()

    blockoffset = 0
    tree_index = 1
    block_index = 1
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
        blockvalues[block_index] = ((blockdim₁, blockdim₂), blockrange)

        # reset
        blockoffset = last(blockrange)
        tree_index += n₁ * n₂
        block_index += 1
    end
    @assert length(structurevalues) == L

    return DegeneracyStructure(blockoffset, blockvalues, structurevalues)
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    strides = Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
    isnothing(strides) &&
        throw(ArgumentError("unexpected error in computing subblock strides"))
    return strides
end

CacheStyle(::typeof(degeneracystructure), ::HomSpace) = GlobalLRUCache()

# Public API: combining the two caches
#--------------------------------------

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

blocksectors(W::HomSpace) = sectorstructure(W).blocksectors

dim(W::HomSpace) = degeneracystructure(W).totaldim

"""
    blockstructure(W::HomSpace) -> Dictionary

Return a `Dictionary` mapping each coupled sector `c::I` to a tuple `((d₁, d₂), r)`,
where `d₁` and `d₂` are the block dimensions for the codomain and domain respectively,
and `r` is the corresponding index range in the flat data vector.

See also [`degeneracystructure`](@ref), [`subblockstructure`](@ref).
"""
function blockstructure(W::HomSpace)
    return Dictionary(sectorstructure(W).blocksectors, degeneracystructure(W).blockstructure)
end

"""
    subblockstructure(W::HomSpace) -> Dictionary

Return a `Dictionary` mapping each fusion tree pair `(f₁, f₂)` to its
[`StridedStructure`](@ref) `(sizes, strides, offset)`.

See also [`degeneracystructure`](@ref), [`blockstructure`](@ref).
"""
function subblockstructure(W::HomSpace)
    return Dictionary(sectorstructure(W).fusiontrees, degeneracystructure(W).subblockstructure)
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
