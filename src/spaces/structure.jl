# sizes, strides, offset
const StridedStructure{N} = Tuple{NTuple{N, Int}, NTuple{N, Int}, Int}

# SectorStructure: sector-dependent characterization of HomSpaces
# ---------------------------------------------------------------
"""
    SectorStructure{I <: Sector, F <: FusionTreePair}

Sector-only structure of a `HomSpace`: the coupled sectors and all valid fusion tree pairs,
depending only on which sectors appear (not their degeneracy dimensions). Shared across
`HomSpace`s with the same sector structure.

## Fields
- `blocksectors`: `Indices` of all coupled sectors `c::I`.
- `fusiontrees`: `FusionTreePairs` of all valid fusion tree pairs `(f₁, f₂)`, in canonical order.

See also [`sectorstructure`](@ref), [`DegeneracyStructure`](@ref).
"""
struct SectorStructure{I <: Sector, F <: FusionTreePair{I}, T <: AbstractIndices{F}}
    blocksectors::Indices{I}
    fusiontrees::T
end

"""
    FusionTreePairs{F₁, F₂} <: AbstractIndices{Tuple{F₁, F₂}}

The fusion tree pairs `(f₁, f₂)` of a `HomSpace`, ordered by coupled sector, then `f₂`, then `f₁`,
stored as the splitting trees `f₁` and the fusion trees `f₂` of every coupled sector. The token of
a pair is its position.
"""
struct FusionTreePairs{F₁, F₂} <: AbstractIndices{Tuple{F₁, F₂}}
    trees₁::Indices{F₁} # splitting trees, grouped by coupled sector
    trees₂::Indices{F₂} # fusion trees, grouped by coupled sector
    blocks₁::Vector{Int} # the coupled sector of every splitting tree
    blocks₂::Vector{Int} # the coupled sector of every fusion tree
    ranges₁::Vector{UnitRange{Int}} # the splitting trees of every coupled sector
    ranges₂::Vector{UnitRange{Int}} # the fusion trees of every coupled sector
    offsets::Vector{Int} # the number of pairs before every coupled sector, and in total
end

# from the trees of both sides of every coupled sector
function FusionTreePairs{F₁, F₂}(sides₁, sides₂) where {F₁, F₂}
    lengths(ss) = map(length, ss)
    ranges(ss) = map((n, l) -> (n - l + 1):n, cumsum(lengths(ss)), lengths(ss))
    blocks(ss) = Int[b for (b, ts) in enumerate(ss) for _ in ts]
    offsets = cumsum(vcat(0, lengths(sides₁) .* lengths(sides₂)))
    return FusionTreePairs{F₁, F₂}(
        Indices(F₁[f for ts in sides₁ for f in ts]), Indices(F₂[f for ts in sides₂ for f in ts]),
        blocks(sides₁), blocks(sides₂), ranges(sides₁), ranges(sides₂), offsets
    )
end

Base.length(t::FusionTreePairs) = last(t.offsets)
Dictionaries.istokenizable(::FusionTreePairs) = true
Dictionaries.tokentype(::FusionTreePairs) = Int
Dictionaries.iteratetoken(t::FusionTreePairs, i::Int = 0) = i < length(t) ? (i + 1, i + 1) : nothing
# tokens as returned by `gettoken` look like those of `Indices`, with the position second
function Dictionaries.gettoken(t::FusionTreePairs, (f₁, f₂))
    found₁, (_, i₁) = gettoken(t.trees₁, f₁)
    found₂, (_, i₂) = gettoken(t.trees₂, f₂)
    (found₁ && found₂ && t.blocks₁[i₁] == t.blocks₂[i₂]) || return false, (0, 0)
    b = t.blocks₁[i₁]
    r₁, r₂ = t.ranges₁[b], t.ranges₂[b]
    return true, (0, t.offsets[b] + (i₂ - first(r₂)) * length(r₁) + (i₁ - first(r₁)) + 1)
end
Dictionaries.gettokenvalue(t::FusionTreePairs, (_, i)::Tuple{Int, Int}) = gettokenvalue(t, i)
function Dictionaries.gettokenvalue(t::FusionTreePairs, i::Int)
    b = searchsortedlast(t.offsets, i - 1)
    r₁, r₂ = t.ranges₁[b], t.ranges₂[b]
    j₂, j₁ = divrem(i - 1 - t.offsets[b], length(r₁))
    return (gettokenvalue(t.trees₁, first(r₁) + j₁), gettokenvalue(t.trees₂, first(r₂) + j₂))
end
Base.iterate(t::FusionTreePairs) = iterate(t, (1, 0, 1))
function Base.iterate(t::FusionTreePairs, (b, i₁, i₂))
    while b <= length(t.ranges₁)
        r₁, r₂ = t.ranges₁[b], t.ranges₂[b]
        i₁ += 1
        i₁ > length(r₁) && (i₁ = 1; i₂ += 1)
        i₂ <= length(r₂) && !isempty(r₁) &&
            return (gettokenvalue(t.trees₁, first(r₁) + i₁ - 1), gettokenvalue(t.trees₂, first(r₂) + i₂ - 1)), (b, i₁, i₂)
        b, i₁, i₂ = b + 1, 0, 1
    end
    return nothing
end

# values for every fusion tree pair, sharing the tokens of `FusionTreePairs`
struct FusionTreePairDict{K, V, T <: FusionTreePairs} <: AbstractDictionary{K, V}
    keys::T
    values::Vector{V}
end
FusionTreePairDict(keys::T, values::Vector{V}) where {T <: FusionTreePairs, V} =
    FusionTreePairDict{eltype(T), V, T}(keys, values)
Base.keys(d::FusionTreePairDict) = d.keys
Dictionaries.istokenizable(::FusionTreePairDict) = true
Dictionaries.istokenassigned(::FusionTreePairDict, _) = true
Dictionaries.gettokenvalue(d::FusionTreePairDict, i::Int) = d.values[i]
Dictionaries.gettokenvalue(d::FusionTreePairDict, (_, i)::Tuple{Int, Int}) = d.values[i]

Base.@assume_effects :foldable function sectorstructuretype(key::Hashed{S}) where {S <: HomSpace}
    I = sectortype(S)
    F₁, F₂ = fusiontreetype(I, numout(S)), fusiontreetype(I, numin(S))
    return SectorStructure{I, Tuple{F₁, F₂}, FusionTreePairs{F₁, F₂}}
end

"""
    sectorstructure(W::HomSpace) -> SectorStructure

Return the [`SectorStructure`](@ref) for `W`, containing the coupled sectors as `Indices` and the
fusion tree pairs as `FusionTreePairs`. The result is cached based on the sector structure of `W` (ignoring
degeneracy dimensions).

See also [`degeneracystructure`](@ref), [`fusiontrees`](@ref), [`blocksectors`](@ref).
""" sectorstructure(::HomSpace)
sectorstructure(W::HomSpace) = sectorstructure(Hashed(W, sectorhash, sectorequal))

@cached function sectorstructure(key::Hashed{S})::sectorstructuretype(key) where {S <: HomSpace}
    W = parent(key)
    # the trees of both sides of every coupled sector; the pairs are not stored
    bs = collect(_blocksectors(W))
    F₁, F₂ = fusiontreetype(sectortype(S), numout(S)), fusiontreetype(sectortype(S), numin(S))
    sides₁ = map(c -> collect(fusiontrees(codomain(W), c)), bs)
    sides₂ = map(c -> collect(fusiontrees(domain(W), c)), bs)
    return sectorstructuretype(key)(Indices(bs), FusionTreePairs{F₁, F₂}(sides₁, sides₂))
end

CacheStyle(::typeof(sectorstructure), ::Hashed{<:HomSpace}) = GlobalLRUCache()

# DegeneracyStructure: degeneracy-dependent characterization of HomSpaces
# -----------------------------------------------------------------------
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
  values, one per fusion tree pair, in the same order as [`sectorstructure`](@ref)`.fusiontrees`.

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

"""
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
    ts = ss.fusiontrees
    structurevalues = Vector{StridedStructure{N}}(undef, length(ts))
    blockvalues = Vector{Tuple{Tuple{Int, Int}, UnitRange{Int}}}(undef, length(ss.blocksectors))

    # the subblocks of every coupled sector, from the degeneracies of the trees of both sides
    blockoffset = 0
    tree_index = 0
    for (block_index, (r₁, r₂)) in enumerate(zip(ts.ranges₁, ts.ranges₂))
        d₁ss = map(i -> dims(codom, gettokenvalue(ts.trees₁, i).uncoupled), r₁)
        d₂ss = map(i -> dims(dom, gettokenvalue(ts.trees₂, i).uncoupled), r₂)
        blockdim₁ = sum(prod, d₁ss; init = 0)
        blockdim₂ = sum(prod, d₂ss; init = 0)
        strides = (1, blockdim₁)
        offset₂ = 0
        for d₂s in d₂ss
            d₂ = prod(d₂s)
            offset₁ = 0
            for d₁s in d₁ss
                d₁ = prod(d₁s)
                totaloffset = blockoffset + offset₂ * blockdim₁ + offset₁
                subsz = (d₁s..., d₂s...)
                @assert !any(==(0), subsz)
                substr = _subblock_strides(subsz, (d₁, d₂), strides)
                structurevalues[tree_index += 1] = (subsz, substr, totaloffset)
                offset₁ += d₁
            end
            offset₂ += d₂
        end
        blockrange = (blockoffset + 1):(blockoffset + blockdim₁ * blockdim₂)
        blockvalues[block_index] = ((blockdim₁, blockdim₂), blockrange)
        blockoffset = last(blockrange)
    end

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
