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
    FusionTreePairs{I, F₁, F₂} <: AbstractIndices{Tuple{F₁, F₂}}

The fusion tree pairs `(f₁, f₂)` of a `HomSpace`, ordered by coupled sector, then `f₂`, then `f₁`,
stored as the coupled sectors and the splitting trees `f₁` and fusion trees `f₂` of every coupled
sector. The token of a pair is `(i₁, i₂)`, the positions of `f₁` and `f₂` among all splitting and
fusion trees; `pairposition` converts it into the position of the pair.
"""
struct FusionTreePairs{I, F₁, F₂} <: AbstractIndices{Tuple{F₁, F₂}}
    blocksectors::Indices{I}
    trees₁::Indices{F₁} # splitting trees, grouped by coupled sector
    trees₂::Indices{F₂} # fusion trees, grouped by coupled sector
    ranges₁::Vector{UnitRange{Int}} # the splitting trees of every coupled sector
    ranges₂::Vector{UnitRange{Int}} # the fusion trees of every coupled sector
    positions₁::Vector{Int} # the position of every splitting tree within its coupled sector
    positions₂::Vector{Int} # the number of pairs before the first pair with every fusion tree
    offsets::Vector{Int} # the number of pairs before every coupled sector, and in total
end

# from the trees of both sides of every coupled sector
function FusionTreePairs{I, F₁, F₂}(blocksectors::Indices{I}, sides₁, sides₂) where {I, F₁, F₂}
    lengths(ss) = map(length, ss)
    ranges(ss) = map((n, l) -> (n - l + 1):n, cumsum(lengths(ss)), lengths(ss))
    offsets = cumsum(vcat(0, lengths(sides₁) .* lengths(sides₂)))
    positions₁ = Int[j for ts in sides₁ for j in eachindex(ts)]
    positions₂ = Int[offsets[b] + (j - 1) * length(sides₁[b]) for (b, ts) in enumerate(sides₂) for j in eachindex(ts)]
    return FusionTreePairs{I, F₁, F₂}(
        blocksectors,
        Indices(F₁[f for ts in sides₁ for f in ts]), Indices(F₂[f for ts in sides₂ for f in ts]),
        ranges(sides₁), ranges(sides₂), positions₁, positions₂, offsets
    )
end

_cache_payload_size(t::FusionTreePairs) =
    sum(f -> (x = getfield(t, f); x isa Indices ? _cache_payload_size(x) : sizeof(x)), fieldnames(typeof(t)))

Base.length(t::FusionTreePairs) = last(t.offsets)
Dictionaries.istokenizable(::FusionTreePairs) = true
Dictionaries.tokentype(::FusionTreePairs) = Tuple{Int, Int}

"""
    pairposition(t::FusionTreePairs, token) -> Int
    pairposition(t::FusionTreePairs, (f₁, f₂)) -> Int

The position of a fusion tree pair in the canonical order of `t`, from its token or from the pair
itself, which must be present.
"""
pairposition(t::FusionTreePairs, (i₁, i₂)::Tuple{Int, Int}) = @inbounds t.positions₂[i₂] + t.positions₁[i₁]
function pairposition(t::FusionTreePairs, f::FusionTreePair)
    found, token = gettoken(t, f)
    found || throw(KeyError(f))
    return pairposition(t, token)
end
# the token of the pair at position `i`
function pairtoken(t::FusionTreePairs, i::Int)
    b = searchsortedlast(t.offsets, i - 1)
    r₁, r₂ = t.ranges₁[b], t.ranges₂[b]
    j₂, j₁ = divrem(i - 1 - t.offsets[b], length(r₁))
    return (first(r₁) + j₁, first(r₂) + j₂)
end

# iterate over the tokens, with the state `(b, i₁, i₂)`: the coupled sector and the tokens of the
# current pair, or those before the first pair (forward) or after the last pair (reverse)
function Dictionaries.iteratetoken(t::FusionTreePairs, (b, i₁, i₂) = (1, 0, 0))
    while b <= length(t.ranges₁)
        r₁, r₂ = t.ranges₁[b], t.ranges₂[b]
        if !(isempty(r₁) || isempty(r₂))
            if i₁ < first(r₁) # before the first pair of `b`
                i₁, i₂ = first(r₁), first(r₂)
                return (i₁, i₂), (b, i₁, i₂)
            elseif i₁ < last(r₁)
                return (i₁ + 1, i₂), (b, i₁ + 1, i₂)
            elseif i₂ < last(r₂)
                return (first(r₁), i₂ + 1), (b, first(r₁), i₂ + 1)
            end
        end
        b, i₁, i₂ = b + 1, 0, 0
    end
    return nothing
end
function Dictionaries.iteratetoken_reverse(t::FusionTreePairs, (b, i₁, i₂) = (length(t.ranges₁), typemax(Int), typemax(Int)))
    while b >= 1
        r₁, r₂ = t.ranges₁[b], t.ranges₂[b]
        if !(isempty(r₁) || isempty(r₂))
            if i₁ > last(r₁) # after the last pair of `b`
                i₁, i₂ = last(r₁), last(r₂)
                return (i₁, i₂), (b, i₁, i₂)
            elseif i₁ > first(r₁)
                return (i₁ - 1, i₂), (b, i₁ - 1, i₂)
            elseif i₂ > first(r₂)
                return (last(r₁), i₂ - 1), (b, last(r₁), i₂ - 1)
            end
        end
        b, i₁, i₂ = b - 1, typemax(Int), typemax(Int)
    end
    return nothing
end
function Dictionaries.gettoken(t::FusionTreePairs, (f₁, f₂))
    f₁.coupled == f₂.coupled || return false, (0, 0)
    found₁, (_, i₁) = gettoken(t.trees₁, f₁)
    found₂, (_, i₂) = gettoken(t.trees₂, f₂)
    return found₁ & found₂, (i₁, i₂)
end
Dictionaries.gettokenvalue(t::FusionTreePairs, (i₁, i₂)::Tuple{Int, Int}) =
    (gettokenvalue(t.trees₁, i₁), gettokenvalue(t.trees₂, i₂))
# the pair at position `i`
Dictionaries.gettokenvalue(t::FusionTreePairs, i::Int) = gettokenvalue(t, pairtoken(t, i))
function Base.iterate(t::FusionTreePairs, state...)
    next = Dictionaries.iteratetoken(t, state...)
    isnothing(next) && return nothing
    token, newstate = next
    return gettokenvalue(t, token), newstate
end

# values for every fusion tree pair, stored by position
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
Dictionaries.gettokenvalue(d::FusionTreePairDict, token::Tuple{Int, Int}) = d.values[pairposition(d.keys, token)]

MemoizationKit.cachesize(s::SectorStructure) =
    sizeof(s) + _cache_payload_size(s.blocksectors) + _cache_payload_size(s.fusiontrees)

Base.@assume_effects :foldable function sectorstructuretype(::S) where {S <: HomSpace}
    I = sectortype(S)
    F₁, F₂ = fusiontreetype(I, numout(S)), fusiontreetype(I, numin(S))
    return SectorStructure{I, Tuple{F₁, F₂}, FusionTreePairs{I, F₁, F₂}}
end

"""
    sectorstructure(W::HomSpace) -> SectorStructure

Return the [`SectorStructure`](@ref) for `W`, containing the coupled sectors as `Indices` and the
fusion tree pairs as `FusionTreePairs`. The result is cached based on the sector structure of `W` (ignoring
degeneracy dimensions).

See also [`degeneracystructure`](@ref), [`fusiontrees`](@ref), [`blocksectors`](@ref).
"""
@cached function sectorstructure(W::S)::sectorstructuretype(W) where {S <: HomSpace}
    # the trees of both sides of every coupled sector; the pairs are not stored
    I = sectortype(S)
    bs = Indices(vec(collect(_blocksectors(W)))) # `vec`: the units of a product sector have a shape
    F₁, F₂ = fusiontreetype(I, numout(S)), fusiontreetype(I, numin(S))
    sides₁ = map(c -> collect(fusiontrees(codomain(W), c)), bs.values)
    sides₂ = map(c -> collect(fusiontrees(domain(W), c)), bs.values)
    return sectorstructuretype(W)(bs, FusionTreePairs{I, F₁, F₂}(bs, sides₁, sides₂))
end

MemoizationKit.cachekey(::typeof(sectorstructure), W::HomSpace) = (Hashed(W, sectorhash, sectorequal),)

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

MemoizationKit.cachesize(s::DegeneracyStructure) = sizeof(s) + sizeof(s.blockstructure) + sizeof(s.subblockstructure)

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

