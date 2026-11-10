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
pairposition(t::FusionTreePairs, f::FusionTreePair) = pairposition(t, pairtoken(t, f))

"""
    pairtoken(t::FusionTreePairs, i::Int) -> Tuple{Int, Int}
    pairtoken(t::FusionTreePairs, (f₁, f₂)) -> Tuple{Int, Int}

The token of a fusion tree pair of `t`, from its position or from the pair itself, which must be
present.
"""
pairtoken(t::FusionTreePairs, i::Int) = _pairtoken(t.ranges₁, t.ranges₂, t.offsets, i)
function pairtoken(t::FusionTreePairs, f::FusionTreePair)
    found, token = gettoken(t, f)
    found || throw(KeyError(f))
    return token
end
function _pairtoken(ranges₁, ranges₂, offsets, i::Int)
    b = searchsortedlast(offsets, i - 1)
    r₁, r₂ = ranges₁[b], ranges₂[b]
    j₂, j₁ = divrem(i - 1 - offsets[b], length(r₁))
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
struct FusionTreePairDict{K, V, T <: FusionTreePairs, A <: AbstractVector{V}} <: AbstractDictionary{K, V}
    keys::T
    values::A
end
FusionTreePairDict(keys::T, values::AbstractVector{V}) where {T <: FusionTreePairs, V} =
    FusionTreePairDict{eltype(T), V, T, typeof(values)}(keys, values)
Base.keys(d::FusionTreePairDict) = d.keys
Dictionaries.istokenizable(::FusionTreePairDict) = true
Dictionaries.istokenassigned(::FusionTreePairDict, _) = true
Dictionaries.gettokenvalue(d::FusionTreePairDict, i::Int) = d.values[i]
Dictionaries.gettokenvalue(d::FusionTreePairDict, token::Tuple{Int, Int}) = _tokenvalue(d.values, d.keys, token)
_tokenvalue(values::AbstractVector, keys, token) = values[pairposition(keys, token)]

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
    SubblockStructure{N} <: AbstractVector{StridedStructure{N}}

The [`StridedStructure`](@ref) `(sizes, strides, offset)` of the subblock of every fusion tree pair
of a `HomSpace`, in the canonical order of [`fusiontrees`](@ref). The subblock of `(f₁, f₂)` holds
the rows of `f₁` and the columns of `f₂` within the block of their coupled sector, such that only
the structures of the splitting and fusion trees are stored, and that of a pair is computed when it
is accessed. Besides its position, a pair can also be indexed with its token `(i₁, i₂)`.
"""
struct SubblockStructure{N} <: AbstractVector{StridedStructure{N}}
    # the codomain legs of every splitting tree, padded with zeros for the domain legs
    trees₁::Vector{StridedStructure{N}}
    # the domain legs of every fusion tree, padded with zeros for the codomain legs
    trees₂::Vector{StridedStructure{N}}
    # shared with the `FusionTreePairs`, to convert between positions and tokens
    ranges₁::Vector{UnitRange{Int}}
    ranges₂::Vector{UnitRange{Int}}
    positions₁::Vector{Int}
    positions₂::Vector{Int}
    offsets::Vector{Int}
end
SubblockStructure(trees₁, trees₂, t::FusionTreePairs) =
    SubblockStructure(trees₁, trees₂, t.ranges₁, t.ranges₂, t.positions₁, t.positions₂, t.offsets)

# the token `(i₁, i₂)` of a fusion tree pair, which also addresses its subblock
const SubblockToken = Tuple{Int, Int}

Base.size(s::SubblockStructure) = (last(s.offsets),)
Base.IndexStyle(::Type{<:SubblockStructure}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(s::SubblockStructure, i::Int) =
    s[_pairtoken(s.ranges₁, s.ranges₂, s.offsets, i)]
Base.@propagate_inbounds function Base.getindex(s::SubblockStructure, (i₁, i₂)::SubblockToken)
    sz₁, str₁, offset₁ = s.trees₁[i₁]
    sz₂, str₂, offset₂ = s.trees₂[i₂]
    return (sz₁ .+ sz₂, str₁ .+ str₂, offset₁ + offset₂)
end
_tokenvalue(values::SubblockStructure, keys, token) = values[token]
pairposition(s::SubblockStructure, (i₁, i₂)::SubblockToken) = @inbounds s.positions₂[i₂] + s.positions₁[i₁]

# the arrays shared with the `FusionTreePairs` are accounted for in the `sectorstructure` cache
_cache_payload_size(s::SubblockStructure) = sizeof(s) + sizeof(s.trees₁) + sizeof(s.trees₂)

"""
    DegeneracyStructure{N}

Degeneracy-dependent structure of a `HomSpace`: the block sizes, ranges, and sub-block
strides that depend on the degeneracy (multiplicity) dimensions. Specific to a given
`HomSpace` instance.

## Fields
- `totaldim`: total number of elements in the flat data vector.
- `blockstructure`: `Vector` of `((d₁, d₂), range)` values, one per coupled sector, in the
  same order as [`sectorstructure`](@ref)`.blocksectors`.
- `subblockstructure`: [`SubblockStructure`](@ref) of the `(sizes, strides, offset)` of every
  fusion tree pair, in the same order as [`sectorstructure`](@ref)`.fusiontrees`.

See also [`degeneracystructure`](@ref), [`SectorStructure`](@ref).
"""
struct DegeneracyStructure{N}
    totaldim::Int
    blockstructure::Vector{Tuple{Tuple{Int, Int}, UnitRange{Int}}}
    subblockstructure::SubblockStructure{N}
end

MemoizationKit.cachesize(s::DegeneracyStructure) =
    sizeof(s) + sizeof(s.blockstructure) + _cache_payload_size(s.subblockstructure)

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
    codom, dom = codomain(W), domain(W)
    N₁, N₂ = length(codom), length(dom)
    zeros₁, zeros₂ = ntuple(Returns(0), N₁), ntuple(Returns(0), N₂)

    ss = sectorstructure(W)
    ts = ss.fusiontrees
    trees₁ = Vector{StridedStructure{N₁ + N₂}}(undef, length(ts.trees₁))
    trees₂ = Vector{StridedStructure{N₁ + N₂}}(undef, length(ts.trees₂))
    blockvalues = Vector{Tuple{Tuple{Int, Int}, UnitRange{Int}}}(undef, length(ss.blocksectors))

    # every block is column major, with the rows of its splitting trees and the columns of its
    # fusion trees, both in order; the legs of a tree are column major within its rows or columns
    blockoffset = 0
    for (block_index, (r₁, r₂)) in enumerate(zip(ts.ranges₁, ts.ranges₂))
        blockdim₁ = 0
        for i in r₁
            d₁s = dims(codom, gettokenvalue(ts.trees₁, i).uncoupled)
            @assert !any(==(0), d₁s)
            trees₁[i] = ((d₁s..., zeros₂...), (_colmajorstrides(d₁s, 1)..., zeros₂...), blockdim₁)
            blockdim₁ += prod(d₁s)
        end
        blockdim₂ = 0
        for i in r₂
            d₂s = dims(dom, gettokenvalue(ts.trees₂, i).uncoupled)
            @assert !any(==(0), d₂s)
            trees₂[i] = ((zeros₁..., d₂s...), (zeros₁..., _colmajorstrides(d₂s, blockdim₁)...), blockoffset + blockdim₂ * blockdim₁)
            blockdim₂ += prod(d₂s)
        end
        blockrange = (blockoffset + 1):(blockoffset + blockdim₁ * blockdim₂)
        blockvalues[block_index] = ((blockdim₁, blockdim₂), blockrange)
        blockoffset = last(blockrange)
    end

    subblockvalues = SubblockStructure(trees₁, trees₂, ts)
    return DegeneracyStructure(blockoffset, blockvalues, subblockvalues)
end

# the strides of legs with sizes `ds` that are column major, starting from `stride`
_colmajorstrides(ds::Tuple, stride::Int) = Base.front(cumprod((stride, ds...)))
