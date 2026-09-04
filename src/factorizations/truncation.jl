# Strategies
# ----------
"""
    TruncationSpace(V::ElementarySpace, by::Function, rev::Bool)

Truncation strategy to keep the first values for each sector when sorted according to `by` and `rev`,
such that the resulting vector space is no greater than `V`.

See also [`truncspace`](@ref).
"""
struct TruncationSpace{S <: ElementarySpace, F} <: TruncationStrategy
    space::S
    by::F
    rev::Bool
end

"""
    truncspace(space::ElementarySpace; by=abs, rev::Bool=true)

Truncation strategy to keep the first values for each sector when sorted according to `by` and `rev`,
such that the resulting vector space is no greater than `V`.
"""
function truncspace(space::ElementarySpace; by = abs, rev::Bool = true)
    isdual(space) && throw(ArgumentError("truncation space should not be dual"))
    return TruncationSpace(space, by, rev)
end

TensorKit.spacetype(::Type{<:TruncationSpace{S}}) where {S} = S

# truncate!
# ---------
_blocklength(d::Integer, ind) = _blocklength(Base.OneTo(d), ind)
_blocklength(ax, ind) = length(ax[ind])
_blocklength(ax::Base.OneTo, ind::AbstractVector{<:Integer}) = length(ind)
_blocklength(ax::Base.OneTo, ind::AbstractVector{Bool}) = count(ind)

# TODO: it quacks like a duck, just define a subtype of AbstractDict?
# represent the sector-index mapping as Vector{Union{Nothing, V}} where V is the type of the index
# mapping is indexed through findindex
# the type V is needed because the concrete type of ind depends on the strategy (except for intersect/union)
_densenew(::Type{I}, ::Type{V}) where {I <: Sector, V} =
    Vector{Union{Nothing, V}}(nothing, length(values(I)))

function _denseset!(v::Vector, ::Type{I}, c::I, val) where {I <: Sector}
    v[findindex(values(I), c)] = val
    return v
end
_denseget(v::Vector, ::Type{I}, c::I) where {I <: Sector} = v[findindex(values(I), c)]
function _densepairs(v::Vector, ::Type{I}) where {I <: Sector}
    vals = values(I)
    return (vals[i] => x for (i, x) in enumerate(v) if !isnothing(x))
end
_densekeys(v::Vector, ::Type{I}) where {I <: Sector} = (c for (c, _) in _densepairs(v, I))

# fallbacks to catch SectorVector/SectorDict, even for NTuple sectorstoragetype
_denseget(v, ::Type{I}, c::I) where {I <: Sector} = get(v, c, nothing)
_densekeys(v, ::Type{I}) where {I <: Sector} = keys(v)
_densepairs(v, ::Type{I}) where {I <: Sector} = pairs(v)

# builds either a dense Vector or SectorDict based on sectorstoragetype
# mapping each (c, v) pair's sector c to f(c, v)
# so every `findtruncated` method shares one output-construction path
# `pairsiter` are c => v pairs, can be c => nothing for NoTruncation/TruncationIntersection/TruncationUnion
function _builddensemap(f, ::Type{D}, ::Type{I}, pairsiter, ::Type{V}) where {D <: Tuple, I <: Sector, V}
    d = _densenew(I, V)
    for (c, v) in pairsiter
        _denseset!(d, I, c, f(c, v))
    end
    return d
end
function _builddensemap(f, ::Type{D}, ::Type{I}, pairsiter, ::Type{V}) where {D <: SectorDict, I <: Sector, V}
    return SectorDict(c => f(c, v) for (c, v) in pairsiter) # V unused
end

function truncate_space(V::ElementarySpace, inds)
    @assert !isdual(V)
    I = sectortype(V)
    @assert I == Trivial
    return spacetype(V)(c => _blocklength(dim(V, c), ind) for (c, ind) in _densepairs(inds, I))
end
function truncate_space(V::GradedSpace{I, NTuple{N, Int}}, inds) where {I <: Sector, N}
    @assert !isdual(V)
    vals = values(I)
    newdims = zeros(Int, N)
    for (c, ind) in _densepairs(inds, I)
        d = dim(V, c)
        n_write = findindex(vals, c)
        newdims[n_write] = _blocklength(d, ind)
    end
    return typeof(V)(NTuple{N, Int}(newdims), false)
end
function truncate_space(V::GradedSpace{I, <:SectorDict}, inds) where {I <: Sector}
    @assert !isdual(V)
    ks, vs = Vector{I}(), Vector{Int}() # accumulate and sort once at the end
    for (c, ind) in pairs(inds)
        d = dim(V, c)
        len = _blocklength(d, ind)
        if !iszero(len)
            push!(ks, c)
            push!(vs, len)
        end
    end
    perm = sortperm(ks)
    return typeof(V)(SectorDict{I, Int}(ks[perm], vs[perm]), false)
end

function truncate_domain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    Isec = sectortype(tdst)
    for (c, b) in blocks(tdst)
        I = _denseget(inds, Isec, c)
        @assert !isnothing(I) # kept for safety, but should be guaranteed by _densepairs
        b′ = block(tsrc, c)
        b .= view(b′, :, I)
    end
    return tdst
end
function truncate_codomain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    Isec = sectortype(tdst)
    for (c, b) in blocks(tdst)
        I = _denseget(inds, Isec, c)
        @assert !isnothing(I) # kept for safety, but should be guaranteed by _densepairs
        b′ = block(tsrc, c)
        b .= view(b′, I, :)
    end
    return tdst
end
function truncate_diagonal!(Ddst::DiagonalTensorMap, Dsrc::DiagonalTensorMap, inds)
    Isec = sectortype(Ddst)
    for (c, b) in blocks(Ddst)
        I = _denseget(inds, Isec, c)
        @assert !isnothing(I) # kept for safety, but should be guaranteed by _densepairs
        diagview(b) .= view(diagview(block(Dsrc, c)), I)
    end
    return Ddst
end

function MAK.truncate(
        ::typeof(svd_trunc!), (U, S, Vᴴ)::NTuple{3, AbstractTensorMap},
        strategy::TruncationStrategy
    )
    ind = MAK.findtruncated_svd(diagview(S), strategy)
    V_truncated = truncate_space(space(S, 1), ind)

    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, ind)
    S̃ = similar_diagonal(S, V_truncated)
    truncate_diagonal!(S̃, S, ind)
    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    truncate_codomain!(Ṽᴴ, Vᴴ, ind)

    return (Ũ, S̃, Ṽᴴ), ind
end

function MAK.truncate(
        ::typeof(left_null!), (U, S)::NTuple{2, AbstractTensorMap}, strategy::TruncationStrategy
    )
    extended_S = zerovector!(SectorVector{eltype(S), sectortype(S), storagetype(S)}(undef, fuse(codomain(U))))
    for (c, b) in blocks(S)
        copyto!(extended_S[c], diagview(b)) # copyto! since `b` might be shorter
    end
    ind = MAK.findtruncated(extended_S, strategy)
    V_truncated = truncate_space(space(S, 1), ind)
    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, ind)
    return Ũ, ind
end
function MAK.truncate(
        ::typeof(right_null!), (S, Vᴴ)::NTuple{2, AbstractTensorMap}, strategy::TruncationStrategy
    )
    extended_S = zerovector!(SectorVector{eltype(S), sectortype(S), storagetype(S)}(undef, fuse(domain(Vᴴ))))
    for (c, b) in blocks(S)
        copyto!(extended_S[c], diagview(b)) # copyto! since `b` might be shorter
    end
    ind = MAK.findtruncated(extended_S, strategy)
    V_truncated = truncate_space(dual(space(S, 2)), ind)
    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    truncate_codomain!(Ṽᴴ, Vᴴ, ind)
    return Ṽᴴ, ind
end

# special case `NoTruncation` for null: should keep exact zeros due to rectangularity
# need to specialize to avoid ambiguity with special case in MatrixAlgebraKit
function MAK.truncate(
        ::typeof(left_null!), (U, S)::NTuple{2, AbstractTensorMap}, strategy::NoTruncation
    )
    I = sectortype(S)
    ind = _builddensemap(sectorstoragetype(I), I, blocks(S), UnitRange{Int}) do _, b
        (size(b, 2) + 1):size(b, 1)
    end
    V_truncated = truncate_space(space(S, 1), ind)
    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, ind)
    return Ũ, ind
end
function MAK.truncate(
        ::typeof(right_null!), (S, Vᴴ)::NTuple{2, AbstractTensorMap}, strategy::NoTruncation
    )
    I = sectortype(S)
    ind = _builddensemap(sectorstoragetype(I), I, blocks(S), UnitRange{Int}) do _, b
        (size(b, 1) + 1):size(b, 2)
    end
    V_truncated = truncate_space(dual(space(S, 2)), ind)
    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    truncate_codomain!(Ṽᴴ, Vᴴ, ind)
    return Ṽᴴ, ind
end

for f! in (:eig_trunc!, :eigh_trunc!)
    @eval function MAK.truncate(
            ::typeof($f!),
            (D, V)::Tuple{DiagonalTensorMap, AbstractTensorMap},
            strategy::TruncationStrategy
        )
        ind = MAK.findtruncated(diagview(D), strategy)
        V_truncated = truncate_space(space(D, 1), ind)

        D̃ = similar_diagonal(D, V_truncated)
        truncate_diagonal!(D̃, D, ind)

        Ṽ = similar(V, codomain(V) ← V_truncated)
        truncate_domain!(Ṽ, V, ind)

        return (D̃, Ṽ), ind
    end
end

# findtruncated
# -------------
# auxiliary functions
rtol_to_atol(S, p, atol, rtol) = rtol == 0 ? atol : max(atol, norm(S, p) * rtol)

# Generic fallback
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationStrategy)
    return MAK.findtruncated(values, strategy)
end

function MAK.findtruncated(values::SectorVector, ::NoTruncation)
    I = sectortype(values)
    return _builddensemap(sectorstoragetype(I), I, (c => nothing for c in keys(values)), Colon) do _, _
        Colon()
    end
end

# Need to select the first k values here after sorting across blocks, weighted by quantum dimension
# The strategy is therefore to sort all values, and then use a logical array to indicate
# which ones to keep.
# For GenericFusion, we additionally keep a vector of the quantum dimensions to provide the
# correct weight
function MAK.findtruncated(values::SectorVector, strategy::TruncationByOrder)
    I = sectortype(values)

    # dimensions are all 1 so no need to account for weight
    if FusionStyle(I) isa UniqueFusion
        howmany = min(length(parent(values)), strategy.howmany)
        perm = partialsortperm(parent(values), 1:howmany; strategy.by, strategy.rev)
        result = similar(values, Bool)
        fill!(parent(result), false)
        parent(result)[perm] .= true
        return result
    end

    # allocate vector of weights for each value
    dims = similar(values, Base.promote_op(dim, I))
    for (c, v) in pairs(dims)
        fill!(v, dim(c))
    end

    # allocate logical array for the output
    result = similar(values, Bool)
    fill!(parent(result), false)

    # loop over sorted values and mark as to keep until dimension is reached
    totaldim = 0
    for i in sortperm(parent(values); strategy.by, strategy.rev)
        totaldim += dims[i]
        totaldim > strategy.howmany && break
        result[i] = true
    end

    return result
end
# disambiguate
MAK.findtruncated_svd(values::SectorVector, strategy::TruncationByOrder) =
    MAK.findtruncated(values, strategy)

function MAK.findtruncated(values::SectorVector, strategy::TruncationByFilter)
    I = sectortype(values)
    return _builddensemap(sectorstoragetype(I), I, pairs(values), Vector{Int}) do _, v
        findall(strategy.filter, v)
    end
end

function MAK.findtruncated(values::SectorVector, strategy::TruncationByValue)
    I = sectortype(values)
    atol = rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    V = Base.promote_op(MAK.findtruncated, valtype(values), typeof(strategy′))
    return _builddensemap(sectorstoragetype(I), I, pairs(values), V) do _, v
        MAK.findtruncated(v, strategy′)
    end
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationByValue)
    I = sectortype(values)
    atol = rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    V = Base.promote_op(MAK.findtruncated_svd, valtype(values), typeof(strategy′))
    return _builddensemap(sectorstoragetype(I), I, pairs(values), V) do _, v
        MAK.findtruncated_svd(v, strategy′)
    end
end

# Need to select the first k values here after sorting by error across blocks,
# where k is determined by the cumulative truncation error of these values.
# The strategy is therefore to sort all values, and then use a logical array to indicate
# which ones to keep.
function MAK.findtruncated(values::SectorVector, strategy::TruncationByError)
    (isfinite(strategy.p) && strategy.p > 0) ||
        throw(ArgumentError(lazy"p-norm with p = $(strategy.p) is currently not supported."))
    ϵᵖmax = max(strategy.atol^strategy.p, strategy.rtol^strategy.p * norm(values, strategy.p))
    ϵᵖ = similar(values, typeof(ϵᵖmax))

    # dimensions are all 1 so no need to account for weight
    if FusionStyle(sectortype(values)) isa UniqueFusion
        parent(ϵᵖ) .= abs.(parent(values)) .^ strategy.p
    else
        for (c, v) in pairs(values)
            v′ = ϵᵖ[c]
            v′ .= abs.(v) .^ strategy.p .* dim(c)
        end
    end

    # allocate logical array for the output
    result = similar(values, Bool)
    fill!(parent(result), true)

    # loop over sorted values and mark as to discard until maximal error is reached
    totalerr = zero(eltype(ϵᵖ))
    for i in sortperm(parent(values); by = abs, rev = false)
        totalerr += ϵᵖ[i]
        totalerr > ϵᵖmax && break
        result[i] = false
    end

    return result
end
# disambiguate
MAK.findtruncated_svd(values::SectorVector, strategy::TruncationByError) =
    MAK.findtruncated(values, strategy)

function MAK.findtruncated(values::SectorVector, strategy::TruncationSpace)
    I = sectortype(values)
    I == sectortype(strategy) || throw(SectorMismatch("sectortype of truncation strategy does not match values"))
    blockstrategy(c) = truncrank(dim(strategy.space, c); strategy.by, strategy.rev)
    Vstrategy = Base.promote_op(blockstrategy, I)
    V = Base.promote_op(MAK.findtruncated, valtype(values), Vstrategy)
    return _builddensemap(sectorstoragetype(I), I, pairs(values), V) do c, v
        MAK.findtruncated(v, blockstrategy(c))
    end
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationSpace)
    I = sectortype(values)
    I == sectortype(strategy) || throw(SectorMismatch("sectortype of truncation strategy does not match values"))
    blockstrategy(c) = truncrank(dim(strategy.space, c); strategy.by, strategy.rev)
    Vstrategy = Base.promote_op(blockstrategy, I)
    V = Base.promote_op(MAK.findtruncated_svd, valtype(values), Vstrategy)
    return _builddensemap(sectorstoragetype(I), I, pairs(values), V) do c, v
        MAK.findtruncated_svd(v, blockstrategy(c))
    end
end

# The implementations below assume that the `SectorDict` always contains an entry for every block sector
# for example, if a block gets fully truncated, inds[c] = Int[].
# This is always the case in the implementations above.

function MAK.findtruncated(values::SectorVector, strategy::TruncationIntersection)
    I = sectortype(values)
    inds = map(Base.Fix1(MAK.findtruncated, values), strategy.components)
    @assert TensorKit._allequal(v -> collect(_densekeys(v, I)), inds) "missing blocks are not supported right now"
    sectors = collect(_densekeys(first(inds), I))
    return _builddensemap(sectorstoragetype(I), I, (c => nothing for c in sectors), Any) do c, _
        mapreduce(v -> _denseget(v, I, c), MatrixAlgebraKit._ind_intersect, inds)
    end
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationIntersection)
    I = sectortype(values)
    inds = map(Base.Fix1(MAK.findtruncated_svd, values), strategy.components)
    @assert TensorKit._allequal(v -> collect(_densekeys(v, I)), inds) "missing blocks are not supported right now"
    sectors = collect(_densekeys(first(inds), I))
    return _builddensemap(sectorstoragetype(I), I, (c => nothing for c in sectors), Any) do c, _
        mapreduce(v -> _denseget(v, I, c), MatrixAlgebraKit._ind_intersect, inds)
    end
end
function MAK.findtruncated(values::SectorVector, strategy::TruncationUnion)
    I = sectortype(values)
    inds = map(Base.Fix1(MAK.findtruncated, values), strategy.components)
    @assert TensorKit._allequal(v -> collect(_densekeys(v, I)), inds) "missing blocks are not supported right now"
    sectors = collect(_densekeys(first(inds), I))
    return _builddensemap(sectorstoragetype(I), I, (c => nothing for c in sectors), Any) do c, _
        mapreduce(v -> _denseget(v, I, c), MatrixAlgebraKit._ind_union, inds)
    end
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationUnion)
    I = sectortype(values)
    inds = map(Base.Fix1(MAK.findtruncated_svd, values), strategy.components)
    @assert TensorKit._allequal(v -> collect(_densekeys(v, I)), inds) "missing blocks are not supported right now"
    sectors = collect(_densekeys(first(inds), I))
    return _builddensemap(sectorstoragetype(I), I, (c => nothing for c in sectors), Any) do c, _
        mapreduce(v -> _denseget(v, I, c), MatrixAlgebraKit._ind_union, inds)
    end
end

# Truncation error
# ----------------
MAK.truncation_error(values::SectorVector, ind) = MAK.truncation_error!(copy(values), ind)

function MAK.truncation_error!(values::SectorVector, ind)
    Isec = sectortype(values)
    for (c, ind_c) in _densepairs(ind, Isec)
        v = values[c]
        v[ind_c] .= zero(eltype(v))
    end
    return norm(values)
end
