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

# truncate!
# ---------
_blocklength(d::Integer, ind) = _blocklength(Base.OneTo(d), ind)
_blocklength(ax, ind) = length(ax[ind])
_blocklength(ax::Base.OneTo, ind::AbstractVector{<:Integer}) = length(ind)
_blocklength(ax::Base.OneTo, ind::AbstractVector{Bool}) = count(ind)

function truncate_space(V::ElementarySpace, inds)
    return spacetype(V)(c => _blocklength(dim(V, c), ind) for (c, ind) in pairs(inds))
end

function truncate_domain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    for (c, b) in blocks(tdst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        b′ = block(tsrc, c)
        b .= view(b′, :, I)
    end
    return tdst
end
function truncate_codomain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    for (c, b) in blocks(tdst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        b′ = block(tsrc, c)
        b .= view(b′, I, :)
    end
    return tdst
end
function truncate_diagonal!(Ddst::DiagonalTensorMap, Dsrc::DiagonalTensorMap, inds)
    for (c, b) in blocks(Ddst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
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
    ind = SectorDict(c => (size(b, 2) + 1):size(b, 1) for (c, b) in blocks(S))
    V_truncated = truncate_space(space(S, 1), ind)
    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, ind)
    return Ũ, ind
end
function MAK.truncate(
        ::typeof(right_null!), (S, Vᴴ)::NTuple{2, AbstractTensorMap}, strategy::NoTruncation
    )
    ind = SectorDict(c => (size(b, 1) + 1):size(b, 2) for (c, b) in blocks(S))
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
    return SectorDict(c => Colon() for c in keys(values))
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
    return SectorDict(c => findall(strategy.filter, d) for (c, d) in pairs(values))
end

function MAK.findtruncated(values::SectorVector, strategy::TruncationByValue)
    atol = rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    return SectorDict(c => MAK.findtruncated(d, strategy′) for (c, d) in pairs(values))
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationByValue)
    atol = rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    return SectorDict(c => MAK.findtruncated_svd(d, strategy′) for (c, d) in pairs(values))
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
    blockstrategy(c) = truncrank(dim(strategy.space, c); strategy.by, strategy.rev)
    return SectorDict(c => MAK.findtruncated(d, blockstrategy(c)) for (c, d) in pairs(values))
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationSpace)
    blockstrategy(c) = truncrank(dim(strategy.space, c); strategy.by, strategy.rev)
    return SectorDict(c => MAK.findtruncated_svd(d, blockstrategy(c)) for (c, d) in pairs(values))
end

function MAK.findtruncated(values::SectorVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(MAK.findtruncated, values), strategy.components)
    return SectorDict(
        c => mapreduce(
                Base.Fix2(getindex, c), MatrixAlgebraKit._ind_intersect, inds
            ) for c in intersect(map(keys, inds)...)
    )
end
function MAK.findtruncated_svd(values::SectorVector, strategy::TruncationIntersection)
    inds = map(Base.Fix1(MAK.findtruncated_svd, values), strategy.components)
    return SectorDict(
        c => mapreduce(
                Base.Fix2(getindex, c), MatrixAlgebraKit._ind_intersect, inds
            ) for c in intersect(map(keys, inds)...)
    )
end

# Truncation error
# ----------------
MAK.truncation_error(values::SectorVector, ind) = MAK.truncation_error!(copy(values), ind)

function MAK.truncation_error!(values::SectorVector, ind)
    for (c, ind_c) in pairs(ind)
        v = values[c]
        v[ind_c] .= zero(eltype(v))
    end
    return norm(values)
end
