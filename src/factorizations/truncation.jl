# Strategies
# ----------
"""
    TruncationSpace(V::ElementarySpace, by::Function, rev::Bool)

Truncation strategy to keep the first values for each sector when sorted according to `by` and `rev`,
such that the resulting vector space is no greater than `V`.

See also [`truncspace`](@ref).
"""
struct TruncationSpace{S<:ElementarySpace,F} <: TruncationStrategy
    space::S
    by::F
    rev::Bool
end

"""
    truncspace(space::ElementarySpace; by=abs, rev::Bool=true)

Truncation strategy to keep the first values for each sector when sorted according to `by` and `rev`,
such that the resulting vector space is no greater than `V`.
"""
function truncspace(space::ElementarySpace; by=abs, rev::Bool=true)
    isdual(space) && throw(ArgumentError("resulting vector space is never dual"))
    return TruncationSpace(space, by, rev)
end

# truncate!
# ---------
_blocklength(d::Integer, ind) = _blocklength(Base.OneTo(d), ind)
_blocklength(ax, ind) = length(ax[ind])
function truncate_space(V::ElementarySpace, inds)
    return spacetype(V)(c => _blocklength(dim(V, c), ind) for (c, ind) in inds)
end

function truncate_domain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    for (c, b) in blocks(tdst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        copy!(b, @view(block(tsrc, c)[:, I]))
    end
    return tdst
end
function truncate_codomain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    for (c, b) in blocks(tdst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        copy!(b, @view(block(tsrc, c)[I, :]))
    end
    return tdst
end
function truncate_diagonal!(Ddst::DiagonalTensorMap, Dsrc::DiagonalTensorMap, inds)
    for (c, b) in blocks(Ddst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        copy!(diagview(b), @view(diagview(block(Dsrc, c))[I]))
    end
    return Ddst
end

function MAK.truncate(::typeof(svd_trunc!), (U, S, Vᴴ)::NTuple{3,AbstractTensorMap},
                      strategy::TruncationStrategy)
    ind = MAK.findtruncated_svd(diagview(S), strategy)
    V_truncated = truncate_space(space(S, 1), ind)

    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, ind)
    S̃ = DiagonalTensorMap{scalartype(S)}(undef, V_truncated)
    truncate_diagonal!(S̃, S, ind)
    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    truncate_codomain!(Ṽᴴ, Vᴴ, ind)

    return (Ũ, S̃, Ṽᴴ), ind
end

function MAK.truncate(::typeof(left_null!),
                      (U, S)::Tuple{AbstractTensorMap,AbstractTensorMap},
                      strategy::MatrixAlgebraKit.TruncationStrategy)
    extended_S = SectorDict(c => vcat(diagview(b),
                                      zeros(eltype(b), max(0, size(b, 2) - size(b, 1))))
                            for (c, b) in blocks(S))
    ind = MAK.findtruncated(extended_S, strategy)
    V_truncated = truncate_space(space(S, 1), ind)
    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, ind)
    return Ũ, ind
end

for f! in (:eig_trunc!, :eigh_trunc!)
    @eval function MAK.truncate(::typeof($f!),
                                (D, V)::Tuple{DiagonalTensorMap,AbstractTensorMap},
                                strategy::TruncationStrategy)
        ind = MAK.findtruncated(diagview(D), strategy)
        V_truncated = spacetype(D)(c => length(I) for (c, I) in ind)

        D̃ = DiagonalTensorMap{scalartype(D)}(undef, V_truncated)
        truncate_diagonal!(D̃, D, ind)

        Ṽ = similar(V, codomain(V) ← V_truncated)
        truncate_domain!(Ṽ, V, ind)

        return (D̃, Ṽ), ind
    end
end

# Find truncation
# ---------------
# auxiliary functions
rtol_to_atol(S, p, atol, rtol) = rtol > 0 ? max(atol, TensorKit._norm(S, p) * rtol) : atol

function _compute_truncerr(Σdata, truncdim, p=2)
    I = keytype(Σdata)
    S = scalartype(valtype(Σdata))
    return TensorKit._norm((c => @view(v[(get(truncdim, c, 0) + 1):end])
                            for (c, v) in Σdata),
                           p, zero(S))
end

function _findnexttruncvalue(S, truncdim::SectorDict{I,Int}; by=identity,
                             rev::Bool=true) where {I<:Sector}
    # early return
    (isempty(S) || all(iszero, values(truncdim))) && return nothing
    if rev
        σmin, imin = findmin(keys(truncdim)) do c
            d = truncdim[c]
            return by(S[c][d])
        end
        return σmin, keys(truncdim)[imin]
    else
        σmax, imax = findmax(keys(truncdim)) do c
            d = truncdim[c]
            return by(S[c][d])
        end
        return σmax, keys(truncdim)[imax]
    end
end

# findtruncated
# -------------
# Generic fallback
function MAK.findtruncated_svd(values::SectorDict, strategy::TruncationStrategy)
    return MAK.findtruncated(values, strategy)
end

function MAK.findtruncated(values::SectorDict, ::NoTruncation)
    return SectorDict(c => Colon() for (c, b) in values)
end

function MAK.findtruncated(values::SectorDict, strategy::TruncationByOrder)
    perms = SectorDict(c => (sortperm(d; strategy.by, strategy.rev)) for (c, d) in values)
    values_sorted = SectorDict(c => d[perms[c]] for (c, d) in values)
    inds = MAK.findtruncated_svd(values_sorted, truncrank(strategy.howmany))
    return SectorDict(c => perms[c][I] for (c, I) in inds)
end
function MAK.findtruncated_svd(values::SectorDict, strategy::TruncationByOrder)
    I = keytype(values)
    truncdim = SectorDict{I,Int}(c => length(d) for (c, d) in values)
    totaldim = sum(dim(c) * d for (c, d) in truncdim; init=0)
    while true
        next = _findnexttruncvalue(values, truncdim; strategy.by, strategy.rev)
        isnothing(next) && break
        _, cmin = next
        truncdim[cmin] -= 1
        totaldim -= dim(cmin)
        truncdim[cmin] == 0 && delete!(truncdim, cmin)
        totaldim <= strategy.howmany && break
    end
    return SectorDict(c => Base.OneTo(d) for (c, d) in truncdim)
end

function MAK.findtruncated(values::SectorDict, strategy::TruncationByFilter)
    return SectorDict(c => findall(strategy.filter, d) for (c, d) in values)
end

function MAK.findtruncated(values::SectorDict, strategy::TruncationByValue)
    atol = rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    return SectorDict(c => MAK.findtruncated(d, strategy′) for (c, d) in values)
end
function MAK.findtruncated_svd(values::SectorDict, strategy::TruncationByValue)
    atol = rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    return SectorDict(c => MAK.findtruncated_svd(d, strategy′) for (c, d) in values)
end

function MAK.findtruncated(values::SectorDict, strategy::TruncationByError)
    perms = SectorDict(c => sortperm(d; by=abs, rev=true) for (c, d) in values)
    values_sorted = SectorDict(c => d[perms[c]] for (c, d) in Sd)
    inds = MAK.findtruncated_svd(values_sorted, truncrank(strategy.howmany))
    return SectorDict(c => perms[c][I] for (c, I) in inds)
end
function MAK.findtruncated_svd(values::SectorDict, strategy::TruncationByError)
    I = keytype(values)
    truncdim = SectorDict{I,Int}(c => length(d) for (c, d) in values)
    by(c, v) = abs(v)^strategy.p * dim(c)
    Nᵖ = sum(((c, v),) -> sum(Base.Fix1(by, c), v), values)
    ϵᵖ = max(strategy.atol^strategy.p, strategy.rtol^strategy.p * Nᵖ)
    truncerrᵖ = zero(real(scalartype(valtype(values))))
    next = _findnexttruncvalue(values, truncdim)
    while !isnothing(next)
        σmin, cmin = next
        truncerrᵖ += by(cmin, σmin)
        truncerrᵖ >= ϵᵖ && break
        (truncdim[cmin] -= 1) == 0 && delete!(truncdim, cmin)
        next = _findnexttruncvalue(values, truncdim)
    end
    return SectorDict{I,Base.OneTo{Int}}(c => Base.OneTo(d) for (c, d) in truncdim)
end

function MAK.findtruncated(values::SectorDict, strategy::TruncationSpace)
    blockstrategy(c) = truncrank(dim(strategy.space, c); strategy.by, strategy.rev)
    return SectorDict(c => MAK.findtruncated(d, blockstrategy(c)) for (c, d) in values)
end
function MAK.findtruncated_svd(values::SectorDict, strategy::TruncationSpace)
    blockstrategy(c) = truncrank(dim(strategy.space, c); strategy.by, strategy.rev)
    return SectorDict(c => MAK.findtruncated_svd(d, blockstrategy(c)) for (c, d) in values)
end

function MAK.findtruncated(values::SectorDict, strategy::TruncationIntersection)
    inds = map(Base.Fix1(MAK.findtruncated, values), strategy)
    return SectorDict(c => mapreduce(Base.Fix2(getindex, c), _ind_intersect, inds;
                                     init=trues(length(values[c])))
                      for c in intersect(map(keys, inds)...))
end
function MAK.findtruncated_svd(Sd::SectorDict, strategy::TruncationIntersection)
    inds = map(Base.Fix1(MAK.findtruncated_svd, Sd), strategy)
    return SectorDict(c => mapreduce(Base.Fix2(getindex, c), _ind_intersect, inds;
                                     init=trues(length(values[c])))
                      for c in intersect(map(keys, inds)...))
end
