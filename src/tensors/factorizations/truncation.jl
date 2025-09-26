# Strategies
# ----------
"""
    notrunc()
"""
notrunc() = NoTruncation()

# deprecate
const TruncationScheme = TruncationStrategy

# TODO: add this to MatrixAlgebraKit
struct TruncationError{T<:Real} <: TruncationStrategy
    ϵ::T
    p::Real
end

"""
    truncerr(epsilon, p)
"""
truncerr(epsilon::Real, p::Real=2) = TruncationError(epsilon, p)

struct TruncationSpace{S<:ElementarySpace} <: TruncationStrategy
    space::S
end

"""
    truncspace(space::ElementarySpace)

Truncation strategy to keep the first values such that the resulting space is the infimum of
the total space and the provided space.
"""
truncspace(space::ElementarySpace) = TruncationSpace(space)

# Truncation
# ----------
function truncate!(::typeof(svd_trunc!),
                   (U, S, Vᴴ)::Tuple{AbstractTensorMap,AbstractTensorMap,AbstractTensorMap},
                   strategy::TruncationStrategy)
    strategy == notrunc() && return (U, S, Vᴴ)
    ind = findtruncated_sorted(diagview(S), strategy)
    V_truncated = spacetype(S)(c => length(I) for (c, I) in ind)

    Ũ = similar(U, codomain(U) ← V_truncated)
    for (c, b) in blocks(Ũ)
        I = get(ind, c, nothing)
        @assert !isnothing(I)
        copy!(b, @view(block(U, c)[:, I]))
    end

    S̃ = DiagonalTensorMap{scalartype(S)}(undef, V_truncated)
    for (c, b) in blocks(S̃)
        I = get(ind, c, nothing)
        @assert !isnothing(I)
        copy!(b.diag, @view(block(S, c).diag[I]))
    end

    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    for (c, b) in blocks(Ṽᴴ)
        I = get(ind, c, nothing)
        @assert !isnothing(I)
        copy!(b, @view(block(Vᴴ, c)[I, :]))
    end

    return Ũ, S̃, Ṽᴴ
end

function truncate!(::typeof(left_null!),
                   (U, S)::Tuple{AbstractTensorMap,AbstractTensorMap},
                   strategy::MatrixAlgebraKit.TruncationStrategy)
    strategy == notrunc() && return (U, S)
    extended_S = SectorDict(c => vcat(diagview(b),
                                      zeros(eltype(b), max(0, size(b, 2) - size(b, 1))))
                            for (c, b) in blocks(S))
    ind = findtruncated(extended_S, strategy)
    V_truncated = spacetype(S)(c => length(axes(b, 1)[ind[c]]) for (c, b) in blocks(S))
    Ũ = similar(U, codomain(U) ← V_truncated)
    for (c, b) in blocks(Ũ)
        copy!(b, @view(block(U, c)[:, ind[c]]))
    end
    return Ũ
end

for f! in (:eig_trunc!, :eigh_trunc!)
    @eval function truncate!(::typeof($f!),
                             (D, V)::Tuple{AbstractTensorMap,AbstractTensorMap},
                             strategy::TruncationStrategy)
        strategy == notrunc() && return (D, V)
        ind = findtruncated(diagview(D), strategy)
        V_truncated = spacetype(D)(c => length(I) for (c, I) in ind)

        D̃ = DiagonalTensorMap{scalartype(D)}(undef, V_truncated)
        for (c, b) in blocks(D̃)
            I = get(ind, c, nothing)
            @assert !isnothing(I)
            copy!(b.diag, @view(block(D, c).diag[I]))
        end

        Ṽ = similar(V, codomain(V) ← V_truncated)
        for (c, b) in blocks(Ṽ)
            I = get(ind, c, nothing)
            @assert !isnothing(I)
            copy!(b, @view(block(V, c)[:, I]))
        end

        return D̃, Ṽ
    end
end

# Find truncation
# ---------------
# auxiliary functions
rtol_to_atol(S, p, atol, rtol) = rtol > 0 ? max(atol, _norm(S, p) * rtol) : atol

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

# implementations
function findtruncated_sorted(S::SectorDict, strategy::TruncationStrategy)
    return findtruncated(S, strategy)
end

function findtruncated_sorted(S::SectorDict, strategy::TruncationKeepAbove)
    atol = rtol_to_atol(S, strategy.p, strategy.atol, strategy.rtol)
    findtrunc = Base.Fix2(findtruncated_sorted, truncbelow(atol))
    return SectorDict(c => findtrunc(d) for (c, d) in S)
end
function findtruncated(S::SectorDict, strategy::TruncationKeepAbove)
    atol = rtol_to_atol(S, strategy.p, strategy.atol, strategy.rtol)
    findtrunc = Base.Fix2(findtruncated, truncbelow(atol))
    return SectorDict(c => findtrunc(d) for (c, d) in Sd)
end

function findtruncated_sorted(S::SectorDict, strategy::TruncationKeepBelow)
    atol = rtol_to_atol(S, strategy.p, strategy.atol, strategy.rtol)
    findtrunc = Base.Fix2(findtruncated_sorted, truncabove(atol))
    return SectorDict(c => findtrunc(d) for (c, d) in Sd)
end
function findtruncated(S::SectorDict, strategy::TruncationKeepBelow)
    atol = rtol_to_atol(S, strategy.p, strategy.atol, strategy.rtol)
    findtrunc = Base.Fix2(findtruncated, truncabove(atol))
    return SectorDict(c => findtrunc(d) for (c, d) in Sd)
end

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationError)
    I = keytype(Sd)
    truncdim = SectorDict{I,Int}(c => length(d) for (c, d) in Sd)
    while true
        next = _findnexttruncvalue(Sd, truncdim)
        isnothing(next) && break
        σmin, cmin = next
        truncdim[cmin] -= 1
        err = _compute_truncerr(Sd, truncdim, strategy.p)
        if err > strategy.ϵ
            truncdim[cmin] += 1
            break
        end
        if truncdim[cmin] == 0
            delete!(truncdim, cmin)
        end
    end
    return SectorDict{I,Base.OneTo{Int}}(c => Base.OneTo(d) for (c, d) in truncdim)
end

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationKeepSorted)
    return findtruncated(Sd, strategy)
end
function findtruncated(Sd::SectorDict, strategy::TruncationKeepSorted)
    permutations = SectorDict(c => (sortperm(d; strategy.by, strategy.rev))
                              for (c, d) in Sd)
    Sd = SectorDict(c => sort(d; strategy.by, strategy.rev) for (c, d) in Sd)

    I = keytype(Sd)
    truncdim = SectorDict{I,Int}(c => length(d) for (c, d) in Sd)
    totaldim = sum(dim(c) * d for (c, d) in truncdim; init=0)
    while true
        next = _findnexttruncvalue(Sd, truncdim; strategy.by, strategy.rev)
        isnothing(next) && break
        _, cmin = next
        truncdim[cmin] -= 1
        totaldim -= dim(cmin)
        if truncdim[cmin] == 0
            delete!(truncdim, cmin)
        end
        totaldim <= strategy.howmany && break
    end
    return SectorDict(c => permutations[c][Base.OneTo(d)] for (c, d) in truncdim)
end

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationSpace)
    I = keytype(Sd)
    return SectorDict{I,Base.OneTo{Int}}(c => Base.OneTo(min(length(d),
                                                             dim(strategy.space, c)))
                                         for (c, d) in Sd)
end

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationKeepFiltered)
    return SectorDict(c => findtruncated_sorted(d, strategy) for (c, d) in Sd)
end
function findtruncated(Sd::SectorDict, strategy::TruncationKeepFiltered)
    return SectorDict(c => findtruncated(d, strategy) for (c, d) in Sd)
end

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated_sorted, Sd), strategy)
    return SectorDict(c => intersect(map(Base.Fix2(getindex, c), inds)...)
                      for c in intersect(map(keys, inds)...))
end
function findtruncated(Sd::SectorDict, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated, Sd), strategy)
    return SectorDict(c => intersect(map(Base.Fix2(getindex, c), inds)...)
                      for c in intersect(map(keys, inds)...))
end
