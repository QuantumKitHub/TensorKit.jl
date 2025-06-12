# truncation.jl
#
# Implements truncation schemes for truncating a tensor with svd, leftorth or rightorth

notrunc() = NoTruncation()
# deprecate
const TruncationScheme = TruncationStrategy

struct TruncationError{T<:Real} <: TruncationStrategy
    ϵ::T
    p::Real
end
truncerr(epsilon::Real, p::Real=2) = TruncationError(epsilon, p)

# struct TruncationDimension <: TruncationScheme
#     dim::Int
# end
@deprecate truncdim(d::Int) truncrank(d)

struct TruncationSpace{S<:ElementarySpace} <: TruncationStrategy
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)

struct TruncationCutoff{T<:Real} <: TruncationStrategy
    ϵ::T
    add_back::Int
end
@deprecate truncbelow(ϵ::Real, add_back::Int=0) begin
    add_back == 0 || @warn "add_back is ignored"
    trunctol(ϵ)
end
# truncbelow(epsilon::Real, add_back::Int=0) = TruncationCutoff(epsilon, add_back)

# Compute the total truncation error given truncation dimensions
function _compute_truncerr(Σdata, truncdim, p=2)
    I = keytype(Σdata)
    S = scalartype(valtype(Σdata))
    return TensorKit._norm((c => @view(v[(get(truncdim, c, 0) + 1):end])
                            for (c, v) in Σdata),
                           p, zero(S))
end

# Compute truncation dimensions
# function _compute_truncdim(Σdata, ::NoTruncation, p=2)
#     I = keytype(Σdata)
#     truncdim = SectorDict{I,Int}(c => length(v) for (c, v) in Σdata)
#     return truncdim
# end
# function _compute_truncdim(Σdata, trunc::TruncationDimension, p=2)
#     I = keytype(Σdata)
#     truncdim = SectorDict{I,Int}(c => length(v) for (c, v) in Σdata)
#     while sum(dim(c) * d for (c, d) in truncdim) > trunc.dim
#         cmin = _findnexttruncvalue(Σdata, truncdim, p)
#         isnothing(cmin) && break
#         truncdim[cmin] -= 1
#     end
#     return truncdim
# end
function _compute_truncdim(Σdata, trunc::TruncationSpace, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{I,Int}(c => min(length(v), dim(trunc.space, c))
                                 for (c, v) in Σdata)
    return truncdim
end

# function _compute_truncdim(Σdata, trunc::TruncationCutoff, p=2)
#     I = keytype(Σdata)
#     truncdim = SectorDict{I,Int}(c => length(v) for (c, v) in Σdata)
#     for (c, v) in Σdata
#         newdim = findlast(Base.Fix2(>, trunc.ϵ), v)
#         if newdim === nothing
#             truncdim[c] = 0
#         else
#             truncdim[c] = newdim
#         end
#     end
#     for i in 1:(trunc.add_back)
#         cmax = _findnextgrowvalue(Σdata, truncdim, p)
#         isnothing(cmax) && break
#         truncdim[cmax] += 1
#     end
#     return truncdim
# end

# Combine truncations
# struct MultipleTruncation{T<:Tuple{Vararg{TruncationScheme}}} <: TruncationScheme
#     truncations::T
# end
# function Base.:&(a::MultipleTruncation, b::MultipleTruncation)
#     return MultipleTruncation((a.truncations..., b.truncations...))
# end
# function Base.:&(a::MultipleTruncation, b::TruncationScheme)
#     return MultipleTruncation((a.truncations..., b))
# end
# function Base.:&(a::TruncationScheme, b::MultipleTruncation)
#     return MultipleTruncation((a, b.truncations...))
# end
# Base.:&(a::TruncationScheme, b::TruncationScheme) = MultipleTruncation((a, b))

# function _compute_truncdim(Σdata, trunc::MultipleTruncation, p::Real=2)
#     truncdim = _compute_truncdim(Σdata, trunc.truncations[1], p)
#     for k in 2:length(trunc.truncations)
#         truncdimₖ = _compute_truncdim(Σdata, trunc.truncations[k], p)
#         for (c, d) in truncdim
#             truncdim[c] = min(d, truncdimₖ[c])
#         end
#     end
#     return truncdim
# end

# auxiliary function
function _findnexttruncvalue(S, truncdim::SectorDict{I,Int}) where {I<:Sector}
    # early return
    (isempty(S) || all(iszero, values(truncdim))) && return nothing
    σmin, imin = findmin(keys(truncdim)) do c
        d = truncdim[c]
        return S[c][d]
    end
    return σmin, keys(truncdim)[imin]
end

function _findnextgrowvalue(Σdata, truncdim::SectorDict{I,Int}, p::Real) where {I<:Sector}
    istruncated = SectorDict{I,Bool}(c => (d < length(Σdata[c])) for (c, d) in truncdim)
    # early return
    (isempty(Σdata) || !any(values(istruncated))) && return nothing

    # find some suitable starting candidate
    cmax = findfirst(istruncated)
    σmax = Σdata[cmax][truncdim[cmax] + 1]

    # find the actual maximal singular value
    for (c, σs) in Σdata
        if istruncated[c]
            σ = σs[truncdim[c] + 1]
            if σ > σmax
                cmax, σmax = c, σ
            end
        end
    end
    return cmax
end

# Truncation
# ----------

function truncate!(::typeof(svd_trunc!), (U, S, Vᴴ)::_T_USVᴴ, strategy::TruncationStrategy)
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

function findtruncated_sorted(S::SectorDict, strategy::TruncationKeepAbove)
    atol = if strategy.rtol > 0
        max(strategy.atol, _norm(S, strategy.p) * strategy.rtol)
    else
        strategy.atol
    end
    findtrunc = Base.Fix2(findtruncated_sorted, truncbelow(atol))
    return SectorDict(c => findtrunc(d) for (c, d) in Sd)
end

function findtruncated_sorted(S::SectorDict, strategy::TruncationKeepBelow)
    atol = if strategy.rtol > 0
        max(strategy.atol, _norm(S, strategy.p) * strategy.rtol)
    else
        strategy.atol
    end
    findtrunc = Base.Fix2(findtruncated_sorted, truncabove(atol))
    return SectorDict(c => findtrunc(d) for (c, d) in Sd)
end

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationError)
    I = keytype(Sd)
    S = real(scalartype(valtype(Sd)))
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
    @assert strategy.by === abs && strategy.rev == true "Not implemented"
    I = keytype(Sd)
    S = real(scalartype(valtype(Sd)))
    truncdim = SectorDict{I,Int}(c => length(d) for (c, d) in Sd)
    totaldim = sum(dim(c) * d for (c, d) in truncdim; init=0)
    while true
        next = _findnexttruncvalue(Sd, truncdim)
        isnothing(next) && break
        _, cmin = next
        truncdim[cmin] -= 1
        totaldim -= dim(cmin)
        if totaldim < strategy.howmany
            truncdim[cmin] += 1
            break
        end
        if truncdim[cmin] == 0
            delete!(truncdim, cmin)
        end
    end
    return SectorDict{I,Base.OneTo{Int}}(c => Base.OneTo(d) for (c, d) in truncdim)
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

function findtruncated_sorted(Sd::SectorDict, strategy::TruncationIntersection)
    inds = map(Base.Fix1(findtruncated_sorted, Sd), strategy)
    return SectorDict(c => intersect(map(Base.Fix2(getindex, c), inds)...)
                      for c in intersect(map(keys, inds)...))
end

function MatrixAlgebraKit.truncate!(::typeof(left_null!),
                                    (U, S)::Tuple{<:AbstractTensorMap,
                                                  <:AbstractTensorMap},
                                    strategy::MatrixAlgebraKit.TruncationStrategy)
    extended_S = SectorDict(c => vcat(MatrixAlgebraKit.diagview(b),
                                      zeros(eltype(b), max(0, size(b, 2) - size(b, 1))))
                            for (c, b) in blocks(S))
    ind = MatrixAlgebraKit.findtruncated(extended_S, strategy)
    V_truncated = spacetype(S)(c => length(axes(b, 1)[ind[c]]) for (c, b) in blocks(S))
    Ũ = similar(U, codomain(U) ← V_truncated)
    for (c, b) in blocks(Ũ)
        copy!(b, @view(block(U, c)[:, ind[c]]))
    end
    return Ũ
end
