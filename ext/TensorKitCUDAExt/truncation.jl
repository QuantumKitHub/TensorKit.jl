const CuSectorVector{T, I} = TensorKit.SectorVector{T, I, <:CuVector{T}}

function MatrixAlgebraKit.findtruncated(
        values::CuSectorVector, strategy::MatrixAlgebraKit.TruncationByOrder
    )
    I = sectortype(values)

    dims = similar(values, Base.promote_op(dim, I))
    for (c, v) in pairs(dims)
        fill!(v, dim(c))
    end

    isempty(parent(values)) && return similar(values, Bool)

    perm = sortperm(parent(values); strategy.by, strategy.rev)
    cumulative_dim = cumsum(Base.permute!(parent(dims), perm))

    result = similar(values, Bool)
    parent(result)[perm] .= cumulative_dim .<= strategy.howmany
    return result
end

function MatrixAlgebraKit.findtruncated(
        values::CuSectorVector, strategy::MatrixAlgebraKit.TruncationByError
    )
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

    isempty(parent(values)) && return similar(values, Bool)

    perm = sortperm(parent(values); by = abs, rev = false)
    cumulative_err = cumsum(Base.permute!(parent(ϵᵖ), perm))

    result = similar(values, Bool)
    parent(result)[perm] .= cumulative_err .> ϵᵖmax
    return result
end

function MatrixAlgebraKit.findtruncated_svd(values::CuSectorVector, strategy::S) where {S <: MatrixAlgebraKit.TruncationStrategy}
    # returning a CuSectorVector wrecks things in truncate_{co}domain
    # because of scalar indexing
    return CUDA.CUDACore.Adapt.adapt(Vector, MatrixAlgebraKit.findtruncated(values, strategy))
end

for strat in (:(MatrixAlgebraKit.TruncationByOrder), :(MatrixAlgebraKit.TruncationByError), :(MatrixAlgebraKit.TruncationIntersection), :(TensorKit.Factorizations.TruncationSpace))
    @eval function MatrixAlgebraKit.findtruncated_svd(values::CuSectorVector, strategy::$strat)
        # returning a CuSectorVector wrecks things in truncate_{co}domain
        # because of scalar indexing
        return CUDA.CUDACore.Adapt.adapt(Vector, MatrixAlgebraKit.findtruncated(values, strategy))
    end
end

function MatrixAlgebraKit.findtruncated_svd(values::CuSectorVector, strategy::MatrixAlgebraKit.TruncationByValue)
    atol = TensorKit.Factorizations.rtol_to_atol(values, strategy.p, strategy.atol, strategy.rtol)
    strategy′ = trunctol(; atol, strategy.by, strategy.keep_below)
    return SectorDict(c => CUDA.CUDACore.Adapt.adapt(Vector, MatrixAlgebraKit.findtruncated_svd(d, strategy′)) for (c, d) in pairs(values))
end

# Needed until MatrixAlgebraKit patch hits...
function MatrixAlgebraKit._ind_intersect(A::CuVector{Bool}, B::CuVector{Int})
    result = fill!(similar(A), false)
    result[B] .= @view A[B]
    return result
end
