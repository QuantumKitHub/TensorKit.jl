const CuSectorVector{T, I} = TensorKit.SectorVector{T, I, <:CuVector{T}}

function MatrixAlgebraKit.findtruncated(
        values::CuSectorVector, strategy::MatrixAlgebraKit.TruncationByOrder
    )
    I = sectortype(values)

    dims = similar(values, Base.promote_op(dim, I))
    for (c, v) in pairs(dims)
        fill!(v, dim(c))
    end

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

    perm = sortperm(parent(values); by = abs, rev = false)
    cumulative_err = cumsum(Base.permute!(parent(ϵᵖ), perm))

    result = similar(values, Bool)
    parent(result)[perm] .= cumulative_err .> ϵᵖmax
    return result
end

# Needed until MatrixAlgebraKit patch hits...
function MatrixAlgebraKit._ind_intersect(A::CuVector{Bool}, B::CuVector{Int})
    result = fill!(similar(A), false)
    result[B] .= @view A[B]
    return result
end
