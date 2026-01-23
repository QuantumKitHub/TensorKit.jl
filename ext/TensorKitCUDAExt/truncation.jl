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

# Needed until MatrixAlgebraKit patch hits...
function MatrixAlgebraKit._ind_intersect(A::CuVector{Bool}, B::CuVector{Int})
    result = fill!(similar(A), false)
    result[B] .= @view A[B]
    return result
end
