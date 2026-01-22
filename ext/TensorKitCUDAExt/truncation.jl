const CuSectorVector{T, I} = TensorKit.SectorVector{T, I, <: CuVector{T}}

function Factorizations._findtruncvalue_order(
    values::CuSectorVector, n::Int; by = identity, rev::Bool = false
)
    I = sectortype(values)
    p = sortperm(parent(values); by, rev)

    if FusionStyle(I) isa UniqueFusion # dimensions are all 1
        return n <= 0 ? nothing : @allowscalar(by(values[p[min(n, length(p))]]))
    else
        dims = similar(values, Base.promote_op(dim, I))
        for (c, v) in pairs(dims)
            fill!(v, dim(c))
        end
        cumulative_dim = cumsum(Base.permute!(parent(dims), p))
        k = findlast(<=(n), cumulative_dim)
        return isnothing(k) ? k : @allowscalar(by(values[p[k]]))
    end
end
