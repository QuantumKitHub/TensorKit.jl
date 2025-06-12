# convenience to set default
macro check_space(x, V)
    return esc(:($MatrixAlgebraKit.@check_size($x, $V, $space)))
end
macro check_scalar(x, y, op=:identity, eltype=:scalartype)
    return esc(:($MatrixAlgebraKit.@check_scalar($x, $y, $op, $eltype)))
end

function factorisation_scalartype(t::AbstractTensorMap)
    T = scalartype(t)
    return promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))
end
factorisation_scalartype(f, t) = factorisation_scalartype(t)

function permutedcopy_oftype(t::AbstractTensorMap, T::Type{<:Number}, p::Index2Tuple)
    return permute!(similar(t, T, permute(space(t), p)), t, p)
end
function copy_oftype(t::AbstractTensorMap, T::Type{<:Number})
    return copy!(similar(t, T), t)
end

function _reverse!(t::AbstractTensorMap; dims=:)
    for (c, b) in blocks(t)
        reverse!(b; dims)
    end
    return t
end

diagview(t::AbstractTensorMap) = SectorDict(c => diagview(b) for (c, b) in blocks(t))
