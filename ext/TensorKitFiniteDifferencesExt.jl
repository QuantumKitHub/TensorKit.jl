module TensorKitFiniteDifferencesExt

using TensorKit
using TensorKit: sqrtdim, invsqrtdim, SectorVector
using VectorInterface: scale!
using FiniteDifferences

function FiniteDifferences.to_vec(t::AbstractTensorMap)
    # convert to vector of vectors to make use of existing functionality
    vec_of_vecs = [b * sqrtdim(c) for (c, b) in blocks(t)]
    vec, back = FiniteDifferences.to_vec(vec_of_vecs)

    function from_vec(x)
        t′ = similar(t)
        xvec_of_vecs = back(x)
        for (i, (c, b)) in enumerate(blocks(t′))
            scale!(b, xvec_of_vecs[i], invsqrtdim(c))
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

function FiniteDifferences.to_vec(t::DiagonalTensorMap)
    x_vec, back = to_vec(TensorMap(t))
    function DiagonalTensorMap_from_vec(x_vec)
        return DiagonalTensorMap(back(x_vec))
    end
    return x_vec, DiagonalTensorMap_from_vec
end

function FiniteDifferences.to_vec(v::SectorVector{T, <:Sector}) where {T}
    v_normalized = similar(v)
    for (c, b) in pairs(v)
        scale!(v_normalized[c], b, sqrtdim(c))
    end
    vec = parent(v_normalized)
    vec_real = T <: Real ? vec : collect(reinterpret(real(T), vec))

    function from_vec(x_real)
        x = T <: Real ? x_real : reinterpret(T, x_real)
        v_result = SectorVector(x, v.structure)
        for (c, b) in pairs(v_result)
            scale!(b, invsqrtdim(c))
        end
        return v_result
    end
    return vec_real, from_vec
end

end

# TODO: Investigate why the approach below doesn't work
# module TensorKitFiniteDifferencesExt

# using TensorKit
# using TensorKit: sqrtdim, invsqrtdim
# using VectorInterface: scale!
# using FiniteDifferences

# function FiniteDifferences.to_vec(t::AbstractTensorMap{T}) where {T}
#     # convert to vector of vectors to make use of existing functionality
#     structure = TensorKit.fusionblockstructure(t)
#     vec = storagetype(t)(undef, structure.totaldim)
#     for (c, ((d₁, d₂), r)) in structure.blockstructure
#         scale!(reshape(view(vec, r), (d₁, d₂)), block(t, c), sqrtdim(c))
#     end

#     function from_vec(x)
#         y = T <: Complex ? reinterpret(T, x) : x
#         t′ = similar(t)
#         for (c, ((d₁, d₂), r)) in structure.blockstructure
#             scale!(block(t′, c), reshape(view(y, r), (d₁, d₂)), invsqrtdim(c))
#         end
#         return t′
#     end

#     return T <: Complex ? reinterpret(real(T), vec) : vec, from_vec
# end

# end
