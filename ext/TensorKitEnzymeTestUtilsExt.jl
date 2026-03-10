module TensorKitEnzymeTestUtilsExt

using TensorKit
using EnzymeTestUtils
using EnzymeTestUtils: Enzyme
import EnzymeTestUtils: to_vec, from_vec, rand_tangent

function EnzymeTestUtils.to_vec(x::TensorMap, seen_vecs::EnzymeTestUtils.AliasDict)
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const(Core.Typeof(x))
    if has_seen || is_const
        x_vec = Float32[]
    else
        vec_of_vecs = [b * TensorKit.sqrtdim(c) for (c, b) in blocks(x)]
        x_vec, back = to_vec(vec_of_vecs)
        seen_vecs[x] = x_vec
    end
    function TensorMap_from_vec(x_vec_new::AbstractVector, seen_xs::EnzymeTestUtils.AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Arrays must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return seen_xs[x]
        is_const && return x

        x_new = similar(x)
        xvec_of_vecs = back(x_vec_new)
        for (i, (c, b)) in enumerate(blocks(x_new))
            scale!(b, xvec_of_vecs[i], TensorKit.invsqrtdim(c))
        end
        if Core.Typeof(x_new) != Core.Typeof(x)
            x_new = Core.Typeof(x)(x_new)
        end
        seen_xs[x] = x_new
        return x_new
    end
    return x_vec, TensorMap_from_vec
end
function EnzymeTestUtils.to_vec(t::TensorKit.AdjointTensorMap, seen_vecs::EnzymeTestUtils.AliasDict)
    parent_vec, parent_t = to_vec(parent(t), seen_vecs)
    return parent_vec, adjoint ∘ parent_t
end
function EnzymeTestUtils.to_vec(t::TensorKit.DiagonalTensorMap, seen_vecs::EnzymeTestUtils.AliasDict)
    parent_vec, parent_t = to_vec(TensorMap(t), seen_vecs)
    return parent_vec, TensorKit.DiagonalTensorMap ∘ parent_t
end

# generate random tangents for testing
function EnzymeTestUtils.rand_tangent(rng, t::TensorMap)
    return TensorMap(EnzymeTestUtils.rand_tangent(rng, t.data), space(t))
end

function EnzymeTestUtils.rand_tangent(rng, t::TensorKit.AdjointTensorMap)
    return adjoint(rand_tangent(rng, parent(t)))
end

function EnzymeTestUtils.rand_tangent(rng, t::DiagonalTensorMap)
    return DiagonalTensorMap(EnzymeTestUtils.rand_tangent(rng, t.data), space(t, 1))
end

end
