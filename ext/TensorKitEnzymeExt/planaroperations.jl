# planartrace!
# ------------
# TODO: Fix planartrace pullback
# This implementation is slightly more involved than its non-planar counterpart
# this is because we lack a general `pAB` argument in `planarcontract`, and need
# to keep things planar along the way.
# In particular, we can't simply tensor product with multiple identities in one go
# if they aren't "contiguous", e.g. p = ((1, 4, 5), ()), q = ((2, 6), (3, 7))

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(TensorKit.planartrace!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const, allocator::Const
    ) where {RT}
    cacheC = !isa(β, Const) && copy(C.val)
    cacheA = EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing

    TensorKit.planartrace!(C.val, A.val, p.val, q.val, α.val, β.val, backend.val, allocator.val)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cacheC, cacheA))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(TensorKit.planartrace!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const, allocator::Const
    ) where {RT}
    cacheC, cacheA = cache
    Cval = something(cacheC, C.val)
    Aval = something(cacheA, A.val)
    if !isa(C, Const)
        if !isa(A, Const)
            TK.planartrace_pullback_ΔA!(A.dval, C.dval, Aval, p.val, q.val, α.val, backend.val, allocator.val)
        end
        Δαr = if !isa(α, Const)
            TK.planartrace_pullback_Δα(C.dval, A.val, p.val, q.val, α.val, backend.val, allocator.val)
        elseif !isa(α, Const)
            zero(α.val)
        else
            nothing
        end
        pullback_dC!(C.dval, β.val)
    end
    Δβr = pullback_dβ(β, C, Cval)
    return nothing, nothing, nothing, nothing, Δαr, Δβr, nothing, nothing
end
