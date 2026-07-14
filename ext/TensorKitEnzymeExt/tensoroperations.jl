# tensorcontract!
# ---------------
# TODO: it might be beneficial to compare here if it would make sense to simply compute the
# rrule of permute-permute-gemm-permute, rather than using the contractions directly.
# This could possibly out save some permutations being carried out twice, at the cost of having
# to store some more intermediate objects.
# For example, the combination `ΔC, pΔC, false` appears in the pullback for ΔA and ΔB, so effectively
# this permutation is done multiple times.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorKit.blas_contract!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        pA::Const{<:Index2Tuple},
        B::Annotation{<:AbstractTensorMap},
        pB::Const{<:Index2Tuple},
        pAB::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    Ccache = isa(β, Const) ? nothing : copy(C.val)
    A_needs_cache = EnzymeRules.overwritten(config)[3] && !(typeof(B) <: Const) && !(typeof(C) <: Const)
    Acache = A_needs_cache ? copy(A.val) : nothing
    B_needs_cache = EnzymeRules.overwritten(config)[5] && !(typeof(A) <: Const) && !(typeof(C) <: Const)
    Bcache = B_needs_cache ? copy(B.val) : nothing
    AB = if !isa(α, Const)
        AB = TO.tensorcontract(A.val, pA.val, false, B.val, pB.val, false, pAB.val, One(), backend.val, allocator.val)
        add!(C.val, AB, α.val, β.val)
        AB
    else
        TensorKit.blas_contract!(C.val, A.val, pA.val, B.val, pB.val, pAB.val, α.val, β.val, backend.val, allocator.val)
        nothing
    end
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    cache = (Ccache, Acache, Bcache, AB)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorKit.blas_contract!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        pA::Const{<:Index2Tuple},
        B::Annotation{<:AbstractTensorMap},
        pB::Const{<:Index2Tuple},
        pAB::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    cacheC, cacheA, cacheB, AB = cache
    Cval = cacheC
    Aval = something(cacheA, A.val)
    Bval = something(cacheB, B.val)

    Δα = pullback_dα(α, C, AB)
    Δβ = pullback_dβ(β, C, Cval)

    if !isa(A, Const)
        TensorKit.blas_contract_pullback_ΔA!(
            A.dval, C.dval, Aval, pA.val, Bval, pB.val, pAB.val, α.val, backend.val, allocator.val
        ) # this typically returns nothing
    end
    if !isa(B, Const)
        TensorKit.blas_contract_pullback_ΔB!(
            B.dval, C.dval, Aval, pA.val, Bval, pB.val, pAB.val, α.val, backend.val, allocator.val
        ) # this typically returns nothing
    end
    !isa(C, Const) && pullback_dC!(C.dval, β.val) # this typically returns nothing
    return nothing, nothing, nothing, nothing, nothing, nothing, Δα, Δβ, nothing, nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(TensorKit.blas_contract!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        pA::Annotation{<:Index2Tuple},
        B::Annotation{<:AbstractTensorMap},
        pB::Annotation{<:Index2Tuple},
        pAB::Annotation{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    # ΔC′ = ΔC*β + C*Δβ + A*B*Δα + ΔA*B*α + A*ΔB*α
    if !isa(C, Const)
        if isa(β, Const)
            scale!(C.dval, β.val)
        else
            add!(C.dval, C.val, β.dval, β.val)
        end
        !isa(α, Const) && TensorKit.blas_contract!(C.dval, A.val, pA.val, B.val, pB.val, pAB.val, α.dval, One(), backend.val, allocator.val)
        !isa(A, Const) && TensorKit.blas_contract!(C.dval, A.dval, pA.val, B.val, pB.val, pAB.val, α.val, One(), backend.val, allocator.val)
        !isa(B, Const) && TensorKit.blas_contract!(C.dval, A.val, pA.val, B.dval, pB.val, pAB.val, α.val, One(), backend.val, allocator.val)
    end
    TensorKit.blas_contract!(C.val, A.val, pA.val, B.val, pB.val, pAB.val, α.val, β.val, backend.val, allocator.val)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return C
    elseif EnzymeRules.needs_primal(config)
        return C.val
    elseif EnzymeRules.needs_shadow(config)
        return C.dval
    else
        return nothing
    end
end
