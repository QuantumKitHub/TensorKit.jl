# planartrace!
# ------------
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.planartrace!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    C_cache = !isa(β, Const) ? copy(C.val) : nothing
    # if we need to compute Δα, store the trace of A instead of recomputing it in the pullback
    At = if !isa(α, Const)
        At = TK.planartrace!(zerovector(C.val), A.val, p.val, q.val, One(), Zero(), backend.val, allocator.val)
        add!(C.val, At, α.val, β.val)
        At
    else
        TK.planartrace!(C.val, A.val, p.val, q.val, α.val, β.val, backend.val, allocator.val)
        nothing
    end
    cache = (C_cache, At)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.planartrace!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    C_cache, At = cache
    Cval = something(C_cache, C.val)
    # the pullback of A is linear, so only the spaces of A are needed
    if !is_inactive(config, A) && !is_inactive(config, C)
        TK.planartrace_pullback_ΔA!(A.dval, C.dval, A.val, p.val, q.val, α.val, backend.val, allocator.val)
    end
    Δα = pullback_dα(α, C, At)
    Δβ = pullback_dβ(β, C, Cval)
    !is_inactive(config, C) && pullback_dC!(C.dval, β.val)
    return nothing, nothing, nothing, nothing, Δα, Δβ, nothing, nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(TK.planartrace!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Annotation{<:Index2Tuple},
        q::Annotation{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    # ΔC′ = ΔC*β + C*Δβ + tr(A)*Δα + tr(ΔA)*α
    if !isa(C, Const)
        if isa(β, Const)
            scale!(C.dval, β.val)
        else
            add!(C.dval, C.val, β.dval, β.val)
        end
        !isa(α, Const) && TK.planartrace!(C.dval, A.val, p.val, q.val, α.dval, One(), backend.val, allocator.val)
        !isa(A, Const) && TK.planartrace!(C.dval, A.dval, p.val, q.val, α.val, One(), backend.val, allocator.val)
    end
    TK.planartrace!(C.val, A.val, p.val, q.val, α.val, β.val, backend.val, allocator.val)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return C
    elseif EnzymeRules.needs_primal(config)
        return C.val
    elseif EnzymeRules.needs_shadow(config) && !isa(C, Const)
        return C.dval
    elseif EnzymeRules.needs_shadow(config) && isa(C, Const)
        return Enzyme.make_zero(C.val)
    else
        return nothing
    end
end
