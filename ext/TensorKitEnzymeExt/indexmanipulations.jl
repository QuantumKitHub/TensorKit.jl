for transform in (:permute, :transpose)
    transform! = Symbol(transform, :!)
    transform_pb = Symbol(transform, :_pullback_dA)
    @eval function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof(TK.$transform!)},
            ::Type{RT},
            C::Annotation{<:AbstractTensorMap},
            A::Annotation{<:AbstractTensorMap},
            p::Const{<:Index2Tuple},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
            ba::Const...
        ) where {RT}
        C_cache = !isa(β, Const) ? copy(C.val) : nothing
        A_cache = EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
        # if we need to compute Δa, it is faster to allocate an intermediate permuted A
        # and store that instead of repeating the permutation in the pullback each time.
        # effectively, we replace `add_permute` by `add ∘ permute`.
        Ap = if !isa(α, Const)
            Ap = $transform(A.val, p.val)
            add!(C.val, Ap, α.val, β.val)
            Ap
        else
            bavs = map(a -> a.val, ba)
            TK.$transform!(C.val, A.val, p.val, α.val, β.val, bavs...)
            nothing
        end
        cache = (C_cache, A_cache, Ap)
        primal = EnzymeRules.needs_primal(config) ? C.val : nothing
        shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
        return EnzymeRules.AugmentedReturn(primal, shadow, cache)
    end
    @eval function EnzymeRules.reverse(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof(TK.$transform!)},
            ::Type{RT},
            cache,
            C::Annotation{<:AbstractTensorMap},
            A::Annotation{<:AbstractTensorMap},
            p::Const{<:Index2Tuple},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
            ba::Const...
        ) where {RT}
        C_cache, A_cache, Ap = cache
        Cval = something(C_cache, C.val)
        bavs = map(a -> a.val, ba)
        # ΔA
        if !isa(A, Const) && !isa(C, Const)
            Aval = something(A_cache, A.val)
            TK.$transform_pb(A.dval, Aval, C.dval, C.val, p.val, α.val, bavs...)
        end
        Δα = pullback_dα(α, C, Ap)
        Δβ = pullback_dβ(β, C, Cval)
        !isa(C, Const) && pullback_dC!(C.dval, β.val)
        return nothing, nothing, nothing, Δα, Δβ, map(Returns(nothing), ba)...
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.braid!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        levels::Const{<:IndexTuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        ba::Const...
    ) where {RT}
    C_cache = !isa(β, Const) ? copy(C.val) : nothing
    A_cache = EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
    # if we need to compute Δa, it is faster to allocate an intermediate braided A
    # and store that instead of repeating the permutation in the pullback each time.
    # effectively, we replace `add_permute` by `add ∘ permute`.
    Ap = if !isa(α, Const)
        Ap = braid(A.val, p.val, levels.val)
        add!(C.val, Ap, α.val, β.val)
        Ap
    else
        bavs = map(a -> a.val, ba)
        TK.braid!(C.val, A.val, p.val, levels.val, α.val, β.val, bavs...)
        nothing
    end
    cache = (C_cache, A_cache, Ap)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.braid!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        levels::Const{<:IndexTuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        ba::Const...
    ) where {RT}
    C_cache, A_cache, Ap = cache
    Cval = something(C_cache, C.val)
    Aval = something(A_cache, A.val)
    bavs = map(a -> a.val, ba)
    # ΔA
    if !isa(A, Const) && !isa(C, Const)
        TK.braid_pb(A.dval, Aval, C.dval, C.val, p.val, levels.val, α.val, bavs...)
    end
    Δαr = pullback_dα(α, C, Ap)
    Δβr = pullback_dβ(β, C, Cval)
    !isa(C, Const) && pullback_dC!(C.dval, β.val)
    return nothing, nothing, nothing, nothing, Δαr, Δβr, map(Returns(nothing), ba)...
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(flip)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        inds::Const;
        inv::Bool = false
    ) where {RT}
    t′ = flip(t.val, inds.val; inv)
    dt′ = make_zero(t′)
    cache = dt′
    primal = EnzymeRules.needs_primal(config) ? t′ : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dt′ : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(flip)},
        ::Type{RT},
        cache,
        t::Annotation{<:AbstractTensorMap},
        inds::Const;
        inv::Bool = false,
    ) where {RT}
    dt′ = cache
    if !isa(t, Const)
        dt′′ = flip(dt′, inds.val; inv = !inv)
        add!(t.dval, scalartype(t.dval) <: Real ? real(dt′′) : dt′′)
    end
    return (nothing, nothing)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(flip)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        inds::Annotation;
        inv::Bool = false,
    ) where {RT}
    t′ = flip(t.val, inds.val; inv)
    dt′ = !isa(t, Const) ? flip(t.dval, inds.val; inv) : make_zero(t.val)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(t′, dt′)
    elseif EnzymeRules.needs_primal(config)
        return t′
    elseif EnzymeRules.needs_shadow(config)
        return dt′
    else
        return nothing
    end
end
