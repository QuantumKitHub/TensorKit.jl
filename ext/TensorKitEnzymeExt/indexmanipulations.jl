function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.add_transform!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Annotation{<:Index2Tuple},
        transformer::Annotation,
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        ba::Const...
    ) where {RT}
    C_cache = !isa(β, Const) ? copy(C.val) : C.val
    A_cache = EnzymeRules.overwritten(config)[3] ? copy(A.val) : A.val
    # if we need to compute Δa, it is faster to allocate an intermediate braided A
    # and store that instead of repeating the permutation in the pullback each time.
    # effectively, we replace `add_permute` by `add ∘ permute`.
    bavs = map(a -> a.val, ba)
    Ap = if !isa(α, Const)
        Ap = TK.add_transform!(Enzyme.make_zero(C.val), A.val, p.val, transformer.val, One(), Zero(), bavs...)
        add!(C.val, Ap, α.val, β.val)
        Ap
    else
        TK.add_transform!(C.val, A.val, p.val, transformer.val, α.val, β.val, bavs...)
        nothing
    end
    cache = (C_cache, A_cache, Ap)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.add_transform!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Annotation{<:Index2Tuple},
        transformer::Annotation,
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        ba::Const...
    ) where {RT}
    C_cache, A_cache, Ap = cache
    Cval = C_cache
    Aval = A_cache
    bavs = map(a -> a.val, ba)
    # ΔA
    if !isa(A, Const) && !isa(C, Const)
        ip = invperm(linearize(p.val))
        pΔA = TO.repartition(ip, numout(Aval))
        TC = VectorInterface.promote_scale(Cval, α.val)
        if scalartype(A.dval) <: Real && !(TC <: Real)
            ΔAc = TO.tensoralloc_add(TC, C.dval, pΔA, false, Val(false))
            TK.add_transform!(ΔAc, C.dval, pΔA, transformer.val, conj(α.val), Zero(), bavs...)
            add!(A.dval, real(ΔAc))
        else
            TK.add_transform!(A.dval, C.dval, pΔA, transformer.val, conj(α.val), One(), bavs...)
        end
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
