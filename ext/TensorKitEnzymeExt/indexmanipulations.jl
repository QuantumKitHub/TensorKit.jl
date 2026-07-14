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
        func::Const{typeof(twist!)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        inds::Const;
        inv::Bool = false
    ) where {RT}
    twist!(t.val, inds.val; inv)
    primal = EnzymeRules.needs_primal(config) ? t.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? t.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(twist!)},
        ::Type{RT},
        cache,
        t::Annotation{<:AbstractTensorMap},
        inds::Const;
        inv::Bool = false
    ) where {RT}
    !isa(t, Const) && twist!(t.dval, inds.val; inv = !inv)
    return (nothing, nothing)
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

for insertunit in (:insertleftunit, :insertrightunit)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($insertunit)},
                ::Type{RT},
                tsrc::Annotation{<:AbstractTensorMap},
                ival::Const{<:Val};
                kwargs...
            ) where {RT}
            if tsrc.val isa TensorMap && !get(kwargs, :copy, false) && !isa(tsrc, Const)
                tsrc_cache = tsrc.val
                tdst = $insertunit(tsrc.val, ival.val; kwargs...)
                Δtdst = $insertunit(tsrc.dval, ival.val; kwargs...)
            else
                tsrc_cache = nothing
                tdst = $insertunit(tsrc.val, ival.val; kwargs...)
                Δtdst = make_zero(tdst)
            end
            primal = EnzymeRules.needs_primal(config) ? tdst : nothing
            shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
            cache = (tsrc_cache, tdst, Δtdst)
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($insertunit)},
                ::Type{RT},
                cache,
                tsrc::Annotation{<:AbstractTensorMap},
                ival::Const{<:Val};
                kwargs...
            ) where {RT}
            tsrc_cache, tdst, Δtdst = cache
            # note: since data is already shared for <:TensorMap, don't have to do anything here!
            if isnothing(tsrc_cache) && !isa(tsrc, Const)
                for (c, b) in blocks(Δtdst)
                    add!(block(tsrc.dval, c), b)
                end
            end
            return (nothing, nothing)
        end
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(removeunit)},
        ::Type{RT},
        tsrc::Annotation{<:AbstractTensorMap},
        ival::Const{<:Val};
        kwargs...
    ) where {RT}
    # tdst shares data with tsrc if <:TensorMap & copy=false, in this case we have to deal with correctly
    # sharing address spaces
    if tsrc.val isa TensorMap && !get(kwargs, :copy, false) && !isa(tsrc, Const)
        tsrc_cache = tsrc.val
        tdst = removeunit(tsrc.val, ival.val; kwargs...)
        Δtdst = removeunit(tsrc.dval, ival.val)
    else
        tsrc_cache = nothing
        tdst = removeunit(tsrc.val, ival.val; kwargs...)
        Δtdst = make_zero(tdst)
    end
    primal = EnzymeRules.needs_primal(config) ? tdst : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
    cache = (tsrc_cache, tdst, Δtdst)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(removeunit)},
        ::Type{RT},
        cache,
        tsrc::Annotation{<:AbstractTensorMap},
        ival::Const{<:Val};
        kwargs...
    ) where {RT}
    tsrc_cache, tdst, Δtdst = cache
    # note: since data for <: TensorMap is already shared, don't have to do anything here!
    if isnothing(tsrc_cache) && !isa(tsrc, Const)
        for (c, b) in blocks(Δtdst)
            add!(block(tsrc.dval, c), b)
        end
    end
    return (nothing, nothing)
end
