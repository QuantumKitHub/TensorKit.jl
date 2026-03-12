for transform in (:permute, :transpose)
    add_transform! = Symbol(:add_, transform, :!)
    @eval function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof(TK.$add_transform!)},
            ::Type{RT},
            C::Annotation{<:AbstractTensorMap},
            A::Annotation{<:AbstractTensorMap},
            p::Const{<:Index2Tuple},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
            ba::Const...
        ) where {RT}
        C_cache = !isa(β, Const) && EnzymeRules.overwritten(config)[2] ? copy(C.val) : nothing
        # if we need to compute Δa, it is faster to allocate an intermediate permuted A
        # and store that instead of repeating the permutation in the pullback each time.
        # effectively, we replace `add_permute` by `add ∘ permute`.
        Ap = if !isa(α, Const)
            Ap = $transform(A.val, p.val)
            add!(C.val, Ap.val, α.val, β.val)
            Ap
        else
            TK.$add_transform!(C.val, A.val, p.val, α.val, β.val, ba.val...)
            nothing
        end
        cache = (C_cache, Ap)
        primal = EnzymeRules.needs_primal(config) ? C.val : nothing
        shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
        return EnzymeRules.AugmentedReturn(primal, shadow, cache)
    end
    @eval function EnzymeRules.reverse(
            config::EnzymeRules.RevConfigWidth{1},
            func::Const{typeof(TK.$add_transform!)},
            ::Type{RT},
            cache,
            C::Annotation{<:AbstractTensorMap},
            A::Annotation{<:AbstractTensorMap},
            p::Const{<:Index2Tuple},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
            ba::Const...
        ) where {RT}
        C_cache, Ap = cache
        Cval = something(C_cache, C.val)

        # ΔA
        ip = invperm(linearize(p.val))
        pΔA = _repartition(ip, A.val)
        TC = VectorInterface.promote_scale(C.dval, α.val)
        if scalartype(A.dval) <: Real && !(TC <: Real)
            ΔAc = TO.tensoralloc_add(TC, C.dval, pΔA, false, Val(false))
            TK.$add_transform!(ΔAc, C.dval, pΔA, conj(α.val), Zero(), ba.val...)
            add!(A.dval, real(ΔAc))
        else
            TK.$add_transform!(A.dval, C.dval, pΔA, conj(α.val), One(), ba.val...)
        end
        Δαr = isnothing(Ap) ? nothing : project_scalar(α.val, inner(Ap, C.dval))
        Δβr = pullback_dβ(C.dval, Cval, β)
        pullback_dC!(C.dval, β.val) # this typically returns nothing

        return nothing, nothing, nothing, nothing, Δαr, Δβr, map(Returns(nothing), ba)...
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.add_braid!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        levels::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        ba::Const...
    ) where {RT}
    C_cache = !isa(β, Const) && EnzymeRules.overwritten(config)[2] ? copy(C.val) : nothing
    # if we need to compute Δa, it is faster to allocate an intermediate braided A
    # and store that instead of repeating the permutation in the pullback each time.
    # effectively, we replace `add_permute` by `add ∘ permute`.
    Ap = if !isa(α, Const)
        Ap = braid(A.val, p.val, levels.val)
        add!(C.val, Ap, α.val, β.val)
        Ap
    else
        TK.add_braid!(C.val, A.val, p.val, levels.val, α.val, β.val, ba.val...)
        nothing
    end
    cache = (C_cache, Ap)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TK.add_braid!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        levels::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        ba::Const...
    ) where {RT}
    C_cache, Ap = cache
    Cval = something(C_cache, C.val)
    # ΔA
    ip = invperm(linearize(p.val))
    pΔA = _repartition(ip, A.val)
    ilevels = TupleTools.permute(levels.val, linearize(p.val))
    TC = VectorInterface.promote_scale(C.dval, α.val)
    if scalartype(A.dval) <: Real && !(TC <: Real)
        ΔAc = TO.tensoralloc_add(TC, C.dval, pΔA, false, Val(false))
        TK.add_braid!(ΔAc, C.dval, pΔA, ilevels, conj(α.val), Zero(), ba.val...)
        add!(A.dval, real(ΔAc))
    else
        TK.add_braid!(A.dval, C.dval, pΔA, ilevels, conj(α.val), One(), ba.val...)
    end
    Δαr = isnothing(Ap) ? nothing : project_scalar(α.val, inner(Ap, C.dval))
    Δβr = pullback_dβ(C.dval, C.val, β)
    pullback_dC!(C.dval, β.val) # this typically returns nothing
    return nothing, nothing, nothing, nothing, nothing, Δαr, Δβr, map(Returns(nothing), ba)...
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(twist!)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    twist!(t.val, inds.val; inv = false)
    primal = EnzymeRules.needs_primal(config) ? t.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? t.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(twist!)},
        ::Type{RT},
        cache,
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    twist!(t.dval, inds.val; inv = true)
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(Core.kwcall)},
        ::Type{RT},
        kwargs::Const{@NamedTuple{inv::Bool}},
        ::Const{typeof(twist!)},
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    inv = kwargs.val.inv
    twist!(t.val, inds.val; inv)
    primal = EnzymeRules.needs_primal(config) ? t.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? t.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(twist!)},
        ::Type{RT},
        cache,
        kwargs::Const{@NamedTuple{inv::Bool}},
        ::Const{typeof(twist!)},
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    inv = kwargs.val.inv
    twist!(t.dval, inds.val; inv = !inv)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(flip)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    t′ = flip(t.val, inds.val; inv = false)
    dt′ = make_zero(t′)
    primal = EnzymeRules.needs_primal(config) ? t′ : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dt′ : nothing
    cache = dt′
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(flip)},
        ::Type{RT},
        cache,
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    dt′ = cache
    dt′′ = flip(dt′, inds.val; inv = true)
    add!(t.dval, scalartype(t.dval) <: Real ? real(dt′′) : dt′′)
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(Core.kwcall)},
        ::Type{RT},
        kwargs::Const{@NamedTuple{inv::Bool}},
        ::Const{typeof(flip)},
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    inv = kwargs.val.inv
    t′ = flip(t.val, inds.val; inv = inv)
    dt′ = make_zero(t′)
    primal = EnzymeRules.needs_primal(config) ? t′ : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dt′ : nothing
    cache = dt′
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(Core.kwcall)},
        ::Type{RT},
        cache,
        kwargs::Const{@NamedTuple{inv::Bool}},
        ::Const{typeof(flip)},
        t::Annotation{<:AbstractTensorMap},
        inds::Const,
    ) where {RT}
    inv = kwargs.val.inv
    dt′ = cache
    dt′′ = flip(dt′, inds.val; inv = !inv)
    add!(t.dval, scalartype(t.dval) <: Real ? real(dt′′) : dt′′)
    return (nothing, nothing, nothing, nothing)
end

for insertunit in (:insertleftunit, :insertrightunit)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($insertunit)},
                ::Type{RT},
                tsrc::Annotation{<:AbstractTensorMap},
                ival::Const{<:Val},
            ) where {RT}
            tdst = $insertunit(tsrc.val, ival.val)
            Δtdst = make_zero(tdst)
            primal = EnzymeRules.needs_primal(config) ? tdst : nothing
            shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
            cache = (nothing, tdst, Δtdst)
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        # tdst shares data with tsrc if <:TensorMap, in this case we have to deal with correctly
        # sharing address spaces
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($insertunit)},
                ::Type{RT},
                tsrc::Annotation{<:TensorMap},
                ival::Const{<:Val},
            ) where {RT}
            tsrc_cache = copy(tsrc.val)
            tdst = $insertunit(tsrc.val, ival.val)
            Δtdst = $insertunit(tsrc.dval, ival.val)
            primal = EnzymeRules.needs_primal(config) ? tdst : nothing
            shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
            return cache = (tsrc_cache, tdst, Δtdst)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof($insertunit)},
                ::Type{RT},
                cache,
                tsrc::Annotation{<:AbstractTensorMap},
                ival::Const{<:Val},
            ) where {RT}
            tsrc_cache, tdst, Δtdst = cache
            # note: since data is already shared for <:TensorMap, don't have to do anything here!
            if isnothing(tsrc_cache)
                for (c, b) in blocks(Δtdst)
                    add!(block(tsrc.dval, c), b)
                end
            end
            return (nothing, nothing)
        end
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof(Core.kwcall)},
                ::Type{RT},
                kwargs::Const{<:NamedTuple},
                ::Const{typeof($insertunit)},
                tsrc::Annotation{<:AbstractTensorMap},
                ival::Const{<:Val},
            ) where {RT}
            if tsrc.val isa TensorMap && !get(kwargs.val, :copy, false)
                tsrc_cache = copy(tsrc.val)
                tdst = $insertunit(tsrc.val, ival.val; kwargs.val...)
                Δtdst = $insertunit(tsrc.dval, ival.val; kwargs.val...)
            else
                tsrc_cache = nothing
                tdst = $insertunit(tsrc.val, ival.val; kwargs.val...)
                Δtdst = make_zero(tdst)
            end
            primal = EnzymeRules.needs_primal(config) ? tdst : nothing
            shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
            cache = (tsrc_cache, tdst, Δtdst)
            return EnzymeRules.AugmentedReturn(primal, shadow, cache)
        end
        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfigWidth{1},
                func::Const{typeof(Core.kwcall)},
                ::Type{RT},
                cache,
                kwargs::Const{<:NamedTuple},
                ::Const{typeof($insertunit)},
                tsrc::Annotation{<:AbstractTensorMap},
                ival::Const{<:Val},
            ) where {RT}
            tsrc_cache, tdst, Δtdst = cache
            if isnothing(tsrc_cache)
                for (c, b) in blocks(Δtdst)
                    add!(block(Δtsrc, c), b)
                end
            end
            return (nothing, nothing, nothing, nothing)
        end
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(removeunit)},
        ::Type{RT},
        tsrc::Annotation{<:AbstractTensorMap},
        ival::Const{<:Val},
    ) where {RT}
    tsrc_cache = nothing
    tdst = removeunit(tsrc.val, ival.val)
    Δtdst = make_zero(tdst)
    primal = EnzymeRules.needs_primal(config) ? tdst : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
    cache = (tsrc_cache, tdst, Δtdst)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(removeunit)},
        ::Type{RT},
        tsrc::Annotation{<:TensorMap},
        ival::Const{<:Val},
    ) where {RT}
    # tdst shares data with tsrc if <:TensorMap, in this case we have to deal with correctly
    # sharing address spaces
    tsrc_cache = copy(tsrc.val)
    tdst = removeunit(tsrc.val, ival.val)
    Δtdst = removeunit(tsrc.dval, ival.val)
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
        ival::Const{<:Val},
    ) where {RT}
    tsrc_cache, tdst, Δtdst = cache
    # note: since data for <: TensorMap is already shared, don't have to do anything here!
    if isnothing(tsrc_cache)
        for (c, b) in blocks(Δtdst)
            add!(block(tsrc.dval, c), b)
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(Core.kwcall)},
        ::Type{RT},
        kwargs::Const{<:NamedTuple},
        ::Const{typeof(removeunit)},
        tsrc::Annotation{<:AbstractTensorMap},
        ival::Const{<:Val},
    ) where {RT}
    # tdst shares data with tsrc if <:TensorMap & copy=false, in this case we have to deal with correctly
    # sharing address spaces
    if tsrc.val isa TensorMap && !get(kwargs.val, :copy, false)
        tsrc_cache = copy(tsrc.val)
        Δtdst = removeunit(tsrc.dval, ival.val)
    else
        tsrc_cache = nothing
        Δtdst = make_zero(tdst)
    end
    tdst = removeunit(tsrc.val, ival.val; kwargs.val...)
    primal = EnzymeRules.needs_primal(config) ? tdst : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Δtdst : nothing
    cache = (tsrc_cache, tdst, Δtdst)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(Core.kwcall)},
        ::Type{RT},
        cache,
        kwargs::Const{<:NamedTuple},
        ::Const{typeof(removeunit)},
        tsrc::Annotation{<:AbstractTensorMap},
        ival::Const{<:Val},
    ) where {RT}
    tsrc_cache, tdst, Δtdst = cache
    # note: since data for <: TensorMap is already shared, don't have to do anything here!
    if isnothing(tsrc_cache)
        for (c, b) in blocks(Δtdst)
            add!(block(tsrc.dval, c), b)
        end
    end
    return (nothing, nothing, nothing, nothing)
end
