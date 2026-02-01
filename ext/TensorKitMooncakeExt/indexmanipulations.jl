for transform in (:permute, :transpose)
    add_transform! = Symbol(:add_, transform, :!)
    add_transform_pullback = Symbol(add_transform!, :_pullback)
    @eval @is_primitive(
        DefaultCtx,
        ReverseMode,
        Tuple{
            typeof(TK.$add_transform!),
            AbstractTensorMap,
            AbstractTensorMap, Index2Tuple,
            Number, Number, Vararg{Any},
        }
    )

    @eval function Mooncake.rrule!!(
            ::CoDual{typeof(TK.$add_transform!)},
            C_ΔC::CoDual{<:AbstractTensorMap},
            A_ΔA::CoDual{<:AbstractTensorMap}, p_Δp::CoDual{<:Index2Tuple},
            α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
            ba_Δba::CoDual...
        )
        # prepare arguments
        C, ΔC = arrayify(C_ΔC)
        A, ΔA = arrayify(A_ΔA)
        p = primal(p_Δp)
        α, β = primal.((α_Δα, β_Δβ))
        ba = primal.(ba_Δba)

        C_cache = copy(C)

        # if we need to compute Δa, it is faster to allocate an intermediate permuted A
        # and store that instead of repeating the permutation in the pullback each time.
        # effectively, we replace `add_permute` by `add ∘ permute`.
        Ap = if _needs_tangent(α)
            Ap = $transform(A, p)
            add!(C, Ap, α, β)
            Ap
        else
            TK.$add_transform!(C, A, p, α, β, ba...)
            nothing
        end

        function $add_transform_pullback(::NoRData)
            copy!(C, C_cache)

            # ΔA
            ip = invperm(linearize(p))
            pΔA = _repartition(ip, A)
            TK.$add_transform!(ΔA, ΔC, pΔA, conj(α), One(), ba...)
            ΔAr = NoRData()

            Δαr = isnothing(Ap) ? NoRData() : project_scalar(α, inner(Ap, ΔC))
            Δβr = pullback_dβ(ΔC, C, β)
            ΔCr = pullback_dC!(ΔC, β) # this typically returns NoRData()

            return NoRData(), ΔCr, ΔAr, NoRData(), Δαr, Δβr, map(Returns(NoRData()), ba)...
        end

        return C_ΔC, $add_transform_pullback
    end
end

@is_primitive(
    DefaultCtx,
    ReverseMode,
    Tuple{
        typeof(TK.add_braid!),
        AbstractTensorMap,
        AbstractTensorMap, Index2Tuple, IndexTuple,
        Number, Number, Vararg{Any},
    }
)

function Mooncake.rrule!!(
        ::CoDual{typeof(TK.add_braid!)},
        C_ΔC::CoDual{<:AbstractTensorMap},
        A_ΔA::CoDual{<:AbstractTensorMap}, p_Δp::CoDual{<:Index2Tuple}, levels_Δlevels::CoDual{<:IndexTuple},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
        ba_Δba::CoDual...
    )
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    p = primal(p_Δp)
    levels = primal(levels_Δlevels)
    α, β = primal.((α_Δα, β_Δβ))
    ba = primal.(ba_Δba)

    C_cache = copy(C)

    # if we need to compute Δa, it is faster to allocate an intermediate braided A
    # and store that instead of repeating the permutation in the pullback each time.
    # effectively, we replace `add_permute` by `add ∘ permute`.
    Ap = if _needs_tangent(α)
        Ap = braid(A, p, levels)
        add!(C, Ap, α, β)
        Ap
    else
        TK.add_braid!(C, A, p, levels, α, β, ba...)
        nothing
    end

    function add_braid!_pullback(::NoRData)
        copy!(C, C_cache)

        # ΔA
        ip = invperm(linearize(p))
        pΔA = _repartition(ip, A)
        ilevels = TupleTools.permute(levels, linearize(p))
        TK.add_braid!(ΔA, ΔC, pΔA, ilevels, conj(α), One(), ba...)
        ΔAr = NoRData()

        Δαr = isnothing(Ap) ? NoRData() : project_scalar(α, inner(Ap, ΔC))
        Δβr = pullback_dβ(C, ΔC, β)
        ΔCr = pullback_dC!(ΔC, β) # this typically returns NoRData()

        return NoRData(), ΔCr, ΔAr, NoRData(), NoRData(), Δαr, Δβr, map(Returns(NoRData()), ba)...
    end

    return C_ΔC, add_braid!_pullback
end

# both are needed for correctly capturing every dispatch
@is_primitive DefaultCtx ReverseMode Tuple{typeof(twist!), AbstractTensorMap, Any}
@is_primitive DefaultCtx ReverseMode Tuple{typeof(Core.kwcall), @NamedTuple{inv::Bool}, typeof(twist!), AbstractTensorMap, Any}

function Mooncake.rrule!!(::CoDual{typeof(twist!)}, t_Δt::CoDual{<:AbstractTensorMap}, inds_Δinds::CoDual)
    # prepare arguments
    t, Δt = arrayify(t_Δt)
    inv = false
    inds = primal(inds_Δinds)

    # primal call
    t_cache = copy(t)
    twist!(t, inds; inv)

    function twist_pullback(::NoRData)
        copy!(t, t_cache)
        twist!(Δt, inds; inv = !inv)
        return ntuple(Returns(NoRData()), 3)
    end

    return t_Δt, twist_pullback

end
function Mooncake.rrule!!(
        ::CoDual{typeof(Core.kwcall)}, kwargs_Δkwargs::CoDual{@NamedTuple{inv::Bool}}, ::CoDual{typeof(twist!)},
        t_Δt::CoDual{<:AbstractTensorMap}, inds_Δinds::CoDual
    )
    # prepare arguments
    t, Δt = arrayify(t_Δt)
    inv = primal(kwargs_Δkwargs).inv
    inds = primal(inds_Δinds)

    # primal call
    t_cache = copy(t)
    twist!(t, inds; inv)

    function twist_pullback(::NoRData)
        copy!(t, t_cache)
        twist!(Δt, inds; inv = !inv)
        return ntuple(Returns(NoRData()), 5)
    end

    return t_Δt, twist_pullback
end

# both are needed for correctly capturing every dispatch
@is_primitive DefaultCtx ReverseMode Tuple{typeof(flip), AbstractTensorMap, Any}
@is_primitive DefaultCtx ReverseMode Tuple{typeof(Core.kwcall), @NamedTuple{inv::Bool}, typeof(flip), AbstractTensorMap, Any}

function Mooncake.rrule!!(::CoDual{typeof(flip)}, t_Δt::CoDual{<:AbstractTensorMap}, inds_Δinds::CoDual)
    # prepare arguments
    t, Δt = arrayify(t_Δt)
    inv = false
    inds = primal(inds_Δinds)

    # primal call
    t_flipped = flip(t, inds; inv)
    t_flipped_Δt_flipped = Mooncake.zero_fcodual(t_flipped)
    _, Δt_flipped = arrayify(t_flipped_Δt_flipped)

    function flip_pullback(::NoRData)
        Δt_flipflipped = flip(Δt_flipped, inds; inv = !inv)
        add!(Δt, scalartype(Δt) <: Real ? real(Δt_flipflipped) : Δt_flipflipped)
        return ntuple(Returns(NoRData()), 3)
    end

    return t_flipped_Δt_flipped, flip_pullback
end
function Mooncake.rrule!!(
        ::CoDual{typeof(Core.kwcall)}, kwargs_Δkwargs::CoDual{@NamedTuple{inv::Bool}}, ::CoDual{typeof(flip)},
        t_Δt::CoDual{<:AbstractTensorMap}, inds_Δinds::CoDual
    )
    # prepare arguments
    t, Δt = arrayify(t_Δt)
    inv = primal(kwargs_Δkwargs).inv
    inds = primal(inds_Δinds)

    # primal call
    t_flipped = flip(t, inds; inv)
    t_flipped_Δt_flipped = Mooncake.zero_fcodual(t_flipped)
    _, Δt_flipped = arrayify(t_flipped_Δt_flipped)

    function flip_pullback(::NoRData)
        Δt_flipflipped = flip(Δt_flipped, inds; inv = !inv)
        add!(Δt, scalartype(Δt) <: Real ? real(Δt_flipflipped) : Δt_flipflipped)
        return ntuple(Returns(NoRData()), 5)
    end

    return t_flipped_Δt_flipped, flip_pullback
end

for insertunit in (:insertleftunit, :insertrightunit)
    insertunit_pullback = Symbol(insertunit, :_pullback)
    @eval begin
        # both are needed for correctly capturing every dispatch
        @is_primitive DefaultCtx ReverseMode Tuple{typeof($insertunit), AbstractTensorMap, Val}
        @is_primitive DefaultCtx ReverseMode Tuple{typeof(Core.kwcall), NamedTuple, typeof($insertunit), AbstractTensorMap, Val}

        function Mooncake.rrule!!(::CoDual{typeof($insertunit)}, tsrc_Δtsrc::CoDual{<:AbstractTensorMap}, ival_Δival::CoDual{<:Val})
            # prepare arguments
            tsrc, Δtsrc = arrayify(tsrc_Δtsrc)
            ival = primal(ival_Δival)

            # tdst shares data with tsrc if <:TensorMap, in this case we have to deal with correctly
            # sharing address spaces
            if tsrc isa TensorMap
                tsrc_cache = copy(tsrc)
                tdst_Δtdst = CoDual(
                    $insertunit(tsrc, ival),
                    $insertunit(Mooncake.tangent(tsrc_Δtsrc), ival)
                )
            else
                tsrc_cache = nothing
                tdst = $insertunit(tsrc, ival)
                tdst_Δtdst = Mooncake.zero_fcodual(tdst)
            end

            _, Δtdst = arrayify(tdst_Δtdst)

            function $insertunit_pullback(::NoRData)
                if isnothing(tsrc_cache)
                    for (c, b) in blocks(Δtdst)
                        add!(block(Δtsrc, c), b)
                    end
                else
                    copy!(tsrc, tsrc_cache)
                    # note: since data is already shared, don't have to do anything here!
                end
                return ntuple(Returns(NoRData()), 3)
            end

            return tdst_Δtdst, $insertunit_pullback
        end
        function Mooncake.rrule!!(
                ::CoDual{typeof(Core.kwcall)}, kwargs_Δkwargs::CoDual{<:NamedTuple},
                ::CoDual{typeof($insertunit)}, tsrc_Δtsrc::CoDual{<:AbstractTensorMap}, ival_Δival::CoDual{<:Val}
            )
            # prepare arguments
            tsrc, Δtsrc = arrayify(tsrc_Δtsrc)
            ival = primal(ival_Δival)
            kwargs = primal(kwargs_Δkwargs)

            # tdst shares data with tsrc if <:TensorMap & copy=false, in this case we have to deal with correctly
            # sharing address spaces
            if tsrc isa TensorMap && !get(kwargs, :copy, false)
                tsrc_cache = copy(tsrc)
                tdst = $insertunit(tsrc, ival; kwargs...)
                tdst_Δtdst = CoDual(
                    $insertunit(tsrc, ival; kwargs...),
                    $insertunit(Mooncake.tangent(tsrc_Δtsrc), ival; kwargs...)
                )
            else
                tsrc_cache = nothing
                tdst = $insertunit(tsrc, ival; kwargs...)
                tdst_Δtdst = Mooncake.zero_fcodual(tdst)
            end

            _, Δtdst = arrayify(tdst_Δtdst)

            function $insertunit_pullback(::NoRData)
                if isnothing(tsrc_cache)
                    for (c, b) in blocks(Δtdst)
                        add!(block(Δtsrc, c), b)
                    end
                else
                    copy!(tsrc, tsrc_cache)
                    # note: since data is already shared, don't have to do anything here!
                end
                return ntuple(Returns(NoRData()), 5)
            end

            return tdst_Δtdst, $insertunit_pullback
        end
    end
end


@is_primitive DefaultCtx ReverseMode Tuple{typeof(removeunit), AbstractTensorMap, Val}
@is_primitive DefaultCtx ReverseMode Tuple{typeof(Core.kwcall), NamedTuple, typeof(removeunit), AbstractTensorMap, Val}

function Mooncake.rrule!!(::CoDual{typeof(removeunit)}, tsrc_Δtsrc::CoDual{<:AbstractTensorMap}, ival_Δival::CoDual{Val{i}}) where {i}
    # prepare arguments
    tsrc, Δtsrc = arrayify(tsrc_Δtsrc)
    ival = primal(ival_Δival)

    # tdst shares data with tsrc if <:TensorMap, in this case we have to deal with correctly
    # sharing address spaces
    if tsrc isa TensorMap
        tsrc_cache = copy(tsrc)
        tdst_Δtdst = CoDual(
            removeunit(tsrc, ival),
            removeunit(Mooncake.tangent(tsrc_Δtsrc), ival)
        )
    else
        tsrc_cache = nothing
        tdst = removeunit(tsrc, ival)
        tdst_Δtdst = Mooncake.zero_fcodual(tdst)
    end

    _, Δtdst = arrayify(tdst_Δtdst)

    function removeunit_pullback(::NoRData)
        if isnothing(tsrc_cache)
            for (c, b) in blocks(Δtdst)
                add!(block(Δtsrc, c), b)
            end
        else
            copy!(tsrc, tsrc_cache)
            # note: since data is already shared, don't have to do anything here!
        end
        return ntuple(Returns(NoRData()), 3)
    end

    return tdst_Δtdst, removeunit_pullback
end
function Mooncake.rrule!!(
        ::CoDual{typeof(Core.kwcall)}, kwargs_Δkwargs::CoDual{<:NamedTuple},
        ::CoDual{typeof(removeunit)}, tsrc_Δtsrc::CoDual{<:AbstractTensorMap}, ival_Δival::CoDual{<:Val}
    )
    # prepare arguments
    tsrc, Δtsrc = arrayify(tsrc_Δtsrc)
    ival = primal(ival_Δival)
    kwargs = primal(kwargs_Δkwargs)

    # tdst shares data with tsrc if <:TensorMap & copy=false, in this case we have to deal with correctly
    # sharing address spaces
    if tsrc isa TensorMap && !get(kwargs, :copy, false)
        tsrc_cache = copy(tsrc)
        tdst_Δtdst = CoDual(
            removeunit(tsrc, ival; kwargs...),
            removeunit(Mooncake.tangent(tsrc_Δtsrc), ival)
        )
    else
        tsrc_cache = nothing
        tdst = removeunit(tsrc, ival; kwargs...)
        tdst_Δtdst = Mooncake.zero_fcodual(tdst)
    end

    _, Δtdst = arrayify(tdst_Δtdst)

    function removeunit_pullback(::NoRData)
        if isnothing(tsrc_cache)
            for (c, b) in blocks(Δtdst)
                add!(block(Δtsrc, c), b)
            end
        else
            copy!(tsrc, tsrc_cache)
            # note: since data is already shared, don't have to do anything here!
        end
        return ntuple(Returns(NoRData()), 5)
    end

    return tdst_Δtdst, removeunit_pullback
end
