for transform in (:permute, :transpose)
    add_transform! = Symbol(:add_, transform, :!)
    add_transform_pullback = Symbol(add_transform!, :_pullback)
    @eval Mooncake.@is_primitive(
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
        Tdα = Mooncake.rdata_type(Mooncake.tangent_type(typeof(α)))
        Ap = if Tdα === NoRData
            TK.$add_transform!(C, A, p, α, β, ba...)
            nothing
        else
            Ap = $transform(A, p)
            add!(C, Ap, α, β)
            Ap
        end

        function $add_transform_pullback(::NoRData)
            copy!(C, C_cache)

            scale!(ΔC, conj(β))
            ΔCr = NoRData()

            # ΔA
            ip = invperm(linearize(p))
            pΔA = _repartition(ip, A)
            TK.$add_transform!(ΔA, ΔC, pΔA, conj(α), One(), ba...)
            ΔAr = NoRData()

            # Δα
            Δαr = if isnothing(Ap)
                NoRData()
            else
                Mooncake._rdata(inner(Ap, ΔC))
            end

            # Δβ
            Tdβ = Mooncake.rdata_type(Mooncake.tangent_type(typeof(β)))
            Δβr = if Tdβ === NoRData
                NoRData()
            else
                Mooncake._rdata(inner(C, ΔC))
            end


            return NoRData(), ΔCr, ΔAr, NoRData(), Δαr, Δβr, map(Returns(NoRData()), ba)...
        end

        return C_ΔC, $add_transform_pullback
    end
end

Mooncake.@is_primitive(
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
    Tdα = Mooncake.rdata_type(Mooncake.tangent_type(typeof(α)))
    Ap = if Tdα === NoRData
        TK.add_braid!(C, A, p, levels, α, β, ba...)
        nothing
    else
        Ap = braid(A, p, levels)
        add!(C, Ap, α, β)
        Ap
    end

    function add_braid!_pullback(::NoRData)
        copy!(C, C_cache)

        scale!(ΔC, conj(β))
        ΔCr = NoRData()

        # ΔA
        ip = invperm(linearize(p))
        pΔA = _repartition(ip, A)
        ilevels = TupleTools.permute(levels, linearize(p))
        TK.add_braid!(ΔA, ΔC, pΔA, ilevels, conj(α), One(), ba...)
        ΔAr = NoRData()

        # Δα
        Δαr = if isnothing(Ap)
            NoRData()
        else
            Mooncake._rdata(inner(Ap, ΔC))
        end

        # Δβ
        Tdβ = Mooncake.rdata_type(Mooncake.tangent_type(typeof(β)))
        Δβr = if Tdβ === NoRData
            NoRData()
        else
            Mooncake._rdata(inner(C, ΔC))
        end


        return NoRData(), ΔCr, ΔAr, NoRData(), NoRData(), Δαr, Δβr, map(Returns(NoRData()), ba)...
    end

    return C_ΔC, add_braid!_pullback
end

# both are needed for correctly capturing every dispatch
Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(twist!), AbstractTensorMap, Any}
Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(Core.kwcall), @NamedTuple{inv::Bool}, typeof(twist!), AbstractTensorMap, Any}

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
