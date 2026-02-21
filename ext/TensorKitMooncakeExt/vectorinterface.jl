@is_primitive DefaultCtx ReverseMode Tuple{typeof(scale!), AbstractTensorMap, Number}

function Mooncake.rrule!!(::CoDual{typeof(scale!)}, C_ΔC::CoDual{<:AbstractTensorMap}, α_Δα::CoDual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    α = primal(α_Δα)

    # primal call
    C_cache = copy(C)
    scale!(C, α)

    function scale_pullback(::NoRData)
        copy!(C, C_cache)
        Δαr = _needs_tangent(α) ? project_scalar(α, inner(C, ΔC)) : NoRData()
        scale!(ΔC, conj(α))
        return NoRData(), NoRData(), Δαr
    end

    return C_ΔC, scale_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(scale!), AbstractTensorMap, AbstractTensorMap, Number}

function Mooncake.rrule!!(::CoDual{typeof(scale!)}, C_ΔC::CoDual{<:AbstractTensorMap}, A_ΔA::CoDual{<:AbstractTensorMap}, α_Δα::CoDual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α = primal(α_Δα)

    # primal call
    C_cache = copy(C)
    scale!(C, A, α)

    function scale_pullback(::NoRData)
        copy!(C, C_cache)
        add!(ΔA, ΔC, conj(α))
        Δαr = _needs_tangent(α) ? project_scalar(α, inner(A, ΔC)) : NoRData()
        zerovector!(ΔC)
        return NoRData(), NoRData(), NoRData(), Δαr
    end

    return C_ΔC, scale_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(add!), AbstractTensorMap, AbstractTensorMap, Number, Number}

function Mooncake.rrule!!(::CoDual{typeof(add!)}, C_ΔC::CoDual{<:AbstractTensorMap}, A_ΔA::CoDual{<:AbstractTensorMap}, α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α = primal(α_Δα)
    β = primal(β_Δβ)

    # primal call
    C_cache = copy(C)
    add!(C, A, α, β)

    function add_pullback(::NoRData)
        copy!(C, C_cache)

        Δαr = _needs_tangent(α) ? project_scalar(α, inner(A, ΔC)) : NoRData()
        Δβr = _needs_tangent(β) ? project_scalar(β, inner(C, ΔC)) : NoRData()
        add!(ΔA, ΔC, conj(α))
        scale!(ΔC, conj(β))

        return NoRData(), NoRData(), NoRData(), Δαr, Δβr
    end

    return C_ΔC, add_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(inner), AbstractTensorMap, AbstractTensorMap}

function Mooncake.rrule!!(::CoDual{typeof(inner)}, A_ΔA::CoDual{<:AbstractTensorMap}, B_ΔB::CoDual{<:AbstractTensorMap})
    # prepare arguments
    A, ΔA = arrayify(A_ΔA)
    B, ΔB = arrayify(B_ΔB)

    # primal call
    s = inner(A, B)

    function inner_pullback(Δs)
        add!(ΔA, B, conj(Δs))
        add!(ΔB, A, Δs)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(s, NoFData()), inner_pullback
end
