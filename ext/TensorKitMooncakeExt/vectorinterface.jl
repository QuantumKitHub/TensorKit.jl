@is_primitive DefaultCtx Tuple{typeof(scale!), AbstractTensorMap, Number}

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

function Mooncake.frule!!(::Dual{typeof(scale!)}, C_ΔC::Dual{<:AbstractTensorMap}, α_Δα::Dual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    α, Δα = Mooncake.extract(α_Δα)

    if !isa(Δα, Mooncake.NoTangent)
        add!(ΔC, C, Δα, α)
    else
        scale!(ΔC, α)
    end
    scale!(C, α)

    return C_ΔC
end

@is_primitive DefaultCtx Tuple{typeof(scale!), AbstractTensorMap, AbstractTensorMap, Number}

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

function Mooncake.frule!!(::Dual{typeof(scale!)}, C_ΔC::Dual{<:AbstractTensorMap}, A_ΔA::Dual{<:AbstractTensorMap}, α_Δα::Dual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α, Δα = Mooncake.extract(α_Δα)

    scale!(ΔC, ΔA, α)
    if !isa(Δα, Mooncake.NoTangent)
        add!(ΔC, A, Δα, One())
    end
    scale!(C, A, α)
    return C_ΔC
end

@is_primitive DefaultCtx Tuple{typeof(add!), AbstractTensorMap, AbstractTensorMap, Number, Number}

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

function Mooncake.frule!!(::Dual{typeof(add!)}, C_ΔC::Dual{<:AbstractTensorMap}, A_ΔA::Dual{<:AbstractTensorMap}, α_Δα::Dual{<:Number}, β_Δβ::Dual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α, Δα = Mooncake.extract(α_Δα)
    β, Δβ = Mooncake.extract(β_Δβ)
    add!(ΔC, ΔA, α, β)
    if isa(Δβ, Mooncake.NoTangent) && !isa(Δα, Mooncake.NoTangent)
        add!(ΔC, A, Δα, One())
    elseif isa(Δα, Mooncake.NoTangent) && !isa(Δβ, Mooncake.NoTangent)
        add!(ΔC, C, Δβ, One())
    elseif !isa(Δα, Mooncake.NoTangent) && !isa(Δβ, Mooncake.NoTangent)
        add!(ΔC, A, Δα, One())
        add!(ΔC, C, Δβ, One())
    end
    add!(C, A, α, β)
    return C_ΔC
end

@is_primitive DefaultCtx Tuple{typeof(inner), AbstractTensorMap, AbstractTensorMap}

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

function Mooncake.frule!!(::Dual{typeof(inner)}, A_ΔA::Dual{<:AbstractTensorMap}, B_ΔB::Dual{<:AbstractTensorMap})
    # prepare arguments
    A, ΔA = arrayify(A_ΔA)
    B, ΔB = arrayify(B_ΔB)

    s = inner(A, B)
    Δs = inner(A, ΔB) + inner(ΔA, B)

    return Dual(s, Δs)
end
