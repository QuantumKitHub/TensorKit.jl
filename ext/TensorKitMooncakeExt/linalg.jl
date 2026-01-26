# Shared
# ------
pullback_dC!(ΔC, β) = (scale!(ΔC, conj(β)); return NoRData())
pullback_dβ(C, ΔC, β) = _needs_tangent(β) ? inner(C, ΔC) : NoRData()

@is_primitive DefaultCtx ReverseMode Tuple{typeof(mul!), AbstractTensorMap, AbstractTensorMap, AbstractTensorMap, Number, Number}

function Mooncake.rrule!!(
        ::CoDual{typeof(mul!)},
        C_ΔC::CoDual{<:AbstractTensorMap}, A_ΔA::CoDual{<:AbstractTensorMap}, B_ΔB::CoDual{<:AbstractTensorMap},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number}
    )
    (C, ΔC), (A, ΔA), (B, ΔB) = arrayify.((C_ΔC, A_ΔA, B_ΔB))
    α, β = primal.((α_Δα, β_Δβ))

    # primal call
    C_cache = copy(C)
    AB = if _needs_tangent(α)
        AB = A * B
        add!(C, AB, α, β)
        AB
    else
        mul!(C, A, B, α, β)
        nothing
    end

    function mul_pullback(::NoRData)
        copy!(C, C_cache)

        mul!(ΔA, ΔC, B', conj(α), One())
        mul!(ΔB, A', ΔC, conj(α), One())
        ΔAr = NoRData()
        ΔBr = NoRData()
        Δαr = isnothing(AB) ? NoRData() : inner(AB, ΔC)
        Δβr = pullback_dβ(C, ΔC, β)
        ΔCr = pullback_dC!(ΔC, β)

        return NoRData(), ΔCr, ΔAr, ΔBr, Δαr, Δβr
    end

    return C_ΔC, mul_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(norm), AbstractTensorMap, Real}

function Mooncake.rrule!!(::CoDual{typeof(norm)}, tΔt::CoDual{<:AbstractTensorMap}, pdp::CoDual{<:Real})
    t, Δt = arrayify(tΔt)
    p = primal(pdp)
    p == 2 || error("currently only implemented for p = 2")
    n = norm(t, p)
    function norm_pullback(Δn)
        x = (Δn' + Δn) / 2 / hypot(n, eps(one(n)))
        add!(Δt, t, x)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(n, Mooncake.NoFData()), norm_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(tr), AbstractTensorMap}

function Mooncake.rrule!!(::CoDual{typeof(tr)}, A_ΔA::CoDual{<:AbstractTensorMap})
    A, ΔA = arrayify(A_ΔA)
    trace = tr(A)

    function tr_pullback(Δtrace)
        for (_, b) in blocks(ΔA)
            TensorKit.diagview(b) .+= Δtrace
        end
        return NoRData(), NoRData()
    end

    return CoDual(trace, Mooncake.NoFData()), tr_pullback
end

@is_primitive DefaultCtx ReverseMode Tuple{typeof(inv), AbstractTensorMap}

function Mooncake.rrule!!(::CoDual{typeof(inv)}, A_ΔA::CoDual{<:AbstractTensorMap})
    A, ΔA = arrayify(A_ΔA)
    Ainv_ΔAinv = Mooncake.zero_fcodual(inv(A))
    Ainv, ΔAinv = arrayify(Ainv_ΔAinv)

    function inv_pullback(::NoRData)
        mul!(ΔA, Ainv' * ΔAinv, Ainv', -1, One())
        return NoRData(), NoRData()
    end

    return Ainv_ΔAinv, inv_pullback
end
