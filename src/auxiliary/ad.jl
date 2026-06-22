# in-place multiplication and accumulation which might project to (real)
# TODO: this could probably be done without allocating
function project_mul!(C, A, B, α, β = One())
    TC = TO.promote_contract(scalartype(A), scalartype(B), scalartype(α))
    return if !(TC <: Real) && scalartype(C) <: Real
        add!(C, real(mul!(zerovector(C, TC), A, B, α)))
    else
        mul!(C, A, B, α, β)
    end
end
function project_contract!(C, A, pA, conjA, B, pB, conjB, pAB, α, backend, allocator)
    TA = TensorKit.promote_permute(A)
    TB = TensorKit.promote_permute(B)
    TC = TO.promote_contract(TA, TB, scalartype(α))

    return if scalartype(C) <: Real && !(TC <: Real)
        add!(C, real(TO.tensorcontract!(zerovector(C, TC), A, pA, conjA, B, pB, conjB, pAB, α, Zero(), backend, allocator)))
    else
        TO.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, One(), backend, allocator)
    end
end

_pullback_dα(α, ΔC, A) = TO.project_scalar(α, inner(A, ΔC))
_pullback_dβ(β, ΔC, C) = TO.project_scalar(β, inner(C, ΔC))
