Mooncake.@is_primitive(
    DefaultCtx,
    ReverseMode,
    Tuple{
        typeof(TO.tensorcontract!),
        AbstractTensorMap,
        AbstractTensorMap, Index2Tuple, Bool,
        AbstractTensorMap, Index2Tuple, Bool,
        Index2Tuple,
        Number, Number,
        Vararg{Any},
    }
)

function Mooncake.rrule!!(
        ::CoDual{typeof(TO.tensorcontract!)},
        C_ΔC::CoDual{<:AbstractTensorMap},
        A_ΔA::CoDual{<:AbstractTensorMap}, pA_ΔpA::CoDual{<:Index2Tuple}, conjA_ΔconjA::CoDual{Bool},
        B_ΔB::CoDual{<:AbstractTensorMap}, pB_ΔpB::CoDual{<:Index2Tuple}, conjB_ΔconjB::CoDual{Bool},
        pAB_ΔpAB::CoDual{<:Index2Tuple},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
        ba_Δba::CoDual...,
    )
    # prepare arguments
    (C, ΔC), (A, ΔA), (B, ΔB) = arrayify.((C_ΔC, A_ΔA, B_ΔB))
    pA, pB, pAB = primal.((pA_ΔpA, pB_ΔpB, pAB_ΔpAB))
    conjA, conjB = primal.((conjA_ΔconjA, conjB_ΔconjB))
    α, β = primal.((α_Δα, β_Δβ))
    ba = primal.(ba_Δba)

    # primal call
    C_cache = copy(C)
    TO.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

    function tensorcontract_pullback(::NoRData)
        copy!(C, C_cache)

        ΔCr = tensorcontract_pullback_ΔC!(ΔC, β)
        ΔAr = tensorcontract_pullback_ΔA!(
            ΔA, ΔC, A, pA, conjA, B, pB, conjB, pAB, α, ba...
        )
        ΔBr = tensorcontract_pullback_ΔB!(
            ΔB, ΔC, A, pA, conjA, B, pB, conjB, pAB, α, ba...
        )
        Δαr = tensorcontract_pullback_Δα(
            ΔC, A, pA, conjA, B, pB, conjB, pAB, α, ba...
        )
        Δβr = tensorcontract_pullback_Δβ(ΔC, C, β)

        return NoRData(), ΔCr,
            ΔAr, NoRData(), NoRData(),
            ΔBr, NoRData(), NoRData(),
            NoRData(),
            Δαr, Δβr,
            map(ba_ -> NoRData(), ba)...
    end

    return C_ΔC, tensorcontract_pullback
end

tensorcontract_pullback_ΔC!(ΔC, β) = (scale!(ΔC, conj(β)); NoRData())

function tensorcontract_pullback_ΔA!(
        ΔA, ΔC, A, pA, conjA, B, pB, conjB, pAB, α, ba...
    )
    ipAB = invperm(linearize(pAB))
    pΔC = _repartition(ipAB, TO.numout(pA))
    ipA = _repartition(invperm(linearize(pA)), A)
    conjΔC = conjA
    conjB′ = conjA ? conjB : !conjB

    tB = twist(
        B,
        TupleTools.vcat(
            filter(x -> !isdual(space(B, x)), pB[1]),
            filter(x -> isdual(space(B, x)), pB[2])
        ); copy = false
    )

    TO.tensorcontract!(
        ΔA,
        ΔC, pΔC, conjΔC,
        tB, reverse(pB), conjB′,
        ipA,
        conjA ? α : conj(α), Zero(),
        ba...
    )

    return NoRData()
end

function tensorcontract_pullback_ΔB!(
        ΔB, ΔC, A, pA, conjA, B, pB, conjB, pAB, α, ba...
    )
    ipAB = invperm(linearize(pAB))
    pΔC = _repartition(ipAB, TO.numout(pA))
    ipB = _repartition(invperm(linearize(pB)), B)
    conjΔC = conjB
    conjA′ = conjB ? conjA : !conjA

    tA = twist(
        A,
        TupleTools.vcat(
            filter(x -> isdual(space(A, x)), pA[1]),
            filter(x -> !isdual(space(A, x)), pA[2])
        ); copy = false
    )

    TO.tensorcontract!(
        ΔB,
        tA, reverse(pA), conjA′,
        ΔC, pΔC, conjΔC,
        ipB,
        conjB ? α : conj(α), Zero(), ba...
    )

    return NoRData()
end

function tensorcontract_pullback_Δα(
        ΔC, A, pA, conjA, B, pB, conjB, pAB, α, ba...
    )
    Tdα = Mooncake.rdata_type(Mooncake.tangent_type(typeof(α)))
    Tdα === NoRData && return NoRData()

    AB = TO.tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
    Δα = inner(AB, ΔC)
    return Mooncake._rdata(Δα)
end

function tensorcontract_pullback_Δβ(ΔC, C, β)
    Tdβ = Mooncake.rdata_type(Mooncake.tangent_type(typeof(β)))
    Tdβ === NoRData && return NoRData()

    Δβ = inner(C, ΔC)
    return Mooncake._rdata(Δβ)
end
