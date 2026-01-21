# tensoradd!
# ----------
# Mooncake.@is_primitive(
#     DefaultCtx,
#     ReverseMode,
#     Tuple{
#         typeof(TO.tensoradd!),
#         AbstractTensorMap,
#         AbstractTensorMap, Index2Tuple, Bool,
#         Number, Number, Vararg{Any},
#     }
# )
#
# function Mooncake.rrule!!(
#         ::CoDual{typeof(TO.tensoradd!)},
#         C_ΔC::CoDual{<:AbstractTensorMap},
#         A_ΔA::CoDual{<:AbstractTensorMap}, pA_ΔpA::CoDual{<:Index2Tuple}, conjA_ΔconjA::CoDual{Bool},
#         α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
#         ba_Δba::CoDual...
#     )
#     # prepare arguments
#     C, ΔC = arrayify(C_ΔC)
#     A, ΔA = arrayify(A_ΔA)
#     pA = primal(pA_ΔpA)
#     conjA = primal(conjA_ΔconjA)
#     α, β = primal.((α_Δα, β_Δβ))
#     ba = primal.(ba_Δba)
#
#     # primal call
#     C_cache = copy(C)
#     TO.tensoradd!(C, A, pA, conjA, α, β, ba...)
#
#     function tensoradd_pullback(::NoRData)
#         copy!(C, C_cache)
#
#         ΔCr = tensoradd_pullback_ΔC!(ΔC, β)
#         ΔAr = tensoradd_pullback_ΔA!(ΔA, ΔC, A, pA, conjA, α, ba...)
#         Δαr = tensoradd_pullback_Δα(ΔC, A, pA, conjA, α, ba...)
#         Δβr = tensoradd_pullback_Δβ(ΔC, C, β)
#
#         return NoRData(),
#             ΔCr,
#             ΔAr, NoRData(), NoRData(),
#             Δαr, Δβr,
#             map(Returns(NoRData()), ba)...
#     end
#
#     return C_ΔC, tensoradd_pullback
# end
#
# tensoradd_pullback_ΔC!(ΔC, β) = (scale!(ΔC, conj(β)); NoRData())
#
# function tensoradd_pullback_ΔA!(
#         ΔA, ΔC, A, pA, conjA, α, ba...
#     )
#     ipA = invperm(linearize(pA))
#     pΔA = _repartition(ipA, A)
#     TO.tensoradd!(ΔA, ΔC, pΔA, conjA, conjA ? α : conj(α), Zero(), ba...)
#     return NoRData()
# end
#
# function tensoradd_pullback_Δα(
#         ΔC, A, pA, conjA, α, ba...
#     )
#     Tdα = Mooncake.rdata_type(Mooncake.tangent_type(typeof(α)))
#     Tdα === NoRData && return NoRData()
#
#     tΔC = twist(ΔC, filter(x -> isdual(space(ΔC, x)), allind(ΔC)); copy = false)
#     Δα = TO.tensorscalar(
#         TO.tensorcontract(
#             A, ((), linearize(pA)), !conjA,
#             tΔC, (trivtuple(TO.numind(pA)), ()), false,
#             ((), ()), One(), ba...
#         )
#     )
#     return Mooncake._rdata(Δα)
# end
#
# function tensoradd_pullback_Δβ(ΔC, C, β)
#     Tdβ = Mooncake.rdata_type(Mooncake.tangent_type(typeof(β)))
#     Tdβ === NoRData && return NoRData()
#
#     Δβ = inner(C, ΔC)
#     return Mooncake._rdata(Δβ)
# end

# tensorcontract!
# ---------------
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

# tensortrace!
# ------------
Mooncake.@is_primitive(
    DefaultCtx,
    ReverseMode,
    Tuple{
        typeof(TensorKit.trace_permute!),
        AbstractTensorMap,
        AbstractTensorMap, Index2Tuple, Index2Tuple,
        Number, Number,
        Any,
    }
)

function Mooncake.rrule!!(
        ::CoDual{typeof(TensorKit.trace_permute!)},
        C_ΔC::CoDual{<:AbstractTensorMap},
        A_ΔA::CoDual{<:AbstractTensorMap}, p_Δp::CoDual{<:Index2Tuple}, q_Δq::CoDual{<:Index2Tuple},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
        backend_Δbackend::CoDual
    )
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    p = primal(p_Δp)
    q = primal(q_Δq)
    α, β = primal.((α_Δα, β_Δβ))
    backend = primal(backend_Δbackend)

    # primal call
    C_cache = copy(C)
    TensorKit.trace_permute!(C, A, p, q, α, β, backend)

    function trace_permute_pullback(::NoRData)
        copy!(C, C_cache)

        ΔAr = trace_permute_pullback_ΔA!(ΔA, ΔC, A, p, q, α, backend)
        Δαr = trace_permute_pullback_Δα(ΔC, A, p, q, α, backend)
        Δβr = trace_permute_pullback_Δβ(ΔC, C, β)
        ΔCr = trace_permute_pullback_ΔC!(ΔC, β)

        return NoRData(),
            ΔCr, ΔAr, NoRData(), NoRData(),
            Δαr, Δβr, NoRData()
    end

    return C_ΔC, trace_permute_pullback
end

trace_permute_pullback_ΔC!(ΔC, β) = (scale!(ΔC, conj(β)); NoRData())

function trace_permute_pullback_ΔA!(
        ΔA, ΔC, A, p, q, α, backend
    )
    ip = invperm((linearize(p)..., q[1]..., q[2]...))
    pdA = _repartition(ip, A)
    E = one!(TO.tensoralloc_add(scalartype(A), A, q, false))
    twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
    pE = ((), trivtuple(TO.numind(q)))
    pΔC = (trivtuple(TO.numind(p)), ())
    TO.tensorproduct!(
        ΔA, ΔC, pΔC, false, E, pE, false, pdA, conj(α), One(), backend
    )
    return NoRData()
end

function trace_permute_pullback_Δα(
        ΔC, A, p, q, α, backend
    )
    Tdα = Mooncake.rdata_type(Mooncake.tangent_type(typeof(α)))
    Tdα === NoRData && return NoRData()

    # TODO: this result might be easier to compute as:
    # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
    At = TO.tensortrace(A, p, q, false, One(), backend)
    Δα = inner(At, ΔC)
    return Mooncake._rdata(Δα)
end

function trace_permute_pullback_Δβ(ΔC, C, β)
    Tdβ = Mooncake.rdata_type(Mooncake.tangent_type(typeof(β)))
    Tdβ === NoRData && return NoRData()

    Δβ = inner(C, ΔC)
    return Mooncake._rdata(Δβ)
end
