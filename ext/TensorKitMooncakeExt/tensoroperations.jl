# tensorcontract!
# ---------------
# TODO: it might be beneficial to compare here if it would make sense to simply compute the
# rrule of permute-permute-gemm-permute, rather than using the contractions directly.
# This could possibly out save some permutations being carried out twice, at the cost of having
# to store some more intermediate objects.
# For example, the combination `ΔC, pΔC, false` appears in the pullback for ΔA and ΔB, so effectively
# this permutation is done multiple times.
@is_primitive(
    DefaultCtx,
    ReverseMode,
    Tuple{
        typeof(TensorKit.blas_contract!),
        AbstractTensorMap,
        AbstractTensorMap, Index2Tuple,
        AbstractTensorMap, Index2Tuple,
        Index2Tuple,
        Number, Number,
        Any, Any,
    }
)

function Mooncake.rrule!!(
        ::CoDual{typeof(TensorKit.blas_contract!)},
        C_ΔC::CoDual{<:AbstractTensorMap},
        A_ΔA::CoDual{<:AbstractTensorMap}, pA_ΔpA::CoDual{<:Index2Tuple},
        B_ΔB::CoDual{<:AbstractTensorMap}, pB_ΔpB::CoDual{<:Index2Tuple},
        pAB_ΔpAB::CoDual{<:Index2Tuple},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
        backend_Δbackend::CoDual, allocator_Δallocator::CoDual
    )
    # prepare arguments
    (C, ΔC), (A, ΔA), (B, ΔB) = arrayify.((C_ΔC, A_ΔA, B_ΔB))
    pA, pB, pAB = primal.((pA_ΔpA, pB_ΔpB, pAB_ΔpAB))
    α, β = primal.((α_Δα, β_Δβ))
    backend, allocator = primal.((backend_Δbackend, allocator_Δallocator))

    # primal call
    C_cache = copy(C)
    AB = if _needs_tangent(α)
        AB = TO.tensorcontract(A, pA, false, B, pB, false, pAB, One(), backend, allocator)
        add!(C, AB, α, β)
        AB
    else
        TensorKit.blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
        nothing
    end

    function blas_contract_pullback(::NoRData)
        copy!(C, C_cache)

        ΔAr = blas_contract_pullback_ΔA!(
            ΔA, ΔC, A, pA, B, pB, pAB, α, backend, allocator
        ) # this typically returns NoRData()
        ΔBr = blas_contract_pullback_ΔB!(
            ΔB, ΔC, A, pA, B, pB, pAB, α, backend, allocator
        ) # this typically returns NoRData()
        Δαr = isnothing(AB) ? NoRData() : project_scalar(α, inner(AB, ΔC))
        Δβr = pullback_dβ(ΔC, C, β)
        ΔCr = pullback_dC!(ΔC, β) # this typically returns NoRData()

        return NoRData(), ΔCr,
            ΔAr, NoRData(),
            ΔBr, NoRData(),
            NoRData(),
            Δαr, Δβr,
            NoRData(), NoRData()
    end

    return C_ΔC, blas_contract_pullback
end

function blas_contract_pullback_ΔA!(
        ΔA, ΔC, A, pA, B, pB, pAB, α, backend, allocator
    )
    ipAB = invperm(linearize(pAB))
    pΔC = _repartition(ipAB, TO.numout(pA))
    ipA = _repartition(invperm(linearize(pA)), A)

    tB = twist(
        B,
        TupleTools.vcat(
            filter(x -> !isdual(space(B, x)), pB[1]),
            filter(x -> isdual(space(B, x)), pB[2])
        ); copy = false
    )

    project_contract!(
        ΔA,
        ΔC, pΔC, false,
        tB, reverse(pB), true,
        ipA, conj(α), backend, allocator
    )

    return NoRData()
end

function blas_contract_pullback_ΔB!(
        ΔB, ΔC, A, pA, B, pB, pAB, α, backend, allocator
    )
    ipAB = invperm(linearize(pAB))
    pΔC = _repartition(ipAB, TO.numout(pA))
    ipB = _repartition(invperm(linearize(pB)), B)

    tA = twist(
        A,
        TupleTools.vcat(
            filter(x -> isdual(space(A, x)), pA[1]),
            filter(x -> !isdual(space(A, x)), pA[2])
        ); copy = false
    )

    project_contract!(
        ΔB,
        tA, reverse(pA), true,
        ΔC, pΔC, false,
        ipB, conj(α), backend, allocator
    )

    return NoRData()
end

# tensortrace!
# ------------
@is_primitive(
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
    At = if _needs_tangent(α)
        At = TO.tensortrace(A, p, q, false, One(), backend)
        add!(C, A, α, β)
        At
    else
        TensorKit.trace_permute!(C, A, p, q, α, β, backend)
        nothing
    end

    function trace_permute_pullback(::NoRData)
        copy!(C, C_cache)

        ΔAr = trace_permute_pullback_ΔA!(ΔA, ΔC, A, p, q, α, backend) # this typically returns NoRData()

        Δαr = isnothing(At) ? NoRData() : project_scalar(α, inner(At, ΔC))
        Δβr = pullback_dβ(ΔC, C, β)
        ΔCr = pullback_dC!(ΔC, β) # this typically returns NoRData()

        return NoRData(),
            ΔCr, ΔAr, NoRData(), NoRData(),
            Δαr, Δβr, NoRData()
    end

    return C_ΔC, trace_permute_pullback
end

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
