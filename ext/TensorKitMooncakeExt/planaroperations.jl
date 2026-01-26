# planartrace!
# ------------
@is_primitive(
    DefaultCtx,
    ReverseMode,
    Tuple{
        typeof(TensorKit.planartrace!),
        AbstractTensorMap,
        AbstractTensorMap, Index2Tuple, Index2Tuple,
        Number, Number,
        Any, Any,
    }
)

function Mooncake.rrule!!(
        ::CoDual{typeof(TensorKit.planartrace!)},
        C_ΔC::CoDual{<:AbstractTensorMap},
        A_ΔA::CoDual{<:AbstractTensorMap}, p_Δp::CoDual{<:Index2Tuple}, q_Δq::CoDual{<:Index2Tuple},
        α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number},
        backend_Δbackend::CoDual, allocator_Δallocator::CoDual
    )
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    p = primal(p_Δp)
    q = primal(q_Δq)
    α, β = primal.((α_Δα, β_Δβ))
    backend, allocator = primal.((backend_Δbackend, allocator_Δallocator))

    # primal call
    C_cache = copy(C)
    TensorKit.planartrace!(C, A, p, q, α, β, backend, allocator)

    function planartrace_pullback(::NoRData)
        copy!(C, C_cache)

        ΔAr = planartrace_pullback_ΔA!(ΔA, ΔC, A, p, q, α, backend, allocator) # this typically returns NoRData()
        Δαr = planartrace_pullback_Δα(ΔC, A, p, q, α, backend, allocator)
        Δβr = planartrace_pullback_Δβ(ΔC, C, β)
        ΔCr = planartrace_pullback_ΔC!(ΔC, β) # this typically returns NoRData()

        return NoRData(),
            ΔCr, ΔAr, NoRData(), NoRData(),
            Δαr, Δβr, NoRData(), NoRData()
    end

    return C_ΔC, planartrace_pullback
end

planartrace_pullback_ΔC!(ΔC, β) = (scale!(ΔC, conj(β)); NoRData())

# This implementation is slightly more involved than its non-planar counterpart
# this is because we lack a general `pAB` argument in `planarcontract`, and need
# to keep things planar along the way.
# In particular, we can't simply tensor product with multiple identities in one go
# if they aren't "contiguous", e.g. p = ((1, 4, 5), ()), q = ((2, 6), (3, 7))
function planartrace_pullback_ΔA!(
        ΔA, ΔC, A, p, q, α, backend, allocator
    )
    if length(q[1]) == 0
        ip = invperm(linearize(p))
        pΔA = _repartition(ip, A)
        TK.add_transpose!(ΔA, ΔC, pΔA, conj(α), One(), backend, allocator)
        return NoRData()
    end
    # if length(q[1]) == 1
    #     ip = invperm((p[1]..., q[2]..., p[2]..., q[1]...))
    #     pdA = _repartition(ip, A)
    #     E = one!(TO.tensoralloc_add(scalartype(A), A, q, false))
    #     twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
    #     # pE = ((), trivtuple(TO.numind(q)))
    #     # pΔC = (trivtuple(TO.numind(p)), ())
    #     TensorKit.planaradd!(ΔA, ΔC ⊗ E, pdA, conj(α), One(), backend, allocator)
    #     return NoRData()
    # end
    error("The reverse rule for `planartrace` is not yet implemented")
end

function planartrace_pullback_Δα(
        ΔC, A, p, q, α, backend, allocator
    )
    Tdα = Mooncake.rdata_type(Mooncake.tangent_type(typeof(α)))
    Tdα === NoRData && return NoRData()

    # TODO: this result might be easier to compute as:
    # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
    At = TO.tensoralloc_add(scalartype(A), A, p, false, Val(true), allocator)
    TensorKit.planartrace!(At, A, p, q, One(), Zero(), backend, allocator)
    Δα = inner(At, ΔC)
    TO.tensorfree!(At, allocator)
    return Δα
end

function planartrace_pullback_Δβ(ΔC, C, β)
    Tdβ = Mooncake.rdata_type(Mooncake.tangent_type(typeof(β)))
    Tdβ === NoRData && return NoRData()

    Δβ = inner(C, ΔC)
    return Δβ
end
