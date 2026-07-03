for transform in (:permute, :transpose)
    transform! = Symbol(transform, :!)
    transform_pb = Symbol(transform, :_pullback_dA)
    @eval function $transform_pb(ΔA, A, ΔC, C, p, α, ba...)
        ip = invperm(linearize(p))
        pΔA = TO.repartition(ip, numout(A))
        TC = VectorInterface.promote_scale(C, α)
        if scalartype(ΔA) <: Real && !(TC <: Real)
            ΔAc = TO.tensoralloc_add(TC, ΔC, pΔA, false, Val(false))
            $transform!(ΔAc, ΔC, pΔA, conj(α), Zero(), ba...)
            add!(ΔA, real(ΔAc))
        else
            $transform!(ΔA, ΔC, pΔA, conj(α), One(), ba...)
        end
        return
    end
end
function braid_pb(ΔA, A, ΔC, C, p, levels, α, ba...)
    ip = invperm(linearize(p))
    pΔA = TO.repartition(ip, numout(A))
    ilevels = TupleTools.permute(levels, linearize(p))
    TC = VectorInterface.promote_scale(ΔC, α)
    if scalartype(ΔA) <: Real && !(TC <: Real)
        ΔAc = TO.tensoralloc_add(TC, ΔC, pΔA, false, Val(false))
        braid!(ΔAc, ΔC, pΔA, ilevels, conj(α), Zero(), ba...)
        add!(ΔA, real(ΔAc))
    else
        braid!(ΔA, ΔC, pΔA, ilevels, conj(α), One(), ba...)
    end
    return
end
