function planartrace_pullback_dA!(
        ΔA, ΔC, A, p, q, α, backend, allocator
    )
    if length(q[1]) == 0
        ip = invperm(linearize(p))
        pΔA = TO.repartition(ip, numout(A))
        transpose!(ΔA, ΔC, pΔA, conj(α), One(), backend, allocator)
        return nothing
    end
    if length(q[1]) == 1
        ip = invperm((p[1]..., q[2]..., p[2]..., q[1]...))
        pdA = TO.repartition(ip, numout(A))
        E = one!(TO.tensoralloc_add(scalartype(A), A, q, false))
        twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
        pE = ((), trivtuple(TO.numind(q)))
        pΔC = (trivtuple(TO.numind(p)), ())
        planaradd!(ΔA, ΔC ⊗ E, pdA, conj(α), One(), backend, allocator)
        return nothing
    end
    error("The reverse rule for `planartrace` is not yet implemented")
end

function planartrace_pullback_dα(
        ΔC, A, p, q, α, backend, allocator
    )
    # TODO: this result might be easier to compute as:
    # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
    At = TO.tensoralloc_add(scalartype(A), A, p, false, Val(true), allocator)
    planartrace!(At, A, p, q, One(), Zero(), backend, allocator)
    Δα = project_scalar(α, inner(At, ΔC))
    TO.tensorfree!(At, allocator)
    return Δα
end
