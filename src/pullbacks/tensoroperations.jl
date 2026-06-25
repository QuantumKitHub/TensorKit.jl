function blas_contract_pullback_ΔA!(
        ΔA, ΔC, A, pA, B, pB, pAB, α, backend, allocator
    )
    ipAB = invperm(linearize(pAB))
    pΔC = TO.repartition(ipAB, TO.numout(pA))
    ipA = TO.repartition(invperm(linearize(pA)), numout(A))
    tB = twist(
        B,
        vcat(
            [i for i in pB[1] if !isdual(space(B, i))],
            [i for i in pB[2] if isdual(space(B, i))]
        ); copy = false
    )

    project_contract!(
        ΔA,
        ΔC, pΔC, false,
        tB, reverse(pB), true,
        ipA, conj(α), backend, allocator
    )

    return nothing
end

function blas_contract_pullback_ΔB!(
        ΔB, ΔC, A, pA, B, pB, pAB, α, backend, allocator
    )
    ipAB = invperm(linearize(pAB))
    pΔC = TO.repartition(ipAB, TO.numout(pA))
    ipB = TO.repartition(invperm(linearize(pB)), numout(B))

    tA = twist(
        A,
        vcat(
            [i for i in pA[1] if isdual(space(A, i))],
            [i for i in pA[2] if !isdual(space(A, i))]
        ); copy = false
    )

    project_contract!(
        ΔB,
        tA, reverse(pA), true,
        ΔC, pΔC, false,
        ipB, conj(α), backend, allocator
    )

    return nothing
end

function trace_permute_pullback_ΔA!(
        ΔA, ΔC, A, p, q, α, backend
    )
    ip = invperm((linearize(p)..., q[1]..., q[2]...))
    pdA = TO.repartition(ip, numout(A))
    E = one!(TO.tensoralloc_add(scalartype(A), A, q, false))
    twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
    pE = ((), TO.trivialpermutation(TO.numind(q)))
    pΔC = (TO.trivialpermutation(TO.numind(p)), ())
    TO.tensorproduct!(
        ΔA, ΔC, pΔC, false, E, pE, false, pdA, conj(α), One(), backend
    )
    return nothing
end
