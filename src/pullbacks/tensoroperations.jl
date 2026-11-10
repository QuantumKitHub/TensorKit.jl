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

# Only the primal `A` is annotated: it fixes the element and space types of the workspace `E`
# below, which `twist!` needs. The tangents stay untyped because their representation differs
# per AD backend (Mooncake passes a tangent object, Enzyme the `.dval` shadow).
function trace_permute_pullback_ΔA!(
        ΔA, ΔC, A::AbstractTensorMap, p, q, α, backend
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

# The planar trace adds `α * U[i, j]` times the dense trace of every source subblock `j` to
# destination subblock `i`. Since `inner` weighs every subblock with the dimension of its coupled
# sector, the adjoint adds `ΔC` to the traced diagonal of every source subblock, with coefficients
# `conj(α * U[i, j]) * dim(cᵢ) / dim(cⱼ)`.
function planartrace_pullback_ΔA!(
        ΔA, ΔC, A::AbstractTensorMap, p, q, α, backend, allocator
    )
    BraidingStyle(sectortype(A)) == Bosonic() &&
        return trace_permute_pullback_ΔA!(ΔA, ΔC, A, p, q, α, backend)
    transformer = treetracer(storagetype(ΔC), space(ΔC), space(A), p, q)
    T = promote_type(scalartype(ΔC), scalartype(α), scalartype(transformer))
    if T <: Real || !(scalartype(ΔA) <: Real)
        _planartrace_pullback_ΔA!(ΔA, ΔC, transformer, p, q, α, backend, allocator)
    else
        ΔAc = zerovector(ΔA, T)
        _planartrace_pullback_ΔA!(ΔAc, ΔC, transformer, p, q, α, backend, allocator)
        add!(ΔA, real(ΔAc))
    end
    return nothing
end

function _planartrace_pullback_ΔA!(
        ΔA, ΔC, transformer::TraceTransformer, (p₁, p₂), (q₁, q₂), α, backend, allocator
    )
    lp = (p₁..., p₂...)
    trees_dst, trees_src = fusiontrees(ΔC), fusiontrees(ΔA)
    dims_dst = [dim(f₁.coupled) for (f₁, _) in trees_dst]
    dims_src = [dim(f₁.coupled) for (f₁, _) in trees_src]
    dst, src = _transform_subblocks(ΔC, ΔA, transformer)
    for (U, inds_dst, inds_src) in transformer.data, (j, isrc) in enumerate(inds_src)
        # the traced diagonal of the source subblock, with the remaining legs permuted by `p`
        a = StridedView(src[isrc])
        sz, str = size(a), strides(a)
        sz_traced = TupleTools.getindices(sz, q₁)
        str_traced = TupleTools.getindices(str, q₁) .+ TupleTools.getindices(str, q₂)
        diag = StridedView(
            parent(a), (TupleTools.getindices(sz, lp)..., sz_traced...),
            (TupleTools.getindices(str, lp)..., str_traced...), a.offset, a.op
        )
        pdiag = (ntuple(identity, ndims(diag)), ())
        for (i, idst) in enumerate(inds_dst)
            iszero(U[i, j]) && continue
            # the destination subblock, constant along the traced legs
            c = StridedView(dst[idst])
            ΔCi = StridedView(
                parent(c), (size(c)..., sz_traced...), (strides(c)..., map(zero, sz_traced)...),
                c.offset, c.op
            )
            coeff = conj(α * U[i, j]) * (dims_dst[pairposition(trees_dst, idst)] / dims_src[pairposition(trees_src, isrc)])
            TO.tensoradd!(diag, ΔCi, pdiag, false, coeff, One(), backend, allocator)
        end
    end
    return ΔA
end
