# tensorcontract!
# ---------------
# TODO: it might be beneficial to compare here if it would make sense to simply compute the
# rrule of permute-permute-gemm-permute, rather than using the contractions directly.
# This could possibly out save some permutations being carried out twice, at the cost of having
# to store some more intermediate objects.
# For example, the combination `ΔC, pΔC, false` appears in the pullback for ΔA and ΔB, so effectively
# this permutation is done multiple times.

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorKit.blas_contract!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        pA::Const{<:Index2Tuple},
        B::Annotation{<:AbstractTensorMap},
        pB::Const{<:Index2Tuple},
        pAB::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    Ccache = isa(β, Const) ? nothing : copy(C.val)
    A_needs_cache = EnzymeRules.overwritten(config)[3] && !(typeof(B) <: Const) && !(typeof(C) <: Const)
    Acache = A_needs_cache ? copy(A.val) : nothing
    B_needs_cache = EnzymeRules.overwritten(config)[5] && !(typeof(A) <: Const) && !(typeof(C) <: Const)
    Bcache = B_needs_cache ? copy(B.val) : nothing
    AB = if !isa(α, Const)
        AB = TO.tensorcontract(A.val, pA.val, false, B.val, pB.val, false, pAB.val, One(), backend.val, allocator.val)
        add!(C.val, AB, α.val, β.val)
        AB
    else
        TensorKit.blas_contract!(C.val, A.val, pA.val, B.val, pB.val, pAB.val, α.val, β.val, backend.val, allocator.val)
        nothing
    end
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    cache = (Ccache, Acache, Bcache, AB)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorKit.blas_contract!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        pA::Const{<:Index2Tuple},
        B::Annotation{<:AbstractTensorMap},
        pB::Const{<:Index2Tuple},
        pAB::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
        allocator::Const
    ) where {RT}
    cacheC, cacheA, cacheB, AB = cache
    Cval = cacheC
    Aval = something(cacheA, A.val)
    Bval = something(cacheB, B.val)

    Δα = isnothing(AB) ? nothing : project_scalar(α.val, inner(AB, C.dval))
    Δβ = isa(β, Const) ? nothing : pullback_dβ(C.dval, Cval, β)

    if !isa(A, Const)
        blas_contract_pullback_ΔA!(
            A.dval, C.dval, Aval, pA.val, Bval, pB.val, pAB.val, α.val, backend.val, allocator.val
        ) # this typically returns nothing
    end
    if !isa(B, Const)
        blas_contract_pullback_ΔB!(
            B.dval, C.dval, Aval, pA.val, Bval, pB.val, pAB.val, α.val, backend.val, allocator.val
        ) # this typically returns nothing
    end
    pullback_dC!(C.dval, β.val) # this typically returns nothing
    return nothing, nothing, nothing, nothing, nothing, nothing, Δα, Δβ, nothing, nothing
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

    return nothing
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

    return nothing
end


# tensortrace!
# ------------

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorKit.trace_permute!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
    ) where {RT}
    C_cache = !isa(β, Const) ? copy(C.val) : nothing
    A_cache = EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
    At = if !isa(α, Const)
        At = TO.tensortrace(A.val, p.val, q.val, false, One(), backend.val)
        add!(C.val, At, α.val, β.val)
        At
    else
        TensorKit.trace_permute!(C.val, A.val, p.val, q.val, α.val, β.val, backend.val)
        nothing
    end
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    cache = (C_cache, A_cache, At)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end


function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorKit.trace_permute!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const,
    ) where {RT}
    C_cache, A_cache, At = cache
    Aval = something(A_cache, A.val)
    Cval = something(C_cache, C.val)
    !isa(A, Const) && !isa(C, Const) && trace_permute_pullback_ΔA!(A.dval, C.dval, Aval, p.val, q.val, α.val, backend.val)
    Δαr = if !isa(C, Const) && !isnothing(At)
        project_scalar(α.val, inner(At, C.dval))
    elseif !isnothing(At)
        zero(α.val)
    else
        nothing
    end
    Δβr = if !isa(β, Const) && !isa(C, Const)
        pullback_dβ(C.dval, Cval, β)
    elseif !isa(β, Const)
        zero(β.val)
    else
        nothing
    end
    !isa(C, Const) && pullback_dC!(C.dval, β.val)
    return nothing, nothing, nothing, nothing, Δαr, Δβr, nothing
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
    return nothing
end
