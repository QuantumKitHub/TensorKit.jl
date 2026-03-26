# planartrace!
# ------------
# TODO: Fix planartrace pullback
# This implementation is slightly more involved than its non-planar counterpart
# this is because we lack a general `pAB` argument in `planarcontract`, and need
# to keep things planar along the way.
# In particular, we can't simply tensor product with multiple identities in one go
# if they aren't "contiguous", e.g. p = ((1, 4, 5), ()), q = ((2, 6), (3, 7))

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(TensorKit.planartrace!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const, allocator::Const
    )
    cacheC = !isa(β, Const) && copy(C.val)
    cacheA = EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing

    TensorKit.planartrace!(C.val, A.val, p.val, q.val, α.val, β.val, backend.val, allocator.val)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cacheC, cacheA))
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(TensorKit.planartrace!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Index2Tuple},
        q::Const{<:Index2Tuple},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
        backend::Const, allocator::Const
    )
    cacheC, cacheA = cache
    Cval = something(cacheC, C.val)
    Aval = something(cacheA, A.val)

    if !isa(A, Const) && !isa(C, Const)
        planartrace_pullback_ΔA!(A.dval, C.dval, Aval, p.val, q.val, α.val, backend.val, allocator.val)
    end
    Δαr = if !isa(α, Const) && !isa(C, Const)
        planartrace_pullback_Δα(C.dval, A.val, p.val, q.val, α.val, backend.val, allocator.val)
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    Δβr = if !isa(β, Const) && !isa(C, Const)
        pullback_dβ(C.dval, C.val, β)
    elseif !isa(β, Const)
        zero(β.val)
    else
        nothing
    end
    !isa(C, Const) && pullback_dC!(C.dval, β.val)

    return nothing, nothing, nothing, nothing, Δαr, Δβr, nothing, nothing
end

function planartrace_pullback_dA!(
        ΔA, ΔC, A, p, q, α, backend, allocator
    )
    if length(q[1]) == 0
        ip = invperm(linearize(p))
        pΔA = _repartition(ip, A)
        TK.add_transpose!(ΔA, ΔC, pΔA, conj(α), One(), backend, allocator)
        return nothing
    end
    if length(q[1]) == 1
        ip = invperm((p[1]..., q[2]..., p[2]..., q[1]...))
        pdA = _repartition(ip, A)
        E = one!(TO.tensoralloc_add(scalartype(A), A, q, false))
        twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
        pE = ((), trivtuple(TO.numind(q)))
        pΔC = (trivtuple(TO.numind(p)), ())
        TensorKit.planaradd!(ΔA, ΔC ⊗ E, pdA, conj(α), One(), backend, allocator)
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
    TensorKit.planartrace!(At, A, p, q, One(), Zero(), backend, allocator)
    Δα = project_scalar(α, inner(At, ΔC))
    TO.tensorfree!(At, allocator)
    return Δα
end
