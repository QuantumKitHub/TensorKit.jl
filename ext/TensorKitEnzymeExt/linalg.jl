# Shared
# ------
pullback_dC!(ΔC, β) = scale!(ΔC, conj(β))
pullback_dβ(ΔC, C, β) = !isa(β, Const) ? project_scalar(β.val, inner(C, ΔC)) : nothing

# Can Enzyme do this itself? Apparently not...
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(mul!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    cacheC = copy(C.val)
    AB = if !isa(α, Const)
        AB = A.val * B.val
        add!(C.val, AB, α.val, β.val)
        AB
    else
        mul!(C.val, A.val, B.val, α.val, β.val)
        nothing
    end
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    cacheA = EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
    cacheB = EnzymeRules.overwritten(config)[4] ? copy(B.val) : nothing

    cache = (cacheC, cacheA, cacheB, AB)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(mul!)},
        ::Type{<:Const},
        C::Const{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    )
    mul!(C.val, A.val, B.val, α.val, β.val)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(mul!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    if RT <: Const
        Δα = isa(α, Const) ? nothing : zero(α.val)
        Δβ = isa(β, Const) ? nothing : zero(β.val)
        return (nothing, nothing, nothing, Δα, Δβ)
    end
    cacheC, cacheA, cacheB, AB = cache
    Cval = cacheC
    Aval = something(cacheA, A.val)
    Bval = something(cacheB, B.val)

    !isa(A, Const) && project_mul!(A.dval, C.dval, Bval', conj(α.val))
    !isa(B, Const) && project_mul!(B.dval, Aval', C.dval, conj(α.val))
    Δαr = isnothing(AB) ? nothing : project_scalar(α.val, inner(AB, C.dval))
    Δβr = pullback_dβ(C.dval, Cval, β)
    pullback_dC!(C.dval, β.val)

    return (nothing, nothing, nothing, Δαr, Δβr)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(tr)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    ret = func.val(A.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    cache = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(tr)},
        dret::Active,
        cache,
        A::Annotation{<:AbstractTensorMap},
    )
    Aval = something(cache, A.val)
    Δtrace = dret.val
    if !isa(A, Const)
        for (_, b) in blocks(A.dval)
            TensorKit.diagview(b) .+= Δtrace
        end
    end
    return (nothing,)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(tr)},
        ::Type{<:Const},
        cache,
        A::Annotation{<:AbstractTensorMap},
    )
    return (nothing,)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inv)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    ret = func.val(A.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(ret) : nothing
    cache = (copy(ret), shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inv)},
        ::Type{RT},
        cache,
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    Ainv, ΔAinv = cache
    !isa(A, Const) && mul!(A.dval, Ainv' * ΔAinv, Ainv', -1, One())
    return (nothing,)
end
