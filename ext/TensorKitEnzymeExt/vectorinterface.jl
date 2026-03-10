function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
    ) where {RT}
    C_cache = !isa(α, Const) ? copy(C.val) : nothing
    scale!(C.val, α.val)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, C_cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
    ) where {RT}
    Cval = something(cache, C.val)
    Δα = if !isa(α, Const) && !isa(C, Const)
        project_scalar(α.val, inner(Cval, C.dval))
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    !isa(C, Const) && scale!(C.dval, conj(α.val))
    return (nothing, Δα)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
    ) where {RT}
    A_cache = !isa(α, Const) && EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
    scale!(C.val, A.val, α.val)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    cache = A_cache
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
    ) where {RT}
    Aval = something(cache, A.val)
    Δα = if !isa(α, Const) && !isa(C, Const)
        project_scalar(α.val, inner(Aval, C.dval))
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    !isa(A, Const) && !isa(C, Const) && add!(A.dval, C.dval, conj(α.val))
    !isa(C, Const) && zerovector!(C.dval)
    return (nothing, nothing, Δα)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(add!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    A_cache = !isa(α, Const) && EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
    C_cache = !isa(β, Const) ? copy(C.val) : nothing
    add!(C.val, A.val, α.val, β.val)
    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    cache = (A_cache, C_cache)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(add!)},
        ::Type{RT},
        cache,
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    A_cache, C_cache = cache
    Aval = something(A_cache, A.val)
    Cval = something(C_cache, C.val)
    Δα = if !isa(α, Const) && !isa(C, Const)
        project_scalar(α.val, inner(Aval, C.dval))
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    Δβ = if !isa(β, Const) && !isa(C, Const)
        project_scalar(β.val, inner(Cval, C.dval))
    elseif !isa(β, Const)
        zero(β.val)
    else
        nothing
    end
    !isa(A, Const) && !isa(C, Const) && add!(A.dval, C.dval, conj(α.val))
    !isa(C, Const) && scale!(C.dval, conj(β.val))
    return (nothing, nothing, Δα, Δβ)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inner)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
    ) where {RT}
    A_cache = !isa(B, Const) && EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    B_cache = !isa(A, Const) && EnzymeRules.overwritten(config)[3] ? copy(B.val) : nothing
    ret = inner(A.val, B.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(ret) : nothing
    cache = (A_cache, B_cache)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inner)},
        dret::Active,
        cache,
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
    )
    A_cache, B_cache = cache
    Aval = something(A_cache, A.val)
    Bval = something(B_cache, B.val)
    Δs = dret.val
    !isa(A, Const) && add!(A.dval, Bval, conj(Δs))
    !isa(B, Const) && add!(B.dval, Aval, Δs)
    return (nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inner)},
        ::Type{<:Const},
        cache,
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
    )
    return (nothing, nothing)
end
