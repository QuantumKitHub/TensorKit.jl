# Shared
# ------
# Can Enzyme do this itself? Apparently not...
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(mul!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
        α::Annotation,
        β::Annotation,
    ) where {RT}
    cacheC = !isa(β, Const) && copy(C.val)
    cacheA = !isa(B, Const) && EnzymeRules.overwritten(config)[3] ? copy(A.val) : nothing
    cacheB = !isa(A, Const) && EnzymeRules.overwritten(config)[4] ? copy(B.val) : nothing
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
    cache = (cacheC, cacheA, cacheB, AB)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
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
    Cval = something(cacheC, C.val)
    Aval = something(cacheA, A.val)
    Bval = something(cacheB, B.val)

    !isa(A, Const) && !isa(C, Const) && TK.project_mul!(A.dval, C.dval, Bval', conj(α.val), One())
    !isa(B, Const) && !isa(C, Const) && TK.project_mul!(B.dval, Aval', C.dval, conj(α.val), One())
    Δαr = pullback_dα(α, C, AB)
    Δβr = pullback_dβ(β, C, Cval)
    !isa(C, Const) && pullback_dC!(C.dval, β.val)

    return (nothing, nothing, nothing, Δαr, Δβr)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(mul!)},
        ::Type{RT},
        C::Annotation{<:AbstractTensorMap},
        A::Annotation{<:AbstractTensorMap},
        B::Annotation{<:AbstractTensorMap},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    # ΔC′ = ΔC*β + C*Δβ + A*B*Δα + ΔA*B*α + A*ΔB*α
    if !isa(C, Const)
        scale!(C.dval, β.val)
        !isa(β, Const) && add!(C.dval, C.val, β.dval)
        !isa(A, Const) && mul!(C.dval, A.dval, B.val, α.val, One())
        !isa(B, Const) && mul!(C.dval, A.val, B.dval, α.val, One())
    end
    if !isa(α, Const) && !isa(C, Const)
        if iszero(β.val) && !iszero(α.val)
            # this is probably quite a common case, so maybe worth specializing
            mul!(C.val, A.val, B.val, α.val, β.val)
            add!(C.dval, C.val, α.dval / α.val)
        else
            AB = A.val * B.val
            add!(C.val, AB, α.val, β.val)
            add!(C.dval, AB, α.dval)
        end
    else
        mul!(C.val, A.val, B.val, α.val, β.val)
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return C
    elseif EnzymeRules.needs_primal(config)
        return C.val
    elseif EnzymeRules.needs_shadow(config)
        return C.dval
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(tr)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    ret = func.val(A.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(ret) : nothing
    cache = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
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
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        ::Type{RT},
        func::Const{typeof(tr)},
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    y = EnzymeRules.needs_primal(config) ? tr(A.val) : nothing
    Δy = if EnzymeRules.needs_shadow(config) && !isa(A, Const)
        tr(A.dval)
    elseif EnzymeRules.needs_shadow(config)
        zero(eltype(A.dval))
    else
        nothing
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(y, Δy)
    elseif EnzymeRules.needs_primal(config)
        return y
    elseif EnzymeRules.needs_shadow(config)
        return Δy
    else
        return nothing
    end
end
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(norm)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Real},
    ) where {RT}
    ret = func.val(A.val, p.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(ret) : nothing
    cacheA = EnzymeRules.overwritten(config)[2] ? copy(A.val) : nothing
    cache = (ret, cacheA)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(norm)},
        dret::Active,
        cache,
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Real},
    )
    n, cacheA = cache
    Δn = dret.val
    p.val == 2 || error("currently only implemented for p = 2")
    Aval = something(cacheA, A.val)
    if !isa(A, Const)
        x = real(Δn) / hypot(n, eps(one(n)))
        add!(A.dval, A.val, x)
    end
    return (nothing, nothing)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(norm)},
        ::Type{<:Const},
        cache,
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Real},
    )
    return (nothing, nothing)
end
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(norm)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
        p::Const{<:Real},
    ) where {RT}
    p.val == 2 || error("currently only implemented for p = 2")
    y = norm(A.val, p.val)
    Δy = if EnzymeRules.needs_shadow(config) && !isa(A, Const)
        real(dot(A.val, A.dval)) * pinv(y)
    elseif EnzymeRules.needs_shadow(config)
        zero(eltype(A.dval))
    else
        nothing
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(y, Δy)
    elseif EnzymeRules.needs_primal(config)
        return y
    elseif EnzymeRules.needs_shadow(config)
        return Δy
    else
        return nothing
    end
end
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inv)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    ret = inv(A.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    shadow = EnzymeRules.needs_shadow(config) ? make_zero(ret) : nothing
    cache_ret = EnzymeRules.needs_primal(config) ? copy(ret) : ret
    cache = (cache_ret, shadow)
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

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(inv)},
        ::Type{RT},
        A::Annotation{<:AbstractTensorMap},
    ) where {RT}
    Ainv = inv(A.val)
    ΔAinv = !isa(A, Const) ? scale!(Ainv * A.dval * Ainv, -1) : make_zero(Ainv)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(Ainv, ΔAinv)
    elseif EnzymeRules.needs_primal(config)
        return Ainv
    elseif EnzymeRules.needs_shadow(config)
        return ΔAinv
    else
        return nothing
    end
end
