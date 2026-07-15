# Projection
# ----------
pullback_dα(α::Const, C::Const, A) = nothing
pullback_dα(α::Const, C::Annotation, A) = nothing
pullback_dα(α::Annotation, C::Const, A) = zero(α.val)
pullback_dα(α::Annotation, C::Annotation, A) = TK._pullback_dα(α.val, C.dval, A)

pullback_dβ(β::Const, C::Const, Ccache) = nothing
pullback_dβ(β::Const, C::Annotation, Ccache) = nothing
pullback_dβ(β::Annotation, C::Const, Ccache) = zero(β.val)
pullback_dβ(β::Annotation, C::Annotation, Ccache) = TK._pullback_dβ(β.val, C.dval, Ccache)

pullback_dC!(ΔC, β::Number) = scale!(ΔC, conj(β))

# Ignore derivatives
# ------------------

@inline EnzymeRules.inactive_type(::Type{<:TensorKit.FusionTree}) = true
@inline EnzymeRules.inactive_type(::Type{<:TensorKit.GenericTreeTransformer}) = true
@inline EnzymeRules.inactive_type(::Type{<:TensorKit.VectorSpace}) = true

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(subblock)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        f::Annotation,
    ) where {RT}
    ret = EnzymeRules.needs_primal(config) ? subblock(t.val, f.val) : nothing
    dret = if !isa(t, Const) && EnzymeRules.needs_shadow(config)
        subblock(t.dval, f.val)
    elseif EnzymeRules.needs_shadow(config)
        Enzyme.make_zero(ret)
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(ret, dret, dret)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(subblock)},
        ::Type{RT},
        cache,
        t::Annotation{<:AbstractTensorMap},
        f::Annotation,
    ) where {RT}
    dret = cache
    if !isnothing(dret) && !isa(t, Const)
        subblock(t.dval, f.val) .= dret
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(block)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        c::Annotation{<:Sector},
    ) where {RT}
    ret = EnzymeRules.needs_primal(config) ? block(t.val, c.val) : nothing
    dret = if !isa(t, Const) && EnzymeRules.needs_shadow(config)
        block(t.dval, c.val)
    elseif EnzymeRules.needs_shadow(config)
        Enzyme.make_zero(ret)
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(ret, dret, dret)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(block)},
        ::Type{RT},
        cache,
        t::Annotation{<:AbstractTensorMap},
        c::Annotation{<:Sector},
    ) where {RT}
    dret = cache
    if !isnothing(dret) && !isa(t, Const)
        block(t.dval, c.val) .= dret
    end
    return (nothing, nothing)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        ::Const{typeof(subblock)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        f::Annotation,
    ) where {RT}
    ret = EnzymeRules.needs_primal(config) ? subblock(t.val, f.val) : nothing
    dret = if !isa(t, Const) && EnzymeRules.needs_shadow(config)
        subblock(t.dval, f.val)
    elseif EnzymeRules.needs_shadow(config)
        Enzyme.make_zero(ret)
    else
        nothing
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(ret, dret)
    elseif EnzymeRules.needs_primal(config)
        return ret
    elseif EnzymeRules.needs_shadow(config)
        return dret
    else
        return nothing
    end
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        ::Const{typeof(block)},
        ::Type{RT},
        t::Annotation{<:AbstractTensorMap},
        c::Annotation{<:Sector},
    ) where {RT}
    ret = EnzymeRules.needs_primal(config) ? block(t.val, c.val) : nothing
    dret = if !isa(t, Const) && EnzymeRules.needs_shadow(config)
        block(t.dval, c.val)
    elseif EnzymeRules.needs_shadow(config)
        Enzyme.make_zero(ret)
    else
        nothing
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(ret, dret)
    elseif EnzymeRules.needs_primal(config)
        return ret
    elseif EnzymeRules.needs_shadow(config)
        return dret
    else
        return nothing
    end
end

@inline EnzymeRules.inactive(::typeof(TensorKit.fsbraid), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.fsbraid), ::Any, ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.artin_braid), ::Any, ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.insertleftunit), ::HomSpace, ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.insertrightunit), ::HomSpace, ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.removeunit), ::HomSpace, ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.sectorstructure), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.degeneracystructure), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.select), s::HomSpace, i::Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.flip), s::HomSpace, i::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.permute), s::HomSpace, i::Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.braid), s::HomSpace, i::Index2Tuple, ::IndexTuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.compose), s1::HomSpace, s2::HomSpace) = nothing
@inline EnzymeRules.inactive(::typeof(TensorOperations.tensorcontract), c::HomSpace, p::Index2Tuple, α::Bool, b::HomSpace, q::Index2Tuple, β::Bool, pq::Index2Tuple) = nothing
