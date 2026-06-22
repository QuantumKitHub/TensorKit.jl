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

@inline EnzymeRules.inactive(::typeof(TensorKit.sectorstructure), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.degeneracystructure), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.select), s::HomSpace, i::Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.flip), s::HomSpace, i::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.permute), s::HomSpace, i::Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.braid), s::HomSpace, i::Index2Tuple, ::IndexTuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.compose), s1::HomSpace, s2::HomSpace) = nothing
@inline EnzymeRules.inactive(::typeof(TensorOperations.tensorcontract), c::HomSpace, p::Index2Tuple, α::Bool, b::HomSpace, q::Index2Tuple, β::Bool, pq::Index2Tuple) = nothing
