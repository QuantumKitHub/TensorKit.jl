# Projection
# ----------
pullback_dα(α::Const, C::Const, A) = nothing
pullback_dα(α::Const, C::Annotation, A) = nothing
pullback_dα(α::Annotation, C::Const, A) = zero(α.val)
pullback_dα(α::Annotation, C::Annotation, A) = project_scalar(α.val, inner(A, C.dval))

pullback_dβ(β::Const, C::Const, Ccache) = nothing
pullback_dβ(β::Const, C::Annotation, Ccache) = nothing
pullback_dβ(β::Annotation, C::Const, Ccache) = zero(β.val)
pullback_dβ(β::Annotation, C::Annotation, Ccache) = project_scalar(β.val, inner(Ccache, C.dval))

pullback_dC!(ΔC, β::Number) = scale!(ΔC, conj(β))

"""
    project_scalar(x::Number, dx::Number)

Project a computed tangent `dx` onto the correct tangent type for `x`.
For example, we might compute a complex `dx` but only require the real part.
"""
project_scalar(x::Number, dx::Number) = oftype(x, dx)
project_scalar(x::Real, dx::Complex) = project_scalar(x, real(dx))

# Ignore derivatives
# ------------------

@inline EnzymeRules.inactive_type(::Type{<:TensorKit.FusionTree}) = true
@inline EnzymeRules.inactive_type(::Type{<:TensorKit.GenericTreeTransformer}) = true
@inline EnzymeRules.inactive_type(::Type{<:TensorKit.VectorSpace}) = true

@inline EnzymeRules.inactive(::typeof(TensorKit.sectorstructure), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.degeneracystructure), ::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.select), s::HomSpace, i::TensorKit.Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.flip), s::HomSpace, i::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.permute), s::HomSpace, i::TensorKit.Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.braid), s::HomSpace, i::TensorKit.Index2Tuple, ::TensorKit.IndexTuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.compose), s1::HomSpace, s2::HomSpace) = nothing
#@inline EnzymeRules.inactive(::typeof(TensorOperations.tensorcontract), c::HomSpace, p::TensorKit.Index2Tuple, α::Bool, b::HomSpace, q::TensorKit.Index2Tuple, β::Bool, pq::TensorKit.Index2Tuple) = nothing
