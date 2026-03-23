# Projection
# ----------
"""
    project_scalar(x::Number, dx::Number)

Project a computed tangent `dx` onto the correct tangent type for `x`.
For example, we might compute a complex `dx` but only require the real part.
"""
project_scalar(x::Number, dx::Number) = oftype(x, dx)
project_scalar(x::Real, dx::Complex) = project_scalar(x, real(dx))

# in-place multiplication and accumulation which might project to (real)
# TODO: this could probably be done without allocating
function project_mul!(C, A, B, α)
    TC = TO.promote_contract(scalartype(A), scalartype(B), scalartype(α))
    return if !(TC <: Real) && scalartype(C) <: Real
        add!(C, real(mul!(zerovector(C, TC), A, B, α)))
    else
        mul!(C, A, B, α, One())
    end
end
function project_contract!(C, A, pA, conjA, B, pB, conjB, pAB, α, backend, allocator)
    TA = TensorKit.promote_permute(A)
    TB = TensorKit.promote_permute(B)
    TC = TO.promote_contract(TA, TB, scalartype(α))

    return if scalartype(C) <: Real && !(TC <: Real)
        add!(C, real(TO.tensorcontract!(zerovector(C, TC), A, pA, conjA, B, pB, conjB, pAB, α, Zero(), backend, allocator)))
    else
        TO.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, One(), backend, allocator)
    end
end

# IndexTuple utility
# ------------------
trivtuple(N) = ntuple(identity, N)

Base.@constprop :aggressive function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return TupleTools.getindices(p, trivtuple(N₁)),
        TupleTools.getindices(p, trivtuple(length(p) - N₁) .+ N₁)
end
Base.@constprop :aggressive function _repartition(p::Index2Tuple, N₁::Int)
    return _repartition(linearize(p), N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, t::AbstractTensorMap)
    return _repartition(p, TensorKit.numout(t))
end

# Ignore derivatives
# ------------------

@inline EnzymeRules.inactive_type(::Type{<:TensorKit.FusionTree}) = true
@inline EnzymeRules.inactive_type(::Type{<:TensorKit.GenericTreeTransformer}) = true
@inline EnzymeRules.inactive_type(::Type{<:TensorKit.VectorSpace}) = true

@inline EnzymeRules.inactive(::typeof(TensorKit.fusionblockstructure), arg::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.select), s::HomSpace, i::Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.flip), s::HomSpace, i::Any) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.permute), s::HomSpace, i::Index2Tuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.braid), s::HomSpace, i::Index2Tuple, ::IndexTuple) = nothing
@inline EnzymeRules.inactive(::typeof(TensorKit.compose), s1::HomSpace, s2::HomSpace) = nothing
@inline EnzymeRules.inactive(::typeof(TensorOperations.tensorcontract), c::HomSpace, p::Index2Tuple, α::Bool, b::HomSpace, q::Index2Tuple, β::Bool, pq::Index2Tuple) = nothing
