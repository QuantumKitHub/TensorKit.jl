_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{T}) where {T <: Number} =
    Mooncake.rdata_type(Mooncake.tangent_type(T)) !== NoRData

"""
    project_scalar(x::Number, dx::Number)

Project a computed tangent `dx` onto the correct tangent type for `x`.
For example, we might compute a complex `dx` but only require the real part.
"""
project_scalar(x::Number, dx::Number) = oftype(x, dx)
project_scalar(x::Real, dx::Complex) = project_scalar(x, real(dx))

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

# A VectorSpace has no meaningful notion of a vector space (tangent space)
Mooncake.tangent_type(::Type{<:VectorSpace}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:HomSpace}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{typeof(TensorKit.fusionblockstructure), Any}

@zero_derivative DefaultCtx Tuple{typeof(TensorKit.select), HomSpace, Index2Tuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.flip), HomSpace, Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.permute), HomSpace, Index2Tuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.braid), HomSpace, Index2Tuple, IndexTuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.compose), HomSpace, HomSpace}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorcontract), HomSpace, Index2Tuple, Bool, HomSpace, Index2Tuple, Bool, Index2Tuple}
