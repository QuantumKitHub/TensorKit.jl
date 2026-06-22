_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{T}) where {T <: Number} =
    Mooncake.rdata_type(Mooncake.tangent_type(T)) !== NoRData

pullback_dα(α, ΔC, A) = _needs_tangent(α) ? TK._pullback_dα(α, ΔC, A) : NoRData()
pullback_dβ(β, ΔC, C) = _needs_tangent(β) ? TK._pullback_dβ(β, ΔC, C) : NoRData()

# Ignore derivatives
# ------------------

# A VectorSpace has no meaningful notion of a vector space (tangent space)
Mooncake.tangent_type(::Type{<:VectorSpace}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:HomSpace}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{typeof(TensorKit.sectorstructure), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.degeneracystructure), Any}

@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorstructure), AbstractTensorMap}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorstructure), AbstractTensorMap, Int, Bool}

@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorcontract_structure), AbstractTensorMap, Index2Tuple, Bool, AbstractTensorMap, Index2Tuple, Bool, Index2Tuple}

@zero_derivative DefaultCtx Tuple{typeof(TensorKit.has_shared_permute), AbstractTensorMap, Index2Tuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.select), HomSpace, Index2Tuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.flip), HomSpace, Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.permute), HomSpace, Index2Tuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.braid), HomSpace, Index2Tuple, IndexTuple}
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.compose), HomSpace, HomSpace}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorcontract), HomSpace, Index2Tuple, Bool, HomSpace, Index2Tuple, Bool, Index2Tuple}
