module TensorKitAdaptExt

using TensorKit
using TensorKit: AdjointTensorMap
using Adapt

function Adapt.adapt_structure(to, x::TensorMap)
    data′ = adapt(to, x.data)
    return TensorMap{eltype(data′)}(data′, space(x))
end
function Adapt.adapt_structure(to, x::AdjointTensorMap)
    return adjoint(adapt(to, parent(x)))
end
function Adapt.adapt_structure(to, x::DiagonalTensorMap)
    data′ = adapt(to, x.data)
    return DiagonalTensorMap(data′, x.domain)
end
function Adapt.adapt_structure(::Type{T}, x::BraidingTensor{T′, S, A}) where {T <: Number, T′, S, A}
    A′ = TensorKit.similarstoragetype(A, T)
    return BraidingTensor{T, S, A′}(space(x), x.adjoint)
end
function Adapt.adapt_structure(::Type{TA}, x::BraidingTensor{T, S, A}) where {T′, TA <: DenseArray{T′}, T, S, A}
    return BraidingTensor{T′, S, TA}(space(x), x.adjoint)
end

end
