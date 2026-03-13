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
function Adapt.adapt_structure(::Type{TorA}, x::BraidingTensor) where {TorA <: Union{Number, DenseArray{<:Number}}}
    return BraidingTensor{scalartype(TorA)}(space(x), x.adjoint)
end

end
