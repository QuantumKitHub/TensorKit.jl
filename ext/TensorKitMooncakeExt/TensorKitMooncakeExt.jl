module TensorKitMooncakeExt

using Mooncake
using Mooncake: @zero_derivative, DefaultCtx, ReverseMode, NoRData, CoDual, arrayify, primal
using TensorKit
using TensorOperations: TensorOperations, tensorcontract!, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using VectorInterface: One, Zero
using TupleTools


include("utility.jl")
include("tangent.jl")
include("linalg.jl")
include("tensoroperations.jl")

end
