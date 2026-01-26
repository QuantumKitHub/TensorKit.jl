module TensorKitMooncakeExt

using Mooncake
using Mooncake: @zero_derivative, @is_primitive, DefaultCtx, ReverseMode, NoFData, NoRData, CoDual, arrayify, primal
using TensorKit
import TensorKit as TK
using VectorInterface
using TensorOperations: TensorOperations, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using TupleTools

include("utility.jl")
include("tangent.jl")
include("linalg.jl")
include("indexmanipulations.jl")
include("vectorinterface.jl")
include("tensoroperations.jl")
include("planaroperations.jl")

end
