module TensorKitMooncakeExt

using Mooncake
using Mooncake: @zero_derivative, @is_primitive,
    DefaultCtx, MinimalCtx, ReverseMode, NoFData, NoRData, NoTangent,
    CoDual, Dual, arrayify, primal, tangent, zero_fcodual
using TensorKit
import TensorKit as TK
using VectorInterface
using TensorOperations: TensorOperations, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using MatrixAlgebraKit
using TupleTools
using Random: AbstractRNG

include("utility.jl")
include("tangent.jl")
include("linalg.jl")
include("indexmanipulations.jl")
include("vectorinterface.jl")
include("tensoroperations.jl")
include("planaroperations.jl")
include("factorizations.jl")

end
