module TensorKitEnzymeExt

using Enzyme
using TensorKit
import TensorKit as TK
using TensorKit: subblock
using VectorInterface
using TensorOperations: TensorOperations, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using MatrixAlgebraKit
using TupleTools
using Random: AbstractRNG

include("utility.jl")
include("linalg.jl")
include("indexmanipulations.jl")
include("tensoroperations.jl")

end
