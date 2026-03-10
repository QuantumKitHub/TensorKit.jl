module TensorKitEnzymeExt

using Enzyme
using TensorKit
import TensorKit as TK
using VectorInterface
using TensorOperations: TensorOperations, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using MatrixAlgebraKit
using TupleTools
using Random: AbstractRNG

include("utility.jl")
include("linalg.jl")
include("vectorinterface.jl")
include("tensoroperations.jl")
include("factorizations.jl")
include("indexmanipulations.jl")
#include("planaroperations.jl")

end
