module TensorKitChainRulesCoreExt

using TensorOperations
using VectorInterface
using TensorKit
using TensorKit: foreachblock
using ChainRulesCore
using LinearAlgebra
using TupleTools

import TensorOperations as TO
using TensorOperations: promote_contract, tensoralloc_add, tensoralloc_contract
using VectorInterface: promote_scale, promote_add

include("utility.jl")
include("constructors.jl")
include("linalg.jl")
include("tensoroperations.jl")

end
