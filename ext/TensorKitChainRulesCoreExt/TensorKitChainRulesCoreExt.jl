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

using MatrixAlgebraKit
using MatrixAlgebraKit: TruncationStrategy,
                        svd_compact_pullback!, eig_full_pullback!, eigh_full_pullback!,
                        qr_compact_pullback!, lq_compact_pullback!

include("utility.jl")
include("constructors.jl")
include("linalg.jl")
include("tensoroperations.jl")
include("factorizations.jl")

end
