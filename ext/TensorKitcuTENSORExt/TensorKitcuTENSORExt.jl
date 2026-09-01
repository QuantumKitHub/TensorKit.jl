module TensorKitcuTENSORExt

using CUDA
using CUDA: CuPtr, with_workspace
using cuTENSOR: cuTENSOR

using LRUCache: LRU

using TensorKit
using TensorKit: BlockSparseSigns, BlockSparseStructure, BlockSparseUnsupported,
    CuTENSORBlockSparse, HomSpace, Index2Tuple,
    blocksparse_compatible, blocksparse_contract_signs,
    blocksparsestructure, _scale_subblocks!,
    nblocks, nmodes,
    GLOBAL_CACHES, _generic_tensorcontract!
import TensorKit: _tensorcontract!, plan_contract

using TensorOperations: TensorOperations as TO

"""
    CuTensorMapAny{T, S, N₁, N₂}

Any `TensorMap` whose flat storage lives in CUDA device memory. Dispatching on the concrete
storage type makes the block-sparse `_tensorcontract!` method unambiguously more specific
than the generic one, and lets every other tensor type fall through to the default path.
"""
const CuTensorMapAny{T, S, N₁, N₂} =
    TensorMap{T, S, N₁, N₂, <:CUDA.CuVector{T, CUDA.DeviceMemory}}

"""
The scalar types cuTENSOR's block-sparse backend supports. All of A, B, C, D, α, β and the
compute type must agree.
"""
const BLOCKSPARSE_TYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

"Maximum number of modes the block-sparse backend accepts: 8 up to cuTENSOR 2.5, 32 from 2.7."
const MAX_BLOCKSPARSE_MODES = 8

include("descriptors.jl")
include("plan.jl")
include("contract.jl")

function __init__()
    # register with TensorKit's cache registry, so `empty_globalcaches!` covers the plan cache
    if !any(((name, _),) -> name === :BLOCKSPARSE_PLAN_CACHE, GLOBAL_CACHES)
        push!(GLOBAL_CACHES, :BLOCKSPARSE_PLAN_CACHE => BLOCKSPARSE_PLAN_CACHE)
    end
    return nothing
end

end
