# Block-sparse contraction plans
# ------------------------------

"""
    BlockSparseKey

Everything a block-sparse contraction plan depends on, which notably excludes the tensor
data: block pointers are supplied at execution time, which is what makes plans reusable.
A cuTENSOR plan is only valid on the device active at creation time, hence the `device` field.
"""
struct BlockSparseKey{SC, SA, SB}
    VC::SC
    VA::SA
    VB::SB
    pA::Index2Tuple
    pB::Index2Tuple
    pAB::Index2Tuple
    scalartype::DataType
    jit::Bool
    device::Int
end

function Base.hash(k::BlockSparseKey, h::UInt)
    h = hash(k.VC, hash(k.VA, hash(k.VB, h)))
    h = hash(k.pA, hash(k.pB, hash(k.pAB, h)))
    return hash(k.scalartype, hash(k.jit, hash(k.device, h)))
end
function Base.:(==)(a::BlockSparseKey, b::BlockSparseKey)
    return a.VC == b.VC && a.VA == b.VA && a.VB == b.VB &&
        a.pA == b.pA && a.pB == b.pB && a.pAB == b.pAB &&
        a.scalartype === b.scalartype && a.jit == b.jit && a.device == b.device
end

"""
    BlockSparsePlan <: TensorOperations.AbstractBackend

A precomputed cuTENSOR block-sparse contraction, usable directly as a backend:

```julia
plan = plan_contract(C, A, pA, B, pB, pAB)
@tensor backend = plan C[a, b] = A[a, c] * B[c, b]
```

The device workspace is deliberately *not* retained: two tasks executing the same cached plan
on different streams would race on a shared workspace. Only its size is kept, and the buffer
is taken from the memory pool per call.

The per-block sign correction a fermionic sector needs is carried here too. It depends on
`conjA`/`conjB`, which are not part of a plan's identity, so all four combinations are derived
up front and indexed by [`_signs_index`](@ref).

See also [`plan_contract`](@ref), [`CuTENSORBlockSparse`](@ref).
"""
struct BlockSparsePlan <: TO.AbstractBackend
    plan::cuTENSOR.CuTensorPlan
    workspacesize::Int
    scalartype::DataType
    signs::NTuple{4, BlockSparseSigns}
    key::Any
end

"index of the `(conjA, conjB)` combination into a plan's `signs` tuple"
_signs_index(conjA::Bool, conjB::Bool) = 1 + conjA + 2 * conjB

function Base.show(io::IO, p::BlockSparsePlan)
    k = p.key::BlockSparseKey
    return print(
        io, "BlockSparsePlan(", k.VA, ", ", k.VB, " -> ", k.VC,
        "; ", p.scalartype, ", workspace = ", Base.format_bytes(p.workspacesize), ")"
    )
end

CUDA.unsafe_free!(p::BlockSparsePlan) = CUDA.unsafe_free!(p.plan)

# small by default: an entry holds a cuTENSOR plan handle, and with JIT a compiled kernel
const BLOCKSPARSE_PLAN_CACHE_SIZE = Ref(128)
const BLOCKSPARSE_PLAN_CACHE = LRU{Any, BlockSparsePlan}(;
    maxsize = BLOCKSPARSE_PLAN_CACHE_SIZE[]
)

"""
    blocksparse_plancache_size!(n)

Set the maximum number of cuTENSOR block-sparse contraction plans kept in the global cache.
"""
function blocksparse_plancache_size!(n::Integer)
    BLOCKSPARSE_PLAN_CACHE_SIZE[] = n
    resize!(BLOCKSPARSE_PLAN_CACHE; maxsize = n)
    return n
end

"create the cuTENSOR descriptor and plan; runs the kernel-selection heuristic that caching saves"
function _create_plan(key::BlockSparseKey)
    T = key.scalartype
    descA = bs_descriptor(key.VA, T)
    descB = bs_descriptor(key.VB, T)
    descC = bs_descriptor(key.VC, T)
    modeA, modeB, modeC = map(
        l -> collect(Cint, l), TO.contract_labels(key.pA, key.pB, key.pAB)
    )
    computetype = cuTENSOR.contraction_compute_types[(T, T, T)]

    desc = Ref{cuTENSOR.cutensorOperationDescriptor_t}()
    # `descD` must be the *same* descriptor as `descC`, and C and D must share their layout
    GC.@preserve descA descB descC begin
        cuTENSOR.cutensorCreateBlockSparseContraction(
            cuTENSOR.handle(), desc,
            descA, modeA, cuTENSOR.OP_IDENTITY,
            descB, modeB, cuTENSOR.OP_IDENTITY,
            descC, modeC, cuTENSOR.OP_IDENTITY,
            descC, modeC, computetype
        )
    end
    pref = Ref{cuTENSOR.cutensorPlanPreference_t}()
    cuTENSOR.cutensorCreatePlanPreference(
        cuTENSOR.handle(), pref, cuTENSOR.ALGO_DEFAULT,
        key.jit ? cuTENSOR.JIT_MODE_DEFAULT : cuTENSOR.JIT_MODE_NONE
    )

    plan = cuTENSOR.CuTensorPlan(desc[], pref[])
    cuTENSOR.cutensorDestroyOperationDescriptor(desc[])
    cuTENSOR.cutensorDestroyPlanPreference(pref[])

    # detach the workspace, see the `BlockSparsePlan` docstring
    workspacesize = sizeof(plan.workspace)
    scalartype = plan.scalar_type
    CUDA.unsafe_free!(plan.workspace)
    plan.workspace = CUDA.CuVector{UInt8, CUDA.DeviceMemory}(undef, 0)

    # all four conjugation combinations, since they are not part of the key; this decoding is
    # the inverse of `_signs_index`, a correspondence pinned by a test
    signs = ntuple(4) do i
        conjA, conjB = isodd(i - 1), (i - 1) >= 2
        return blocksparse_contract_signs(
            key.VC, key.VA, key.pA, conjA, key.VB, key.pB, conjB, key.pAB
        )
    end

    return BlockSparsePlan(plan, workspacesize, scalartype, signs, key)
end

"look `key` up in the global plan cache, creating the plan on a miss"
function blocksparse_plan(key::BlockSparseKey)
    return get!(() -> _create_plan(key), BLOCKSPARSE_PLAN_CACHE, key)
end
blocksparse_plan(cache, key::BlockSparseKey) = get!(() -> _create_plan(key), cache, key)
blocksparse_plan(::Nothing, key::BlockSparseKey) = blocksparse_plan(key)

# public API
# ----------
function plan_contract(
        ::Type{T}, VC::HomSpace, VA::HomSpace, pA::Index2Tuple,
        VB::HomSpace, pB::Index2Tuple, pAB::Index2Tuple;
        jit::Bool = false
    ) where {T}
    T <: BLOCKSPARSE_TYPES || throw(
        BlockSparseUnsupported(
            "cuTENSOR's block-sparse backend does not support scalar type $T"
        )
    )
    # a plan is the one entry point that never consults `blocksparse_reason`, so the sector
    # gate has to be applied here rather than failing much later at execution
    I = sectortype(VC)
    blocksparse_compatible(I) || throw(
        BlockSparseUnsupported(
            "sector type $I has non-trivial fusion, braiding or duality data"
        )
    )
    key = BlockSparseKey(
        VC, VA, VB, pA, pB, pAB, T, jit, CUDA.deviceid(CUDA.device())
    )
    return blocksparse_plan(key)
end

function plan_contract(
        C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple, pAB::Index2Tuple; kwargs...
    )
    return plan_contract(
        scalartype(C), space(C), space(A), pA, space(B), pB, pAB; kwargs...
    )
end

"verify `p` was built for this contraction; reusing one would silently read the wrong blocks"
function check_plan(
        p::BlockSparsePlan, C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple, pAB::Index2Tuple
    )
    k = p.key::BlockSparseKey
    # a plan bypasses `blocksparse_reason`, and the fermionic output correction writes into `C`
    # *before* the contraction, so an aliased operand would be modified rather than merely raced on
    (C === A || C === B) && throw(
        ArgumentError("the output aliases an input, which cuTENSOR does not permit")
    )
    expected = BlockSparseKey(
        space(C), space(A), space(B), pA, pB, pAB,
        scalartype(C), k.jit, CUDA.deviceid(CUDA.device())
    )
    k == expected || throw(
        ArgumentError(
            lazy"""
            this `BlockSparsePlan` was not built for this contraction:
              plan:     $(k.VA), $(k.VB) -> $(k.VC) with $(k.pA), $(k.pB), $(k.pAB) on device $(k.device)
              provided: $(expected.VA), $(expected.VB) -> $(expected.VC) with $(expected.pA), $(expected.pB), $(expected.pAB) on device $(expected.device)
            Build one plan per distinct contraction, or pass `CuTENSORBlockSparse()` to use the plan cache.
            """
        )
    )
    return nothing
end
