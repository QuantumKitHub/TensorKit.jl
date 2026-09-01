# Applicability, dispatch and execution
# -------------------------------------

"""
    blocksparse_reason(C, A, pA, conjA, B, pB, conjB, pAB) -> Union{Nothing, String}

Why this contraction cannot be done with cuTENSOR's block-sparse backend, or `nothing` if it
can. Returning the reason rather than a `Bool` is what lets `strict = true` and the `@debug`
message say something useful. Tensor *types* are handled by dispatch, not here.
"""
function blocksparse_reason(
        C::CuTensorMapAny, A::CuTensorMapAny, pA::Index2Tuple, conjA::Bool,
        B::CuTensorMapAny, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple
    )
    T = scalartype(C)
    T <: BLOCKSPARSE_TYPES || return "unsupported scalar type $T"
    (scalartype(A) === T && scalartype(B) === T) ||
        return "mixed scalar types ($(scalartype(A)), $(scalartype(B)), $T)"

    I = sectortype(C)
    blocksparse_compatible(I) ||
        return "sector type $I has non-trivial fusion, braiding or duality data"

    numind(A) <= MAX_BLOCKSPARSE_MODES || return "A has more than $MAX_BLOCKSPARSE_MODES modes"
    numind(B) <= MAX_BLOCKSPARSE_MODES || return "B has more than $MAX_BLOCKSPARSE_MODES modes"
    numind(C) <= MAX_BLOCKSPARSE_MODES || return "C has more than $MAX_BLOCKSPARSE_MODES modes"

    isempty(pA[2]) && return "outer products are not supported"
    numind(C) == 0 && return "scalar output is not supported"
    (dim(A) > 0 && dim(B) > 0 && dim(C) > 0) || return "empty tensor"

    # aliasing: cuTENSOR requires that A and B do not overlap the elements written to D
    (C === A || C === B) && return "the output aliases an input"

    # remaining layout validation, run here rather than in `bs_descriptor` so that a rejected
    # space produces a fallback rather than an error
    for t in (C, A, B)
        try
            blocksparsestructure(space(t))
        catch e
            e isa BlockSparseUnsupported && return e.msg
            rethrow()
        end
    end
    return nothing
end

# anything that is not a CUDA-backed `TensorMap` is out of scope by dispatch
function blocksparse_reason(
        C::AbstractTensorMap, A::AbstractTensorMap, ::Index2Tuple, ::Bool,
        B::AbstractTensorMap, ::Index2Tuple, ::Bool, ::Index2Tuple
    )
    return "unsupported tensor types $(typeof(C)), $(typeof(A)), $(typeof(B)); " *
        "the block-sparse backend requires CUDA-backed `TensorMap`s"
end

"whether this contraction will use the block-sparse backend, rather than falling back silently"
blocksparse_supported(args...) = isnothing(blocksparse_reason(args...))

# dispatch
# --------
# strictly more specific than the generic `_tensorcontract!` in C, A, B and `backend` at once
function _tensorcontract!(
        C::CuTensorMapAny,
        A::CuTensorMapAny, pA::Index2Tuple, conjA::Bool,
        B::CuTensorMapAny, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number,
        backend::CuTENSORBlockSparse, allocator
    )
    reason = blocksparse_reason(C, A, pA, conjA, B, pB, conjB, pAB)
    if isnothing(reason)
        key = BlockSparseKey(
            space(C), space(A), space(B), pA, pB, pAB,
            scalartype(C), false, CUDA.deviceid(CUDA.device())
        )
        # honour a backend-carried cache, so plan lifetime can be scoped to a hot loop
        plan = blocksparse_plan(backend.plans, key)
        return blocksparse_contract!(
            C, A, pA, conjA, B, pB, conjB, pAB, α, β, plan, allocator
        )
    end
    backend.strict && throw(BlockSparseUnsupported(reason))
    @debug "falling back from the block-sparse backend" reason
    return _generic_tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend.fallback, allocator
    )
end

# a plan is a fully specialized backend
function _tensorcontract!(
        C::CuTensorMapAny,
        A::CuTensorMapAny, pA::Index2Tuple, conjA::Bool,
        B::CuTensorMapAny, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number,
        plan::BlockSparsePlan, allocator
    )
    @boundscheck check_plan(plan, C, A, pA, B, pB, pAB)
    return blocksparse_contract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, plan, allocator
    )
end

# execution
# ---------
"""
    blocksparse_contract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, plan, allocator)

Execute `C = α ⋅ opA(A) ⋅ opB(B) + β ⋅ C` as a single block-sparse contraction.

Because `pA`, `pB` and `pAB` only determine cuTENSOR mode labels, no permutation, fusion tree
transformation or temporary is needed for bosonic sectors, with two exceptions: cuTENSOR
accepts only `OP_IDENTITY`, so a complex operand needing conjugation is materialized first,
and a fermionic sector needs a per-block sign, see
[`TensorKit.blocksparse_contract_signs`](@ref).
"""
function blocksparse_contract!(
        C::CuTensorMapAny,
        A::CuTensorMapAny, pA::Index2Tuple, conjA::Bool,
        B::CuTensorMapAny, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number,
        plan::BlockSparsePlan, allocator
    )
    T = scalartype(C)
    # precomputed with the plan, and `nothing` in every field when nothing needs correcting
    signs = @inbounds plan.signs[_signs_index(conjA, conjB)]
    needconjA = conjA && T <: Complex
    needconjB = conjB && T <: Complex
    # copied rather than scaled in place because `A === B` is legal: with distinct `signs.A`
    # and `signs.B` an in-place scaling of one would corrupt the other
    copyA = needconjA || !isnothing(signs.A)
    copyB = needconjB || !isnothing(signs.B)

    A′ = copyA ? _scratch_operand(A, needconjA, signs.A, allocator) : A
    B′ = copyB ? _scratch_operand(B, needconjB, signs.B, allocator) : B
    try
        if isnothing(signs.C)
            _execute!(plan, C, A′, B′, α, β)
        else
            # `β⋅C + α⋅sC ⊙ naive == sC ⊙ (β⋅sC⁻¹ ⊙ C + α⋅naive)`: correcting without a temporary
            prescaled = !iszero(β)
            prescaled && _scale_subblocks!(C, signs.C, true)
            try
                _execute!(plan, C, A′, B′, α, β)
            catch
                # undo the pre-scaling, but *only* if it ran: a failed `β == 0` call may never
                # have touched the data at all
                prescaled && _scale_subblocks!(C, signs.C)
                rethrow()
            end
            _scale_subblocks!(C, signs.C)
        end
    finally
        copyA && TO.tensorfree!(A′, allocator)
        copyB && TO.tensorfree!(B′, allocator)
    end
    return C
end

"materialize `conj(t)` and/or `signs ⊙ t` into a scratch tensor taken from `allocator`"
function _scratch_operand(t::CuTensorMapAny, doconj::Bool, signs, allocator)
    out = TO.tensoralloc(typeof(t), space(t), Val(true), allocator)
    # the space is unchanged, so the plan and its descriptors still apply
    if isnothing(signs)
        # nothing to scale, so a single whole-array launch beats one launch per subblock
        doconj ? (out.data .= conj.(t.data)) : copyto!(out.data, t.data)
    else
        _scale_subblocks!(out, t, signs, doconj)
    end
    return out
end

function _execute!(
        plan::BlockSparsePlan, C::CuTensorMapAny, A::CuTensorMapAny, B::CuTensorMapAny,
        α::Number, β::Number
    )
    ptrsA = block_pointers(A)
    ptrsB = block_pointers(B)
    ptrsC = block_pointers(C)

    ST = plan.scalartype
    αref = Ref{ST}(α)
    βref = Ref{ST}(β)
    wssize = plan.workspacesize

    if iszero(wssize)
        _blocksparse_ccall!(plan, αref, ptrsA, ptrsB, βref, ptrsC, CuPtr{Cvoid}(0), 0)
    else
        with_workspace(wssize) do workspace
            _blocksparse_ccall!(
                plan, αref, ptrsA, ptrsB, βref, ptrsC, pointer(workspace), wssize
            )
        end
    end
    return C
end

# called directly rather than through `cuTENSOR.contractBS!`: that helper materializes a view
# per block, uses the workspace we detached, and mistypes `workspace` as `Ptr{CuPtr{Cvoid}}`
function _blocksparse_ccall!(plan, αref, ptrsA, ptrsB, βref, ptrsC, ws, wssize)
    GC.@preserve ptrsA ptrsB ptrsC αref βref begin
        status = @ccall cuTENSOR.libcutensor.cutensorBlockSparseContract(
            cuTENSOR.handle()::cuTENSOR.cutensorHandle_t,
            plan.plan::cuTENSOR.cutensorPlan_t,
            αref::Ptr{Cvoid}, ptrsA::Ptr{CuPtr{Cvoid}}, ptrsB::Ptr{CuPtr{Cvoid}},
            βref::Ptr{Cvoid}, ptrsC::Ptr{CuPtr{Cvoid}}, ptrsC::Ptr{CuPtr{Cvoid}},
            ws::CuPtr{Cvoid}, UInt64(wssize)::UInt64,
            CUDA.stream()::cuTENSOR.cudaStream_t
        )::cuTENSOR.cutensorStatus_t
        status == cuTENSOR.CUTENSOR_STATUS_SUCCESS ||
            throw(cuTENSOR.CUTENSORError(status))
    end
    return nothing
end
