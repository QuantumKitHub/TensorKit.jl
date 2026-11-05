const CuStoredTensorMap{T} = TensorMap{T, S, N₁, N₂, A} where {S, N₁, N₂, A <: CuVector{T}}
const CuMulOperand{T} = Union{
    CuStoredTensorMap{T},
    AdjointTensorMap{T, S, N₁, N₂, TT} where {S, N₁, N₂, TT <: CuStoredTensorMap{T}},
}

"""
    GROUPED_GEMM_THRESHOLD

Blocks with `m * n * k` at least `GROUPED_GEMM_THRESHOLD[].real` (real element types) or
`GROUPED_GEMM_THRESHOLD[].complex` (complex element types) are multiplied with individual gemm
calls in `mul!` for CUDA tensors, all smaller blocks are combined into a single grouped gemm call.
Complex blocks benefit from individual calls much earlier, since the grouped call works on their
real 1m embedding.
"""
const GROUPED_GEMM_THRESHOLD = Ref((real = 1024^3, complex = 256^3))

const _HAS_GROUPED_GEMM = Ref{Union{Nothing, Bool}}(nothing)
function has_grouped_gemm()
    isnothing(_HAS_GROUPED_GEMM[]) && (_HAS_GROUPED_GEMM[] = cuBLAS.version() >= v"12.5")
    return _HAS_GROUPED_GEMM[]::Bool
end

_parent_adj(t::TensorMap) = (t, false)
_parent_adj(t::AdjointTensorMap) = (parent(t), true)

"""
    GroupedGemmPlan

Everything about `mul!(C, op(A), op(B))` for CUDA tensors that only depends on the spaces, with
`op` either `identity` or `adjoint`:

- `dims` (`m, n, k`) and `offsets` (into the data of `A`, `B` and `C`), one row per multiplied
  block, where `op(A)` is `m × k` and `op(B)` is `k × n`. The first `ngrouped` rows go into a
  single grouped gemm call, the remaining ones are large blocks for individual gemm calls.
- `emptyC`: ranges of blocks of `C` without a matching block in `A` or `B`, which are only scaled.
- `transa`, `transb` and `gdims`: the arguments of the grouped gemm call. For complex element
  types, this call acts on real 1m embeddings of the blocks: `op(A)` is expanded into a buffer of
  length `lenAR`, using the per-block columns `offA, m, k, offAR` of `meta1m`.

Plans are cached by `groupedgemmplan`, since building one requires the block structures of three
spaces and allocates many small arrays, which would otherwise dominate for tensors with many
small blocks.
"""
struct GroupedGemmPlan
    ngrouped::Int
    dims::Matrix{Int64}
    offsets::Matrix{Int64}
    emptyC::Vector{UnitRange{Int}}
    transa::cuBLAS.cublasOperation_t
    transb::cuBLAS.cublasOperation_t
    gdims::Matrix{Int64}
    meta1m::Matrix{Int64}
    lenAR::Int
end

@cached function groupedgemmplan(
        VA::TensorMapSpace, VB::TensorMapSpace, adjA::Bool, adjB::Bool, cplx::Bool, threshold::Int
    )::GroupedGemmPlan
    VC = compose(adjA ? VA' : VA, adjB ? VB' : VB)
    bsA, bsB, bsC = blockstructure(VA), blockstructure(VB), blockstructure(VC)
    blocks = NTuple{6, Int}[]
    emptyC = UnitRange{Int}[]
    for (c, (_, rC)) in pairs(bsC)
        if !(haskey(bsA, c) && haskey(bsB, c))
            push!(emptyC, rC)
            continue
        end
        (dA, rA), (dB, rB) = bsA[c], bsB[c]
        m, k = adjA ? reverse(dA) : dA
        n = adjB ? first(dB) : last(dB)
        push!(blocks, (m, n, k, first(rA) - 1, first(rB) - 1, first(rC) - 1))
    end
    isdirect(b) = b[1] * b[2] * b[3] >= threshold
    sort!(blocks; by = isdirect)
    G = count(!isdirect, blocks)
    table = [b[j] for b in blocks, j in 1:6]
    dims, offsets = table[:, 1:3], table[:, 4:6]

    gdims = dims[1:G, :]
    if cplx # 1m: op(A) is expanded explicitly, B and C are reinterpreted
        gdims[:, 1] .*= 2
        gdims[:, 3] .*= 2
        adjA = adjB = false
    end
    lengths = 4 .* dims[1:G, 1] .* dims[1:G, 3]
    offAR = cumsum(lengths) .- lengths
    meta1m = cplx ? [offsets[1:G, 1] dims[1:G, 1] dims[1:G, 3] offAR] : zeros(Int64, 0, 4)
    return GroupedGemmPlan(
        G, dims, offsets, emptyC,
        adjA ? cuBLAS.CUBLAS_OP_T : cuBLAS.CUBLAS_OP_N,
        adjB ? cuBLAS.CUBLAS_OP_T : cuBLAS.CUBLAS_OP_N,
        gdims, meta1m, sum(lengths; init = 0)
    )
end

function LinearAlgebra.mul!(
        tC::CuStoredTensorMap{T}, tA::CuMulOperand{T}, tB::CuMulOperand{T}, α::Number, β::Number
    ) where {T <: LinearAlgebra.BlasFloat}
    compose(space(tA), space(tB)) == space(tC) ||
        throw(SpaceMismatch(lazy"$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))
    pA, adjA = _parent_adj(tA)
    pB, adjB = _parent_adj(tB)
    α′, β′ = T(TensorKit._blasscalar(α)), T(TensorKit._blasscalar(β))

    # the 1m embedding does not support a conjugated B
    threshold = !has_grouped_gemm() || (T <: Complex && adjB) ? 0 :
        T <: Complex ? GROUPED_GEMM_THRESHOLD[].complex : GROUPED_GEMM_THRESHOLD[].real
    plan = groupedgemmplan(space(pA), space(pB), adjA, adjB, T <: Complex, threshold)

    # the grouped 1m gemm is real, so complex β is applied upfront
    if !isreal(β′)
        rmul!(tC.data, β′)
        β′ = one(T)
    else
        for r in plan.emptyC
            v = view(tC.data, r)
            iszero(β′) ? fill!(v, zero(T)) : isone(β′) || rmul!(v, β′)
        end
    end

    _direct_gemm!(tC.data, pA.data, pB.data, plan, adjA, adjB, α′, β′)
    _grouped_gemm!(tC.data, pA.data, pB.data, plan, adjA, α′, β′)
    return tC
end

function _direct_gemm!(
        C::CuVector{T}, A::CuVector{T}, B::CuVector{T}, plan::GroupedGemmPlan,
        adjA::Bool, adjB::Bool, α::T, β::T
    ) where {T}
    plan.ngrouped == size(plan.dims, 1) && return C
    # upload scalars once instead of on every gemm call
    dα, dβ = CuRef{T}(α), CuRef{T}(β)
    adjop = T <: Complex ? cuBLAS.CUBLAS_OP_C : cuBLAS.CUBLAS_OP_T
    opA = adjA ? adjop : cuBLAS.CUBLAS_OP_N
    opB = adjB ? adjop : cuBLAS.CUBLAS_OP_N
    dt = convert(cuBLAS.cudaDataType_t, T)
    ct = cuBLAS.gemmExComputeType(T, T, T, 1, 1, 1)
    h = cuBLAS.handle()
    for i in (plan.ngrouped + 1):size(plan.dims, 1)
        m, n, k = view(plan.dims, i, :)
        offA, offB, offC = view(plan.offsets, i, :)
        cuBLAS.cublasGemmEx_64(
            h, opA, opB, m, n, k, dα,
            pointer(A, offA + 1), dt, adjA ? k : m,
            pointer(B, offB + 1), dt, adjB ? n : k,
            dβ, pointer(C, offC + 1), dt, m, ct, cuBLAS.CUBLAS_GEMM_DEFAULT
        )
    end
    return C
end

# reduced precision compute types (e.g. TF32) crash on H100 with misaligned blocks
_grouped_computetype(::Type{Float32}) = cuBLAS.CUBLAS_COMPUTE_32F
_grouped_computetype(::Type{Float64}) = cuBLAS.CUBLAS_COMPUTE_64F

function _grouped_gemm!(
        C::CuVector{T}, A::CuVector{T}, B::CuVector{T}, plan::GroupedGemmPlan,
        adjA::Bool, α::T, β::T
    ) where {T}
    G = plan.ngrouped
    G == 0 && return C
    R = real(T)
    AR = T <: Complex ? CuVector{R}(undef, plan.lenAR) : A
    offAR = T <: Complex ? view(plan.meta1m, :, 4) : view(plan.offsets, :, 1)
    # device pointers to the blocks of A (or AR), B and C, followed by the 1m metadata
    ptrs = Matrix{Int64}(undef, G, 3 + size(plan.meta1m, 2))
    for i in 1:G
        ptrs[i, 1] = UInt(pointer(AR, offAR[i] + 1))
        ptrs[i, 2] = UInt(pointer(B, plan.offsets[i, 2] + 1))
        ptrs[i, 3] = UInt(pointer(C, plan.offsets[i, 3] + 1))
    end
    T <: Complex && (ptrs[:, 4:end] = plan.meta1m)
    dptrs = CuArray(ptrs)
    if T <: Complex
        maxlen = maximum(i -> plan.dims[i, 1] * plan.dims[i, 3], 1:G)
        blocks = (min(cld(maxlen, 256), 1024), min(G, 65535))
        @cuda threads = 256 blocks = blocks _expand1m_kernel!(AR, A, dptrs, G, α, adjA)
        α = one(T)
    end

    base = UInt(pointer(dptrs))
    gm, gn, gk = view(plan.gdims, :, 1), view(plan.gdims, :, 2), view(plan.gdims, :, 3)
    lda = plan.transa == cuBLAS.CUBLAS_OP_T ? gk : gm
    ldb = plan.transb == cuBLAS.CUBLAS_OP_T ? gn : gk
    dt = convert(cuBLAS.cudaDataType_t, R)
    h = cuBLAS.handle()
    # grouped gemm only supports host pointers for the scalars
    cuBLAS.cublasSetPointerMode_v2(h, cuBLAS.CUBLAS_POINTER_MODE_HOST)
    try
        cuBLAS.cublasGemmGroupedBatchedEx_64(
            h, fill(plan.transa, G), fill(plan.transb, G), gm, gn, gk, fill(real(α), G),
            Ptr{Ptr{Cvoid}}(base), dt, lda,
            Ptr{Ptr{Cvoid}}(base + 8G), dt, ldb,
            fill(real(β), G), Ptr{Ptr{Cvoid}}(base + 16G), dt, gm,
            G, ones(Int64, G), _grouped_computetype(R)
        )
    finally
        cuBLAS.cublasSetPointerMode_v2(h, cuBLAS.CUBLAS_POINTER_MODE_DEVICE)
    end
    CUDA.unsafe_free!(dptrs)
    T <: Complex && CUDA.unsafe_free!(AR)
    return C
end

# writes the 1m embedding [ar -ai; ai ar] of α * op(A) for every grouped block
function _expand1m_kernel!(AR, A, meta, G, α, adjA)
    b = blockIdx().y
    while b <= G
        @inbounds offA, m, k, offAR = meta[b, 4], meta[b, 5], meta[b, 6], meta[b, 7]
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
        while idx < m * k
            r, s = idx % m, idx ÷ m
            @inbounds a = adjA ? conj(A[offA + s + r * k + 1]) : A[offA + idx + 1]
            re, im = reim(α * a)
            p = offAR + 2r + 4m * s + 1
            @inbounds AR[p] = re
            @inbounds AR[p + 1] = im
            @inbounds AR[p + 2m] = -im
            @inbounds AR[p + 2m + 1] = re
            idx += gridDim().x * blockDim().x
        end
        b += gridDim().y
    end
    return nothing
end
