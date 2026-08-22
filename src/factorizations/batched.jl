"""
    BATCHED_SVD_THRESHOLD

Minimum number of blocks for which a batched driver is used.
"""
const BATCHED_SVD_THRESHOLD = 4

const BatchedSVDAlgorithm = Union{
    MAK.DivideAndConquerBatched, MAK.QRIterationBatched,
    MAK.BisectionBatched, MAK.JacobiBatched,
}

"""
    unbatched(alg)

The unbatched, one-by-one complement of a batched algorithm,
used for tensors with too few blocks to be worth batching.
"""
unbatched(::MAK.DivideAndConquerBatched) = MAK.DivideAndConquer()
unbatched(::MAK.QRIterationBatched) = MAK.QRIteration()
unbatched(::MAK.BisectionBatched) = MAK.Bisection()
unbatched(::MAK.JacobiBatched) = MAK.Jacobi()

"""
    max_batched_blocksize(alg, storagetype) -> Int

Largest block dimension the backend for `alg` accepts. Blocks exceeding it are sent
to the block-at-a-time unbatched fallback instead. Unlimited by default.
"""
max_batched_blocksize(::BatchedSVDAlgorithm, ::Type) = typemax(Int)

"""
    batched_requires_tall(alg) -> Bool

Whether the batched driver for `alg` only accepts `m >= n`. `gesvd_batched`
(the `QRIteration` algorithm) does. The unbatched fallback handles wide
matrices by decomposing the adjoint, so a wide batch is not batched (for now).
"""
batched_requires_tall(::BatchedSVDAlgorithm) = false
batched_requires_tall(::MAK.QRIterationBatched) = true

# Figure out which sectors are even worth batching, and if some share a batch size
# `uniform = true` additionally demands that every block already has that exact size, i.e.
# that no padding is needed. Full decompositions require this, compact decompositions only
# read back the leading `min(m, n)` columns, which padding out with zero doesn't affect.
function _batchable(t::AbstractTensorMap, alg::BatchedSVDAlgorithm, uniform::Bool = false)
    cs = collect(blocksectors(t))
    isempty(cs) && return cs, (0, 0)
    szs = [size(block(t, c)) for c in cs]
    m, n = maximum(first, szs), maximum(last, szs)
    lim = max_batched_blocksize(alg, storagetype(t))
    (length(cs) < BATCHED_SVD_THRESHOLD || m > lim || n > lim) && return empty(cs), (m, n)
    # The *padded* batch is (m, n) even if *individual* blocks are tall
    # so check the padded shape rather than the blocks'.
    (batched_requires_tall(alg) && m < n) && return empty(cs), (m, n)
    (uniform && !all(isequal((m, n)), szs)) && return empty(cs), (m, n)
    return cs, (m, n)
end

# The batched solvers work on 3D arrays: input `(m, n, nbatch)`, and correspondingly
# `U = (m, minmn, nbatch)`, `S = (minmn, nbatch)`, `Vᴴ = (minmn, n, nbatch)`.
function _pack(t::AbstractTensorMap, cs, m, n)
    b1 = block(t, first(cs))
    A = similar(b1, m, n, length(cs))
    fill!(A, zero(scalartype(A)))
    for (i, c) in enumerate(cs)
        b = block(t, c)
        copyto!(view(A, axes(b, 1), axes(b, 2), i), b)
    end
    return A
end

for f! in (:svd_compact!, :svd_full!)
    full = f! === :svd_full!
    @eval function MAK.$f!(t::AbstractTensorMap, F, alg::BatchedSVDAlgorithm)
        U, S, Vᴴ = F
        cs, (m, n) = _batchable(t, alg, $full)
        if isempty(cs)  # not worth batching, or the library doesn't support these sizes
            alg′ = unbatched(alg)
            foreachblock(t, U, S, Vᴴ) do _, (b, u, sv, v)
                MAK.$f!(b, (u, sv, v), alg′)
                return nothing
            end
            return F
        end
        nb, minmn = length(cs), min(m, n)
        A = _pack(t, cs, m, n)
        Ub = similar(A, m, $full ? m : minmn, nb)
        rT = real(scalartype(t))
        Sb = $full ? similar(A, rT, m, n, nb) :
            similar(diagview(block(S, first(cs))), minmn, nb)
        Vb = similar(A, $full ? n : minmn, n, nb)
        $full && fill!(Sb, zero(rT))
        MAK.$f!(A, (Ub, Sb, Vb), alg)
        for (i, c) in enumerate(cs)
            u, sv, v = block(U, c), block(S, c), block(Vᴴ, c)
            copyto!(u, view(Ub, axes(u, 1), axes(u, 2), i))
            if $full
                copyto!(sv, view(Sb, axes(sv, 1), axes(sv, 2), i))
            else
                copyto!(diagview(sv), view(Sb, axes(diagview(sv), 1), i))
            end
            copyto!(v, view(Vb, axes(v, 1), axes(v, 2), i))
        end
        return F
    end
end

function MAK.svd_vals!(t::AbstractTensorMap, S, alg::BatchedSVDAlgorithm)
    cs, (M, N) = _batchable(t, alg)
    if isempty(cs)
        alg′ = unbatched(alg)
        foreachblock(t, S) do _, (b, sv)
            MAK.svd_vals!(b, sv, alg′)
            return nothing
        end
        return S
    end
    nb, K = length(cs), min(M, N)
    A = _pack(t, cs, M, N)
    Sb = similar(block(S, first(cs)), K, nb)
    MAK.svd_vals!(A, Sb, alg)
    for (i, c) in enumerate(cs)
        sv = block(S, c)
        copyto!(sv, view(Sb, axes(sv, 1), i))
    end
    return S
end

"""
    batched_algorithm(alg, storagetype) -> alg

Batched counterpart of `alg` for tensors stored in `storagetype`, or `alg` itself when
batching does not apply. Selecting a batched algorithm here is safe regardless of how many
blocks a tensor has: `_batchable` falls back to the block-at-a-time driver below
`BATCHED_SVD_THRESHOLD`, so this only decides whether batching is *considered*.

The GPUArrays extension opts GPU-backed tensors in. On CPU each block decomposition is one
LAPACK call with no launch overhead to amortize, so batching there would only add packing
and padding work.
"""
batched_algorithm(alg, ::Type) = alg

function _tensor_algorithm(
        f!::Union{typeof(MAK.svd_compact!), typeof(MAK.svd_full!), typeof(MAK.svd_vals!)},
        ::Type{T}; kwargs...
    ) where {T <: AbstractTensorMap}
    alg = MAK.default_algorithm(f!, blocktype(T); kwargs...)
    return batched_algorithm(alg, storagetype(T))
end
