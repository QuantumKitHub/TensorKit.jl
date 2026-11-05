# Randomized truncated SVD
# ------------------------
# A randomized SVD (e.g. `CUSOLVER_Randomized`) only computes the largest `k` singular triplets of
# a matrix. For a tensor, `k` is needed *per block* before the decomposition. We derive it from
# the truncation strategy: each block is decomposed into a reduced space that is guaranteed to
# contain everything the strategy keeps, after which the strategy is applied as usual.

const RandomizedSVDAlgorithm = TruncatedAlgorithm{<:MAK.GPU_Randomized}

_keeps_largest(strategy) = strategy.rev && (strategy.by === abs || strategy.by === identity)

"""
    _svd_rank_bound(strategy, c, n)

Upper bound on the number of singular values that `strategy` keeps in the block of sector `c`,
which has `n` singular values in total, or `nothing` if the strategy does not bound it (or does
not keep the largest values).
"""
_svd_rank_bound(::TruncationStrategy, c, n) = nothing
function _svd_rank_bound(strategy::TruncationByOrder, c, n)
    _keeps_largest(strategy) || return nothing
    # values in block `c` are weighted by `dim(c)`
    return min(n, floor(Int, strategy.howmany / dim(c)))
end
function _svd_rank_bound(strategy::TruncationSpace, c, n)
    _keeps_largest(strategy) || return nothing
    return min(n, dim(strategy.space, c))
end
function _svd_rank_bound(strategy::TruncationIntersection, c, n)
    bounds = map(s -> _svd_rank_bound(s, c, n), strategy.components)
    all(isnothing, bounds) && return nothing
    return minimum(b for b in bounds if !isnothing(b))
end
function _svd_rank_bound(strategy::TruncationUnion, c, n)
    bounds = map(s -> _svd_rank_bound(s, c, n), strategy.components)
    any(isnothing, bounds) && return nothing
    return maximum(bounds)
end

# use `k` values per sector, taken from the truncation strategy and,
# if `k` is provided by the algorithm, capped by that
function _randomized_svd_space(t::AbstractTensorMap, alg::RandomizedSVDAlgorithm)
    V = infimum(fuse(codomain(t)), fuse(domain(t)))
    kmax = get(alg.alg.kwargs, :k, nothing)
    inds = SectorDict(c => Base.OneTo(_randomized_svd_rank(alg.trunc, kmax, c, dim(V, c))) for c in sectors(V))
    return truncate_space(V, inds)
end
function _randomized_svd_rank(trunc, kmax, c, n)
    k = _svd_rank_bound(trunc, c, n)
    if isnothing(kmax)
        isnothing(k) && throw(
            ArgumentError(
                "randomized SVD needs the number of singular values to compute: use a truncation strategy that bounds the rank (`truncrank`, `truncspace`, or an intersection with one of these), or pass `k` to the algorithm"
            )
        )
        return k
    end
    return isnothing(k) ? min(kmax, n) : min(k, kmax)
end

function MAK.initialize_output(
        ::Union{typeof(svd_trunc!), typeof(svd_trunc_no_error!)}, t::AbstractTensorMap,
        alg::RandomizedSVDAlgorithm
    )
    @timeit_debug GLOBAL_TIMER "alloc: initialize_output" begin
        Vk = _randomized_svd_space(t, alg)
        U = similar(t, codomain(t) ← Vk)
        S = similar_diagonal(t, real(scalartype(t)), Vk)
        Vᴴ = similar(t, Vk ← domain(t))
        return U, S, Vᴴ
    end
end

function MAK.svd_trunc_no_error!(t::AbstractTensorMap, USVᴴ, alg::RandomizedSVDAlgorithm)
    U, S, Vᴴ = USVᴴ
    kwargs = Base.structdiff(alg.alg.kwargs, NamedTuple{(:k, :p, :fixgauge)})
    fixgauge = get(alg.alg.kwargs, :fixgauge, MAK.default_fixgauge())::Bool
    @timeit_debug GLOBAL_TIMER "svd_trunc!" begin
        for (c, Sb) in blocks(S)
            tb, Ub, Vᴴb = block(t, c), block(U, c), block(Vᴴ, c)
            s = diagview(Sb)
            k = length(s)
            k == 0 && continue
            n = min(size(tb)...)
            p = get(alg.alg.kwargs, :p, n - k - 1)
            if p > 0 && k + p < n
                MAK._gpu_Xgesvdr!(tb, s, Ub, Vᴴb; k, p, kwargs...)
            else # the requested oversampling does not fit in this block
                Uf, Sf, Vᴴf = svd_compact!(tb, MAK.select_algorithm(svd_compact!, tb, nothing))
                copyto!(Ub, view(Uf, :, 1:k))
                copyto!(s, view(diagview(Sf), 1:k))
                copyto!(Vᴴb, view(Vᴴf, 1:k, :))
            end
            fixgauge && MAK.gaugefix!(svd_trunc!, Ub, Vᴴb)
        end
    end
    USVᴴtrunc, _ = MAK.truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)
    return USVᴴtrunc
end

function MAK.svd_trunc!(t::AbstractTensorMap, USVᴴ, alg::RandomizedSVDAlgorithm)
    normt = norm(t)
    Ũ, S̃, Ṽᴴ = svd_trunc_no_error!(t, USVᴴ, alg)
    normS = norm(S̃)
    ϵ = sqrt((normt + normS) * abs(normt - normS))
    return Ũ, S̃, Ṽᴴ, ϵ
end

function MAK.initialize_output(
        ::typeof(svd_trunc!), t::AdjointTensorMap, alg::RandomizedSVDAlgorithm
    )
    return reverse(adjoint.(MAK.initialize_output(svd_trunc!, adjoint(t), _adjoint(alg))))
end
function MAK.svd_trunc!(t::AdjointTensorMap, F, alg::RandomizedSVDAlgorithm)
    U, S, Vᴴ, ϵ = svd_trunc!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
    return Vᴴ', S, U', ϵ
end
