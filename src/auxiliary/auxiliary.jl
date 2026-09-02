function linearizepermutation(
        p1::NTuple{N₁, Int}, p2::NTuple{N₂}, n₁::Int, n₂::Int
    ) where {N₁, N₂}
    p1′ = ntuple(Val(N₁)) do n
        return p1[n] > n₁ ? n₂ + 2n₁ + 1 - p1[n] : p1[n]
    end
    p2′ = ntuple(Val(N₂)) do n
        return p2[N₂ + 1 - n] > n₁ ? n₂ + 2n₁ + 1 - p2[N₂ + 1 - n] : p2[N₂ + 1 - n]
    end
    return (p1′..., p2′...)
end

function permutation2swaps(perm)
    p = collect(perm)
    @assert isperm(p)
    swaps = Vector{Int}()
    N = length(p)
    for k in 1:(N - 1)
        append!(swaps, (p[k] - 1):-1:k)
        for l in (k + 1):N
            if p[l] < p[k]
                p[l] += 1
            end
        end
        p[k] = k
    end
    return swaps
end

# one-argument version: check whether `p` is a cyclic permutation (of `1:length(p)`)
function iscyclicpermutation(p)
    N = length(p)
    @inbounds for i in 1:N
        p[mod1(i + 1, N)] == mod1(p[i] + 1, N) || return false
    end
    return true
end
# two-argument version: check whether `v1` is a cyclic permutation of `v2`
function iscyclicpermutation(v1, v2)
    length(v1) == length(v2) || return false
    return iscyclicpermutation(indexin(v1, v2))
end

_kron(A, B, C, D...) = _kron(_kron(A, B), C, D...)
function _kron(A, B)
    sA = size(A)
    sB = size(B)
    s = map(*, sA, sB)
    C = similar(A, promote_type(eltype(A), eltype(B)), s)
    for IA in eachindex(IndexCartesian(), A)
        for IB in eachindex(IndexCartesian(), B)
            I = CartesianIndex(IB.I .+ (IA.I .- 1) .* sB)
            C[I] = A[IA] * B[IB]
        end
    end
    return C
end

@noinline _boundserror(P, i) = throw(BoundsError(P, i))
@noinline _nontrivialspaceerror(P, i) = throw(ArgumentError(lazy"Attempting to remove a non-trivial space $(P[i])"))

const VecOrNumber{T <: Number} = Union{DenseVector{T}, T}

"""
    _interleave(a::NTuple{N}, b::NTuple{N}) -> NTuple{2N}

Interleave two tuples of the same length.
"""
_interleave(::Tuple{}, ::Tuple{}) = ()
function _interleave(a::NTuple{N}, b::NTuple{N}) where {N}
    return (a[1], b[1], _interleave(tail(a), tail(b))...)
end

@static if VERSION < v"1.11" # TODO: remove once support for v1.10 is dropped
    _allequal(f, xs) = allequal(Base.Generator(f, xs))
else
    _allequal(f, xs) = allequal(f, xs)
end

"""
    _check_levels(levels::Tuple) -> Nothing

Check whether all elements of a tuple are distinct, throwing an `ArgumentError` if not.
This is used to validate the `levels` of [`braid`](@ref).
For symmetric braidings, this check is skipped since the levels are irrelevant.   
"""
_check_levels(::SymmetricBraiding, levels, s) = nothing
@inline function _check_levels(::BraidingStyle, levels, s)
    levels[s] == levels[s + 1] && _throw_ambiguous_levels(levels[s])
    return nothing
end
@noinline function _throw_ambiguous_levels(l)
    throw(ArgumentError(lazy"ambiguous braid: two indices with equal level $l have to cross"))
end

"""
    taskforeach(f, items, ntasks::Integer) -> Nothing
    taskforeach(f, items, resources) -> Nothing

Apply `f(item)` (respectively `f(item, resource)`) to all elements of `items`, distributing the work over at most `ntasks` workers
(respectively one worker per entry of the indexable collection `resources`), with dynamic load balancing through an atomic counter.
The calling thread acts as the first of the workers, so at most `ntasks - 1` tasks are spawned, and none at all for a single worker.
"""
function taskforeach(f, items, ntasks::Integer)
    items′ = items isa AbstractArray ? items : collect(items)
    n = length(items′)
    nworkers = min(ntasks, n)
    nworkers == 0 && return nothing
    counter = Threads.Atomic{Int}(1)
    Threads.@sync begin
        for _ in 2:nworkers
            Threads.@spawn _taskforeach_worker(f, items′, counter, n)
        end
        _taskforeach_worker(f, items′, counter, n)
    end
    return nothing
end
function taskforeach(f, items, resources)
    items′ = items isa AbstractArray ? items : collect(items)
    n = length(items′)
    nworkers = min(length(resources), n)
    nworkers == 0 && return nothing
    counter = Threads.Atomic{Int}(1)
    Threads.@sync begin
        for j in 2:nworkers
            let r = @inbounds resources[j]
                Threads.@spawn _taskforeach_worker(Base.Fix2(f, r), items′, counter, n)
            end
        end
        _taskforeach_worker(Base.Fix2(f, @inbounds(resources[1])), items′, counter, n)
    end
    return nothing
end

function _taskforeach_worker(f, items, counter, n::Int)
    while true
        i = Threads.atomic_add!(counter, 1)
        i > n && break
        f(@inbounds(items[i]))
    end
    return nothing
end
