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

# Low-overhead implementation of `copyto!` for specific case of `stride(B, 1) < stride(B, 2)`
# used in indexmanipulations: avoids the overhead of Strided.jl
function _copyto!(A::StridedView{<:Any, 1}, B::StridedView{<:Any, 2})
    length(A) == length(B) || throw(DimensionMismatch())
    @assert A.op === identity

    Adata = parent(A)
    Astr = stride(A, 1)
    IA = A.offset

    Bdata = parent(B)
    Bstr = strides(B)

    IB_1 = B.offset
    @inbounds for _ in axes(B, 2)
        IB = IB_1
        for _ in axes(B, 1)
            Adata[IA += Astr] = B.op(Bdata[IB += Bstr[1]])
        end
        IB_1 += Bstr[2]
    end

    return A
end

@static if VERSION < v"1.11" # TODO: remove once support for v1.10 is dropped
    _allequal(f, xs) = allequal(Base.Generator(f, xs))
else
    _allequal(f, xs) = allequal(f, xs)
end
