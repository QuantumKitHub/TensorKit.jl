# planar versions of tensor operations add!, trace! and contract!

# insert default backend
function planaradd!(C, A, p::Index2Tuple, α::Number, β::Number)
    return planaradd!(C, A, p, α, β, TO.DefaultBackend())
end
# insert default allocator
function planaradd!(C, A, p::Index2Tuple, α::Number, β::Number, backend)
    return planaradd!(C, A, p, α, β, backend, TO.DefaultAllocator())
end
# replace default backend with select_backend mechanism
function planaradd!(C, A, p::Index2Tuple, α::Number, β::Number, backend, allocator)
    if backend isa TO.DefaultBackend
        backend = TO.select_backend(planaradd!, C, A)
        return planaradd!(C, A, p, α, β, backend, allocator)
    elseif backend isa TO.NoBackend
        # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("No suitable backend found for planaradd! and tensor types $TC and $TA"))
    else
        # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("Unknown backend $backend for planaradd! and tensor types $TC and $TA"))
    end
end
# implementation
function planaradd!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, p::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    return transpose!(C, A, p, α, β, backend, allocator)
end

# insert default backend
function planartrace!(C, A, p::Index2Tuple, q::Index2Tuple, α::Number, β::Number)
    return planartrace!(C, A, p, q, α, β, TO.DefaultBackend())
end
# insert default allocator
function planartrace!(C, A, p::Index2Tuple, q::Index2Tuple, α::Number, β::Number, backend)
    return planartrace!(C, A, p, q, α, β, backend, TO.DefaultAllocator())
end
# replace default backend with select_backend mechanism
function planartrace!(
        C, A, p::Index2Tuple, q::Index2Tuple, α::Number, β::Number, backend, allocator
    )
    if backend isa TO.DefaultBackend
        backend = TO.select_backend(planartrace!, C, A)
        return planartrace!(C, A, p, q, α, β, backend, allocator)
    elseif backend isa TO.NoBackend
        # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("No suitable backend found for planartrace! and tensor types $TC and $TA"))
    else
        # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("Unknown backend $backend for planartrace! and tensor types $TC and $TA"))
    end
end
# implementation
function planartrace!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, (p₁, p₂)::Index2Tuple, (q₁, q₂)::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    S = check_spacetype(C, A)
    if BraidingStyle(sectortype(S)) == Bosonic()
        return trace_permute!(C, A, (p₁, p₂), (q₁, q₂), α, β, backend)
    end
    (N₃ = length(q₁)) == length(q₂) ||
        throw(IndexError("number of trace indices does not match"))
    N₁, N₂ = length(p₁), length(p₂)

    @boundscheck begin
        numout(C) == N₁ || throw(IndexError("number of output indices does not match"))
        numin(C) == N₂ || throw(IndexError("number of input indices does not match"))
        all(i -> space(A, p₁[i]) == space(C, i), 1:N₁) ||
            throw(SpaceMismatch("trace: A = $(space(A)),
                    C = $(space(C)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(A, p₂[i]) == space(C, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("trace: A = $(space(A)),
                    C = $(space(C)), p₁ = $(p₁), p₂ = $(p₂)"))
        all(i -> space(A, q₁[i]) == dual(space(A, q₂[i])), 1:N₃) ||
            throw(SpaceMismatch("trace: A = $(space(A)),
                    q1 = $(q₁), q2 = $(q₂)"))
    end

    @timeit_debug GLOBAL_TIMER "planartrace!" begin
        if iszero(β)
            fill!(C, β)
        elseif !isone(β)
            rmul!(C, β)
        end
        β′ = One()
        for (f₁, f₂) in fusiontrees(A)
            for ((f₁′, f₂′), coeff) in planar_trace((f₁, f₂), (p₁, p₂), (q₁, q₂))
                @timeit_debug GLOBAL_TIMER "dense: trace" TO.tensortrace!(
                    C[f₁′, f₂′],
                    A[f₁, f₂], (p₁, p₂), (q₁, q₂), false,
                    α * coeff, β′,
                    backend, allocator
                )
            end
        end
    end
    return C
end

# insert default backend
function planarcontract!(
        C, A, pA::Index2Tuple, B, pB::Index2Tuple, pAB::Index2Tuple,
        α::Number, β::Number
    )
    return planarcontract!(C, A, pA, B, pB, pAB, α, β, TO.DefaultBackend())
end
# insert default allocator
function planarcontract!(
        C, A, pA::Index2Tuple, B, pB::Index2Tuple, pAB::Index2Tuple,
        α::Number, β::Number, backend
    )
    return planarcontract!(C, A, pA, B, pB, pAB, α, β, backend, TO.DefaultAllocator())
end
# replace default backend with select_backend mechanism
function planarcontract!(
        C, A, pA::Index2Tuple, B, pB::Index2Tuple, pAB::Index2Tuple,
        α::Number, β::Number, backend, allocator
    )
    if backend isa TO.DefaultBackend
        backend = TO.select_backend(planarcontract!, C, A, B)
        return planarcontract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    elseif backend isa TO.NoBackend
        # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        TB = typeof(B)
        throw(ArgumentError("No suitable backend found for planarcontract! and tensor types $TC, $TA and $TB"))
    else
        # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        TB = typeof(B)
        throw(ArgumentError("Unknown backend $backend for planarcontract! and tensor types $TC, $TA and $TB"))
    end
end
# implementation
function planarcontract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple,
        B::AbstractTensorMap, pB::Index2Tuple,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    if BraidingStyle(sectortype(C)) == Bosonic()
        return contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    end

    @timeit_debug GLOBAL_TIMER "planarcontract!" begin
        (oindA, cindA), (cindB, oindB), pAB′ = planar_contract_indices(A, pA, B, pB, pAB)
        A_in_layout = (oindA, cindA) == (codomainind(A), domainind(A))
        B_in_layout = (cindB, oindB) == (codomainind(B), domainind(B))

        if A_in_layout
            A′ = A
        else
            A′ = @timeit_debug GLOBAL_TIMER "alloc: buffers" TO.tensoralloc_add(
                scalartype(A), A, (oindA, cindA), false, Val(true), allocator
            )
            transpose!(A′, A, (oindA, cindA), One(), Zero(), backend, allocator)
        end

        if B_in_layout
            B′ = B
        else
            B′ = @timeit_debug GLOBAL_TIMER "alloc: buffers" TO.tensoralloc_add(
                scalartype(B), B, (cindB, oindB), false, Val(true), allocator
            )
            transpose!(B′, B, (cindB, oindB), One(), Zero(), backend, allocator)
        end

        if _isdirectoutput(pAB′, length(oindA))
            mul!(C, A′, B′, α, β)
        else # as in `blas_contract!`, a non-trivial `pAB` requires an intermediate
            AB = @timeit_debug GLOBAL_TIMER "alloc: buffers" TO.tensoralloc_contract(
                scalartype(C), A′, (codomainind(A′), domainind(A′)), false,
                B′, (codomainind(B′), domainind(B′)), false,
                TO.trivialpermutation(length(oindA), length(oindB)), Val(true), allocator
            )
            mul!(AB, A′, B′, One(), Zero())
            transpose!(C, AB, pAB′, α, β, backend, allocator)
            TO.tensorfree!(AB, allocator)
        end

        A_in_layout || TO.tensorfree!(A′, allocator)
        B_in_layout || TO.tensorfree!(B′, allocator)
    end
    return C
end

# auxiliary routines
# whether a contraction with `N₁` open indices on `A` directly yields the destination
function _isdirectoutput(pAB::Index2Tuple, N₁::Int)
    return length(pAB[1]) == N₁ && pAB == TO.trivialpermutation(pAB)
end

# rotate `t` such that `x` comes first
_rotate_to(t::Tuple, x) = TupleTools.circshift(t, 1 - something(findfirst(==(x), t)))

# rotate the cycle `indx` into `(head′..., reverse(tail′)...)`
function _planar_rotate(indx::IndexTuple, head::IndexTuple, tail::IndexTuple)
    N₁, N = length(head), length(indx)
    # rotate the arc of `head` indices in front; note that `indx` cannot be reassigned
    # without boxing it in the closures below
    rot = if 0 < N₁ < N
        i = findfirst(ntuple(n -> indx[n] ∈ head && indx[mod1(n - 1, N)] ∉ head, Val(N)))
        TupleTools.circshift(indx, 1 - something(i))
    else
        indx
    end
    head′ = ntuple(n -> rot[n], Val(N₁))
    tail′ = reverse(ntuple(n -> rot[N₁ + n], Val(length(tail))))
    TupleTools.sort(head′) == TupleTools.sort(head) ||
        throw(ArgumentError(lazy"$head and $tail do not partition the cycle $indx planarly"))
    return head′, tail′
end

"""
    planar_contract_indices(A, pA, B, pB, pAB) -> pA′, pB′, pAB′

Bring the index tuples of a planar contraction into canonical form, such that `pA′` and `pB′`
are cyclic partitions of the indices of `A` and `B`, i.e. such that

    C = transpose(transpose(A, pA′) * transpose(B, pB′), pAB′)

For sector types with `GenericUnit()` these are the only partitions with valid intermediate
spaces, so all space computations should use them. `A` and `B` can be anything supporting
`codomainind` and `domainind`, in particular `AbstractTensorMap`s and `HomSpace`s.

See also [`planarcontract!`](@ref) and [`planaralloc_contract`](@ref).
"""
function planar_contract_indices(
        A, (oindA, cindA)::Index2Tuple,
        B, (cindB, oindB)::Index2Tuple,
        pAB::Index2Tuple
    )
    indA = (codomainind(A)..., reverse(domainind(A))...)
    indB = (codomainind(B)..., reverse(domainind(B))...)
    oindA′, cindA′ = _planar_rotate(indA, oindA, cindA)
    cindB′, oindB′ = _planar_rotate(indB, cindB, oindB)

    # if all indices are contracted, fix the residual rotation using the other tensor
    if isempty(oindA′) && !isempty(cindA)
        cindA′ = _rotate_to(cindA′, cindA[something(findfirst(==(first(cindB′)), cindB))])
    end
    if isempty(oindB′) && !isempty(cindB)
        cindB′ = _rotate_to(cindB′, cindB[something(findfirst(==(first(cindA′)), cindA))])
    end
    TupleTools.sort(tuple.(cindA′, cindB′)) == TupleTools.sort(tuple.(cindA, cindB)) ||
        throw(ArgumentError(lazy"contraction of $cindA with $cindB is not planar"))

    # re-express `pAB` in terms of the reordered open indices
    remap = (
        map(something, TupleTools.indexin(oindA, oindA′))...,
        (length(oindA) .+ map(something, TupleTools.indexin(oindB, oindB′)))...,
    )
    pAB′ = (TupleTools.getindices(remap, pAB[1]), TupleTools.getindices(remap, pAB[2]))
    return (oindA′, cindA′), (cindB′, oindB′), pAB′
end

"""
    planaralloc_contract(TC, A, pA, B, pB, pAB, [istemp, allocator])

Allocate the destination of `planarcontract!(C, A, pA, B, pB, pAB, α, β)`.

The planar counterpart of `TensorOperations.tensoralloc_contract`: the index tuples are
canonicalized with [`planar_contract_indices`](@ref) first, such that the space computation
only involves valid intermediate spaces.
"""
function planaralloc_contract(
        TC, A, pA::Index2Tuple, B, pB::Index2Tuple, pAB::Index2Tuple,
        istemp::Val = Val(false), allocator = TO.DefaultAllocator()
    )
    pA′, pB′, pAB′ = planar_contract_indices(A, pA, B, pB, pAB)
    return TO.tensoralloc_contract(TC, A, pA′, false, B, pB′, false, pAB′, istemp, allocator)
end
