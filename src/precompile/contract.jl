"""
    precompile_contract(S::Type{<:IndexSpace}; eltypes=[Float64, ComplexF64], ndims=4)

Run a small, representative set of contraction, trace, and permutation operations on tensors built
from the space `V = unitspace(S)`, for each element type in `eltypes` and each tensor arity (number of
legs) in `1:ndims`.

Note that it can be beneficial to put a more comprehensive set of relevant symmetries in a startup file,
for example by adding:

```julia
@compile_workload begin
    TensorKit.precompile_contract(Vect[MySector])
end
```

See also [`precompile_indexmanipulations`](@ref TensorKit.precompile_indexmanipulations), [`precompile_factorizations`](@ref TensorKit.precompile_factorizations).
"""
function precompile_contract(::Type{S}; eltypes = PRECOMPILE_ELTYPES, ndims = PRECOMPILE_NDIMS) where {S <: IndexSpace}
    V = unitspace(S)
    backend = TO.DefaultBackend()
    allocator = TO.DefaultAllocator()
    symmetric_braiding = BraidingStyle(sectortype(S)) isa SymmetricBraiding
    for T in eltypes
        α, β = rand(T), rand(T)

        # contraction + permutation for each requested arity (leg count); `Val(N)`/`ntuple`
        # keep the index tuples concrete so the machinery specializes per arity
        for N in 1:ndims
            W = V^(N - 1)   # N-1 legs
            # contraction of two arity-N tensors over their N-1 shared legs -> arity-2 result
            A = randn(T, V ← W)
            B = randn(T, W ← V)
            pA = ((1,), ntuple(i -> i + 1, Val(N - 1)))
            pB = (ntuple(identity, Val(N - 1)), (N,))
            pAB = ((1,), (2,))
            C = TO.tensoralloc_contract(T, A, pA, false, B, pB, false, pAB, Val(false))

            planarcontract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
            planarcontract!(C, A, pA, B, pB, pAB, One(), Zero(), backend, allocator)

            if symmetric_braiding
                TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, backend, allocator)
                TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, One(), Zero(), backend, allocator)
            end
        end

        # the conjugated-operand branch (`conjA=true`) is a distinct runtime path (adjoint
        # handling) that `conj(A) * B` networks hit; `@plansor` picks the planar or non-planar
        # implementation depending on the braiding style, so this compiles for any sectortype
        A2 = randn(T, V ← V)
        B2 = randn(T, V ← V)
        @plansor Cc[a; c] := conj(A2[b; a]) * B2[b; c]

        # partial trace (the two traced legs are mutually dual)
        At = randn(T, V ⊗ V' ← V)
        Ct = TO.tensoralloc_add(T, At, ((3,), ()), false, Val(false))
        planartrace!(Ct, At, ((3,), ()), ((1,), (2,)), α, β, backend, allocator)
        symmetric_braiding && TO.tensortrace!(
            Ct, At, ((3,), ()), ((1,), (2,)), false, α, β, backend, allocator
        )
    end
    return nothing
end
