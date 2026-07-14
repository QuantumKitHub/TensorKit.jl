"""
    precompile_contract(V::IndexSpace; eltypes=[Float64, ComplexF64], ndims=4)

Run a small, representative set of contraction, trace, and permutation operations on tensors built
from the space `V`, for each element type in `eltypes` and each tensor arity (number of legs) in `1:ndims`.

Note that it can be beneficial to put a more comprehensive set of relevant symmetries in a startup file,
for example by adding:

```julia
@compile_workload begin
    TensorKit.precompile_contract(Vect[MyAnyon](s₀ => 1, s₁ => 1))
end
```

See also [`precompile_indexmanipulations`](@ref TensorKit.precompile_indexmanipulations), [`precompile_factorizations`](@ref TensorKit.precompile_factorizations).
"""
function precompile_contract(V::IndexSpace; eltypes = PRECOMPILE_ELTYPES, ndims = PRECOMPILE_NDIMS)
    backend = TO.DefaultBackend()
    allocator = TO.DefaultAllocator()
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
            C = TO.tensoralloc_contract(T, A, pA, false, B, pB, false, ((1,), (2,)), Val(false))
            # both scalar-type paths: generic `(α,β)` and the identity `(One(), Zero())` fast path
            TO.tensorcontract!(C, A, pA, false, B, pB, false, ((1,), (2,)), α, β, backend, allocator)
            TO.tensorcontract!(C, A, pA, false, B, pB, false, ((1,), (2,)), One(), Zero(), backend, allocator)

            # a non-trivial permutation exercises the repartition/braid/transform machinery at
            # arity N (the contraction above takes the no-copy view path for its natural partition)
            permute(A, (ntuple(i -> N - i + 1, Val(N)), ()))
        end

        # the conjugated-operand branch (`conjA=true`) is a distinct runtime path (adjoint
        # handling) that `conj(A) * B` networks hit; `@tensor` handles the space bookkeeping
        A2 = randn(T, V ← V)
        B2 = randn(T, V ← V)
        @tensor Cc[a; c] := conj(A2[b; a]) * B2[b; c]

        # partial trace (the two traced legs are mutually dual)
        At = randn(T, V ⊗ V' ← V)
        TO.tensortrace!(
            TO.tensoralloc_add(T, At, ((3,), ()), false, Val(false)),
            At, ((3,), ()), ((1,), (2,)), false, α, β, backend, allocator
        )
    end
    return nothing
end
