"""
    precompile_indexmanipulations(V::IndexSpace; eltypes=[Float64, ComplexF64], ndims=4)

Run a small, representative set of index-manipulation operations on tensors built from the space `V`,
for each element type in `eltypes` and each tensor arity (number of legs) in `1:ndims`.

Note that it can be beneficial to put a more comprehensive set of relevant symmetries in a startup file,
for example by adding:

```julia
@compile_workload begin
    TensorKit.precompile_indexmanipulations(Vect[MyAnyon](s₀ => 1, s₁ => 1))
end
```

See also [`precompile_contract`](@ref TensorKit.precompile_contract), [`precompile_factorizations`](@ref TensorKit.precompile_factorizations).
"""
function precompile_indexmanipulations(V::IndexSpace; eltypes = PRECOMPILE_ELTYPES, ndims = PRECOMPILE_NDIMS)
    for T in eltypes
        # `Val(N)`/`ntuple` keep the index tuples concrete so the machinery specializes per arity
        for N in 1:ndims
            N₁ = cld(N, 2)   # codomain legs
            # a tensor split non-trivially into codomain/domain (N₁ | N-N₁)
            t = randn(T, V^N₁ ← V^(N - N₁))

            # a non-trivial reordering of all N legs (reverse of `1:N`), split back into (N₁ | N-N₁)
            perm = ntuple(i -> N - i + 1, Val(N))
            p1 = ntuple(i -> perm[i], Val(N₁))
            p2 = ntuple(i -> perm[i + N₁], Val(N - N₁))

            # `permute` and `braid` funnel through `add_transform!` with different transformers
            permute(t, (p1, p2))
            braid(t, (p1, p2), ntuple(identity, Val(N)))   # `levels` is a tuple over the source indices

            # canonical transpose `(reverse(domain), reverse(codomain))` is always a valid cyclic transpose
            tp1 = ntuple(i -> N - i + 1, Val(N - N₁))
            tp2 = ntuple(i -> N₁ - i + 1, Val(N₁))
            transpose(t, (tp1, tp2))

            # repartition to a different codomain size, forcing the repartition/transpose path
            repartition(t, N₁ < N ? N₁ + 1 : N₁ - 1)

            twist(t, 1)
        end
    end
    return nothing
end
