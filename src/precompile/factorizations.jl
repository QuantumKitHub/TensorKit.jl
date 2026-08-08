"""
    precompile_factorizations(V::IndexSpace; eltypes=[Float64, ComplexF64])

Run a small, representative set of tensor-factorization operations (singular value, QR, LQ,
eigenvalue, orthogonal/null-space and polar decompositions) on tensors built from the space `V`,
for each element type in `eltypes`.

Note that it can be beneficial to put a more comprehensive set of relevant symmetries in a startup file,
for example by adding:

```julia
@compile_workload begin
    TensorKit.precompile_factorizations(Vect[MyAnyon](s₀ => 1, s₁ => 1))
end
```

See also [`precompile_contract`](@ref TensorKit.precompile_contract), [`precompile_indexmanipulations`](@ref TensorKit.precompile_indexmanipulations).
"""
function precompile_factorizations(V::IndexSpace; eltypes = PRECOMPILE_ELTYPES)
    for T in eltypes
        # Three representative shapes: a square tensor for the bulk of the decompositions, a
        # hermitian one for the `eigh` path, and rectangular ones so the null-space kernels
        # operate on non-empty blocks.
        W = V^2   # V ⊗ V
        t = randn(T, W ← W)            # square
        tr = randn(T, W ← V)           # tall (codomain larger) -> non-empty left null space
        tw = randn(T, V ← W)           # wide (domain larger)   -> non-empty right null space
        th = (t + t') / 2              # hermitian (square, Euclidean inner product)

        # Singular value decomposition
        svd_full(t)
        svd_compact(t)
        svd_vals(t)
        svd_trunc(t; trunc = truncrank(1))

        # QR / LQ decompositions (null-space variants on the appropriately shaped tensors)
        qr_full(t)
        qr_compact(t)
        qr_null(tr)
        lq_full(t)
        lq_compact(t)
        lq_null(tw)

        # Eigenvalue decompositions (hermitian variants require a hermitian input)
        eig_full(t)
        eig_vals(t)
        eigh_full(th)
        eigh_vals(th)

        # Orthogonal / null-space helpers
        left_orth(t)
        right_orth(t)
        left_null(tr)
        right_null(tw)

        # Polar decompositions
        left_polar(t)
        right_polar(t)
    end
    return nothing
end
