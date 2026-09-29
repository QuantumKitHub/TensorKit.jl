"""
    precompile_factorizations(S::Type{<:IndexSpace}; eltypes=[Float64, ComplexF64])

Run a small, representative set of tensor-factorization operations (singular value, QR, LQ,
eigenvalue, orthogonal/null-space and polar decompositions) on tensors built from the space
`V = unitspace(S)`, for each element type in `eltypes`.

Note that it can be beneficial to put a more comprehensive set of relevant symmetries in a startup file,
for example by adding:

```julia
@compile_workload begin
    TensorKit.precompile_factorizations(Vect[MySector])
end
```

See also [`precompile_contract`](@ref TensorKit.precompile_contract), [`precompile_indexmanipulations`](@ref TensorKit.precompile_indexmanipulations).
"""
function precompile_factorizations(::Type{S}; eltypes = PRECOMPILE_ELTYPES) where {S <: IndexSpace}
    V = unitspace(S)
    for T in eltypes
        # Three representative shapes: a square tensor for the bulk of the decompositions, a
        # hermitian one for the `eigh` path, and rectangular ones so the null-space kernels
        # operate on non-empty blocks.
        W = V^2   # V ⊗ V
        t = randn(T, W ← W)            # square
        tr = randn(T, W ← V)           # tall (codomain larger) -> non-empty left null space
        tw = randn(T, V ← W)           # wide (domain larger)   -> non-empty right null space
        th = (t + t') / 2              # hermitian (square, Euclidean inner product)

        # projectors
        project_hermitian(t)
        project_antihermitian(t)
        project_isometric(t)

        # exponential
        exponential(th)

        # Singular value decomposition
        svd_full(t)
        svd_compact(t)
        svd_vals(t)
        svd_trunc(t; trunc = truncrank(1))
        svd_trunc(t; trunc = trunctol(; atol = 1e-10))
        svd_trunc(t; trunc = truncerror(; atol = 1e-10))
        svd_trunc(t; trunc = notrunc())
        svd_trunc(t; trunc = truncspace(space(t, 1)))
        svd_trunc(t; trunc = truncrank(1) | trunctol(; atol = 1e-10))
        svd_trunc(t; trunc = truncrank(1) & trunctol(; atol = 1e-10))

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
        eig_trunc(t; trunc = truncrank(1))
        eig_trunc(t; trunc = trunctol(; atol = 1e-10))
        eig_trunc(t; trunc = truncerror(; atol = 1e-10))
        eig_trunc(t; trunc = notrunc())
        eig_trunc(t; trunc = truncspace(space(t, 1)))
        eig_trunc(t; trunc = truncrank(1) | trunctol(; atol = 1e-10))
        eig_trunc(t; trunc = truncrank(1) & trunctol(; atol = 1e-10))
        eigh_full(th)
        eigh_vals(th)
        eigh_trunc(th; trunc = truncrank(1))
        eigh_trunc(th; trunc = trunctol(; atol = 1e-10))
        eigh_trunc(th; trunc = truncerror(; atol = 1e-10))
        eigh_trunc(th; trunc = notrunc())
        eigh_trunc(th; trunc = truncspace(space(th, 1)))
        eigh_trunc(th; trunc = truncrank(1) | trunctol(; atol = 1e-10))
        eigh_trunc(th; trunc = truncrank(1) & trunctol(; atol = 1e-10))

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
