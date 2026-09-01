# Backends

The `@tensor` and `@planar` macros, as well as the underlying
[`TensorOperations.tensorcontract!`](@extref) family, accept a `backend =` keyword that selects
*how* an operation is carried out, leaving *what* it computes unchanged.
The available backends come from [TensorOperations](https://quantumkithub.github.io/TensorOperations.jl/stable/), which by default
picks one automatically based on the storage type of the individual blocks — for CPU tensors a
strided/BLAS backend, for CUDA tensors the dense cuTENSOR backend.

TensorKit additionally provides one backend of its own, which changes the contraction strategy
at the level of the whole symmetric tensor rather than block by block.

## Block-sparse contractions with cuTENSOR

By default, a symmetric tensor contraction is carried out by first permuting the operands into
matrix form — which requires fusion tree transformations and temporary tensors — and then
performing one matrix multiplication per coupled sector.
For a tensor with many small sectors on the GPU, this is dominated by the cost of launching a
large number of small kernels.

For abelian symmetries, fermionic ones included, there is a more direct route.
The way TensorKit stores such a tensor — one dense block per tuple of uncoupled sectors, laid
out as strided views into a single flat vector — is precisely the *block-sparse* data model of
cuTENSOR's block-sparse contraction API.
The index tuples then only determine cuTENSOR mode labels, so the contraction becomes a single
kernel launch with no permutations and no temporaries:

```julia
using CUDA, cuTENSOR

V = Vect[U1Irrep](-1 => 32, 0 => 64, 1 => 32)
A = CuTensorMap(randn(ComplexF64, V ⊗ V ← V))
B = CuTensorMap(randn(ComplexF64, V ← V ⊗ V))

@tensor backend = CuTENSORBlockSparse() C[a, b, d, e] := A[a, b, c] * B[c, d, e]
```

This backend is opt-in. cuTENSOR's block-sparse functionality is documented by NVIDIA as a
*public beta*, with a restricted feature set and no guarantee of API stability between
releases, so it is never selected automatically. Whether it is faster than one dense
matrix multiplication per coupled sector depends on how the total dimension is spread over
sectors.

Fermionic symmetries are supported too. There a raw block-wise contraction is *not* the
categorical one: the permutations the default path performs, and the twist it inserts for a dual
contracted leg, each contribute a factor per fusion tree — for fermion parity, a sign. Since
those factors are scalars, the discrepancy is one scalar per block, and can be corrected by
scaling blocks rather than by transforming fusion trees. The cost is one temporary per operand
whose correction is non-trivial; an output that already has the codomain/domain *partition* the
contraction produces naturally needs no correction of the result at all. Note that a natural
index *order* is not enough — moving a leg between codomain and domain bends it, and for a
fermionic sector bending is not free, so `pAB = ((1,2,3,4), ())` does carry a correction even
though it permutes nothing.

Contractions the backend cannot express fall back to the default path silently, which keeps it
usable inside a `@tensor` network that mixes supported and unsupported operations. Pass
`strict = true` to turn those fallbacks into errors instead, and consult
[`blocksparse_compatible`](@ref) for which symmetries qualify.

```@docs; canonical=false
CuTENSORBlockSparse
blocksparse_compatible
TensorKit.blocksparse_contract_signs
```

### Reusing plans

Everything cuTENSOR needs to know about a block-sparse contraction — the sparsity pattern, the
block strides, and the kernel chosen by its heuristic — is a function of the spaces, the index
tuples and the scalar type alone; the block pointers are supplied only at execution time.
Plans are therefore reusable, and are cached automatically, so a repeated contraction over
tensors with the same spaces pays for plan construction only once.

For an inner loop where even the cache lookup is unwanted — an MPS sweep at fixed bond
dimension, say — a plan can be hoisted out explicitly. Since a plan *is* a fully specialized
backend, it is passed the same way:

```julia
plan = plan_contract(C, A, ((1, 2), (3,)), B, ((1,), (2, 3)), ((1, 2), (3, 4)))
for i in 1:nsweeps
    @tensor backend = plan C[a, b, d, e] = A[a, b, c] * B[c, d, e]
end
```

```@docs; canonical=false
plan_contract
```

Plan lifetime can also be scoped without changing call sites, by handing the backend a private
cache via `CuTENSORBlockSparse(; plans = ...)`. The global cache is registered with TensorKit's
cache registry, so [`empty_globalcaches!`](@ref) releases plans along with everything else —
which is worth doing around a `CUDA.device_reset!`, since a plan is only valid on the device
it was created for.

## Block schedulers

Independently of the backend, the loop over the blocks of a symmetric tensor can be
parallelized over threads:

```@docs; canonical=false
TensorKit.blockscheduler
TensorKit.with_blockscheduler
```
