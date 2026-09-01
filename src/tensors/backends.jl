# Block-sparse contraction backend
# --------------------------------

"""
    BlockSparseUnsupported(msg) <: Exception

Thrown when a tensor or operation cannot be expressed in the block-sparse form required by
[`CuTENSORBlockSparse`](@ref). Usually caught internally to trigger a fallback.
"""
struct BlockSparseUnsupported <: Exception
    msg::String
end
Base.showerror(io::IO, e::BlockSparseUnsupported) =
    print(io, "BlockSparseUnsupported: ", e.msg)

"""
    blocksparse_compatible(::Type{I}) where {I <: Sector} -> Bool

Whether every fusion tree transformation of `I` reduces to a single scalar per fusion tree
pair, so that a contraction may be carried out on the raw block data *up to a per-block
rescaling* — which a caller doing so itself must apply, see
[`TensorKit.blocksparse_contract_signs`](@ref).

Beware that this is weaker than "the raw block data may be contracted directly": that only
holds in the first of the two cases below. The trait additionally excludes `Trivial`, which
satisfies the property but is better served by the dense path, so a `true` answer means "this
backend supports `I`" rather than any one mathematical property.

The property requires unique fusion and trivial F-symbols. There are two cases:

  - Trivial R-symbols, twists and Frobenius–Schur phases, i.e. `fusiontensor((f₁, f₂)) == 1`
    for every fusion tree pair. A stored subblock then coincides with the corresponding slice
    of the dense array, the identity `F_C = F_A · F_B` holds for every matching triple, and
    the raw block contraction is directly correct. Among the sectors shipped with TensorKit
    this means the irreps of abelian *groups* and products thereof.
  - Fermionic braiding, where R-symbols and twists are `±1`. The raw block contraction then
    differs from the categorical one by a per-block sign, which the backend corrects for; see
    [`TensorKit.blocksparse_contract_signs`](@ref).

The default is `false`, deliberately: `FusionStyle(I) === UniqueFusion()` is *not*
sufficient. `ZNElement`/`Z3Element` have unique fusion, but their anyonic R-symbols are
genuine phases rather than signs, which is **unvalidated** here, so they fall back.

See also [`CuTENSORBlockSparse`](@ref).
"""
blocksparse_compatible(::Type{<:Sector}) = false
blocksparse_compatible(::Type{<:TensorKitSectors.AbelianIrrep}) = true
# fermion parity has unique fusion and trivial F-symbols, with R-symbols and twists in {±1}
blocksparse_compatible(::Type{FermionParity}) = true
Base.@assume_effects :foldable function blocksparse_compatible(
        ::Type{ProductSector{T}}
    ) where {T <: Tuple}
    return all(blocksparse_compatible, fieldtypes(T))
end
blocksparse_compatible(t::AbstractTensorMap) = blocksparse_compatible(sectortype(t))

"""
    CuTENSORBlockSparse(; fallback = TensorOperations.DefaultBackend(), strict = false, plans = nothing)

Backend that maps a symmetric tensor contraction onto a *single* block-sparse contraction of
the underlying storage, with no fusion tree transformation and no intermediate permutations:
the index tuples only determine mode labels. Currently implemented for `TensorMap`s with
CUDA storage, via cuTENSOR's block-sparse API, and only for sector types satisfying
[`blocksparse_compatible`](@ref).

For fermionic sectors the raw block contraction differs from the categorical one by a
per-block sign, which is corrected by scaling blocks. That path is still free of any fusion
tree transformation, but it does allocate one temporary per operand whose signs are
non-trivial.

Requires both `CUDA.jl` and `cuTENSOR.jl` to be loaded. Note that NVIDIA documents the
underlying block-sparse API as a *public beta* with no guarantee of stability across
releases, which is why this backend is opt-in rather than selected automatically:

```julia
@tensor backend = CuTENSORBlockSparse() C[a, b] = A[a, c] * B[c, b]
```

## Keywords
- `fallback`: backend used for contractions that cannot be expressed in block-sparse form,
  and for all non-contraction operations.
- `strict`: throw a [`BlockSparseUnsupported`](@ref) instead of falling back silently. Off by
  default because a single `@tensor` network routinely mixes supported and unsupported
  contractions; turn it on to assert that you are on the fast path.
- `plans`: a plan cache to use instead of the global one, so that plan lifetime can be scoped
  to a hot loop. `nothing` uses the global cache.

See also [`blocksparse_compatible`](@ref).
"""
struct CuTENSORBlockSparse{B <: AbstractBackend, C} <: AbstractBackend
    fallback::B
    strict::Bool
    plans::C
end
function CuTENSORBlockSparse(;
        fallback::AbstractBackend = TO.DefaultBackend(), strict::Bool = false,
        plans = nothing
    )
    return CuTENSORBlockSparse(fallback, strict, plans)
end

# only contraction has a block-sparse form; without these, any permutation or trace in an
# `@tensor` network would reach the leaf `TO.tensoradd!` and hit an "unknown backend" error
function TO.tensoradd!(
        C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend::CuTENSORBlockSparse, allocator
    )
    return TO.tensoradd!(C, A, pA, conjA, α, β, backend.fallback, allocator)
end
function TO.tensortrace!(
        C::AbstractTensorMap, A::AbstractTensorMap, p::Index2Tuple, q::Index2Tuple,
        conjA::Bool, α::Number, β::Number, backend::CuTENSORBlockSparse, allocator
    )
    return TO.tensortrace!(C, A, p, q, conjA, α, β, backend.fallback, allocator)
end

"""
    plan_contract(C, A, pA, B, pB, pAB; kwargs...) -> AbstractBackend
    plan_contract(::Type{T}, VC, VA, pA, VB, pB, pAB; kwargs...) -> AbstractBackend

Precompute everything a block-sparse contraction needs that does not depend on the tensor
*data*, and return it as a backend that can be handed straight to `@tensor` or
`tensorcontract!`:

```julia
plan = plan_contract(C, A, pA, B, pB, pAB)
for i in 1:nsweeps
    @tensor backend = plan C[a, b] = A[a, c] * B[c, b]
end
```

The block-sparse description of a contraction — the sparsity pattern, the block strides and
the selected kernel — is a pure function of the spaces, the index tuples and the scalar type;
the block pointers are only supplied at execution time. Hoisting a plan out of a loop is
therefore safe, and worth doing whenever the spaces are fixed across many iterations, as in an
MPS sweep at fixed bond dimension.

The second form takes spaces rather than tensors, so a plan can be built before the tensors
exist. Reusing a plan with spaces or index tuples other than those it was built for is an
error, checked under `@boundscheck`.

Plans are cached automatically, so calling this is not required for reuse; it exists to avoid
even the cache lookup, and to give explicit control over plan lifetime.

For a fermionic sector this includes the per-block sign correction. That depends additionally
on `conjA`/`conjB`, which are not part of a plan's identity, so a plan carries the correction
for all four combinations and picks one at execution time — no separate lookup, and no way for
the correction to disagree with the descriptors it was built alongside.

Requires `CUDA.jl` and `cuTENSOR.jl`. See also [`CuTENSORBlockSparse`](@ref).
"""
function plan_contract end

# not CUDA-backed, so the extension's more specific method never applies: fall back
function _tensorcontract!(
        C::AbstractTensorMap,
        A::AbstractTensorMap, pA::Index2Tuple, conjA::Bool,
        B::AbstractTensorMap, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number,
        backend::CuTENSORBlockSparse, allocator
    )
    if backend.strict
        throw(
            BlockSparseUnsupported(
                lazy"""
                no block-sparse contraction available for tensor types $(typeof(C)), $(typeof(A)) and $(typeof(B)).
                This requires CUDA-backed `TensorMap`s, with both `CUDA.jl` and `cuTENSOR.jl` loaded.
                """
            )
        )
    end
    return _generic_tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend.fallback, allocator
    )
end

# Scheduler implementation
# ------------------------
function select_scheduler(scheduler = OhMyThreads.Implementation.NotGiven(); kwargs...)
    return if scheduler == OhMyThreads.Implementation.NotGiven() && isempty(kwargs)
        Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler()
    else
        OhMyThreads.Implementation._scheduler_from_userinput(scheduler; kwargs...)
    end
end

"""
    const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())

The default scheduler used when looping over different blocks in the matrix representation of a
tensor.
For controlling this value, see also [`set_blockscheduler`](@ref) and [`with_blockscheduler`](@ref).
"""
const blockscheduler = ScopedValue{Scheduler}(SerialScheduler())

"""
    with_blockscheduler(f, [scheduler]; kwargs...)

Run `f` in a scope where the `blockscheduler` is determined by `scheduler' and `kwargs...`.
"""
@inline function with_blockscheduler(
        f, scheduler = OhMyThreads.Implementation.NotGiven(); kwargs...
    )
    return @with blockscheduler => select_scheduler(scheduler; kwargs...) f()
end

# TODO: disable for trivial symmetry or small tensors?
default_blockscheduler(t::AbstractTensorMap) = default_blockscheduler(typeof(t))
default_blockscheduler(::Type{T}) where {T <: AbstractTensorMap} = blockscheduler[]
