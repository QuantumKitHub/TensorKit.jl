using PrecompileTools: PrecompileTools, @compile_workload
using Preferences: @load_preference, load_preference

# Preferences
# -----------
function _validate_precompile_eltypes(eltypes)
    eltypes isa Vector{String} ||
        throw(ArgumentError("`precompile_eltypes` should be a vector of strings, got $(typeof(eltypes)) instead"))
    return map(eltypes) do Tstr
        T = eval(Meta.parse(Tstr))
        (T isa DataType && T <: Number) ||
            error("Invalid precompile_eltypes entry: `$Tstr`")
        return T
    end
end

const PRECOMPILE_ELTYPES = _validate_precompile_eltypes(
    @load_preference("precompile_eltypes", ["Float64", "ComplexF64"])
)

# Tensor arities (numbers of legs) to precompile the contraction/permutation machinery for.
# The heavy transform/braid machinery is specialized per arity, so this controls how many
# leg-counts get the full precompilation benefit (higher arities cost more precompile time).
const PRECOMPILE_NDIMS = let n = @load_preference("precompile_ndims", [2, 3, 4])
    (n isa Vector{Int} && all(≥(1), n)) ||
        throw(ArgumentError("`precompile_ndims` should be a vector of positive `Int`s, got `$n`"))
    n
end

# Representative space for each supported sector-name in the `precompile_sectors` preference.
# One representative per (fusion style, coefficient type) suffices — the heavy kernels are
# sector-agnostic (see `precompile_contract`). Other sectors can be precompiled by calling
# `precompile_contract` directly.
function _precompile_space(name::AbstractString)
    name == "Trivial" && return ℂ^2
    name == "Z2Irrep" && return Vect[Z2Irrep](0 => 1, 1 => 1)
    name == "U1Irrep" && return Vect[U1Irrep](-1 => 1, 0 => 1, 1 => 1)
    name == "SU2Irrep" && return Vect[SU2Irrep](0 => 1, 1 // 2 => 1)
    name == "FermionParity" && return Vect[FermionParity](0 => 1, 1 => 1)
    return error("unknown precompile_sectors entry `$name`; precompile it directly with `precompile_contract`")
end

const PRECOMPILE_SECTORS = let s = @load_preference(
        "precompile_sectors", ["Trivial", "Z2Irrep", "SU2Irrep", "FermionParity"]
    )
    s isa Vector{String} ||
        throw(ArgumentError("`precompile_sectors` should be a vector of strings, got $(typeof(s))"))
    s
end

# Whether to run the precompile workload. Respects the ecosystem-wide
# `PrecompileTools` switch and a TensorKit-local `precompile_workload` preference
# (default `true`). Mirrors the pattern used by TensorOperations.
function _precompile_workload_enabled(mod::Module = @__MODULE__)
    return try
        if load_preference(PrecompileTools, "precompile_workloads", true)
            return load_preference(mod, "precompile_workload", true)
        else
            return false
        end
    catch
        false
    end
end

# Workload
# --------
"""
    precompile_contract(V::IndexSpace; eltypes=[Float64, ComplexF64], ndims=[2, 3, 4])

Run a small, representative set of tensor operations (contraction, trace, permutation) on
tensors built from the space `V`, for each element type in `eltypes` and each tensor arity
(number of legs) in `ndims`. This forces compilation of the contraction/permutation machinery
for the sector type of `V`.

The machinery is specialized per arity, so `ndims` controls which leg-counts get the full
benefit; arities not covered still benefit partially from the sector- and arity-agnostic parts
(most importantly the per-block BLAS `mul!`).

The expensive, numerically-heavy kernels (`add_transform!`, the tree transformers, and the
per-block BLAS `mul!`) are *sector-agnostic* — they depend only on the element type, the
arity, and `sectorscalartype(sectortype(V))`. Consequently, calling this once for a
representative symmetry precompiles code that is reused by every other symmetry with the same
fusion style and coefficient type.

Downstream packages that define their own symmetry can precompile TensorKit's contraction path
for it by calling this inside their own `PrecompileTools.@compile_workload`, e.g.

```julia
@compile_workload begin
    TensorKit.precompile_contract(Vect[MyAnyon](s₀ => 1, s₁ => 1))
end
```

TensorKit's own workload (enabled by default) calls this for `Trivial`, `Z2Irrep`, `SU2Irrep`
and `FermionParity`. It can be tuned or disabled via preferences:

```julia
using TensorKit, Preferences
set_preferences!(TensorKit, "precompile_eltypes" => ["Float64", "ComplexF64"]; force=true)
set_preferences!(TensorKit, "precompile_workload" => false; force=true)  # disable entirely
```
"""
function precompile_contract(V::IndexSpace; eltypes = PRECOMPILE_ELTYPES, ndims = PRECOMPILE_NDIMS)
    backend = TO.DefaultBackend()
    allocator = TO.DefaultAllocator()
    for T in eltypes
        α, β = rand(T), rand(T)

        # contraction + permutation for each requested arity (leg count)
        for N in ndims
            _precompile_contract_arity(V, T, Val(N), α, β, backend, allocator)
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

# `V^{⊗M}` (the unit space for `M == 0`), with `M` a compile-time constant.
_repeat_space(V, ::Val{0}) = one(V)
_repeat_space(V, ::Val{M}) where {M} = foldl(⊗, ntuple(_ -> V, Val(M)))

# Precompile the contraction and permutation machinery for tensors with `N` legs. `N` is a
# `Val` so the index tuples are compile-time concrete (the machinery specializes per arity).
function _precompile_contract_arity(V, ::Type{T}, ::Val{N}, α, β, backend, allocator) where {T, N}
    W = _repeat_space(V, Val(N - 1))   # N-1 legs
    # contraction of two arity-N tensors over their N-1 shared legs -> arity-2 result
    A = randn(T, V ← W)
    B = randn(T, W ← V)
    pA = ((1,), ntuple(i -> i + 1, Val(N - 1)))
    pB = (ntuple(identity, Val(N - 1)), (N,))
    _precompile_contract!(A, pA, B, pB, ((1,), (2,)), α, β, backend, allocator)

    # a non-trivial permutation exercises the repartition/braid/transform machinery at arity N
    # (the contraction above takes the no-copy view path for its natural partition)
    permute(A, (ntuple(i -> N - i + 1, Val(N)), ()))
    return nothing
end

# both scalar-type paths: generic `(α,β)` and the identity `(One(), Zero())` fast path
function _precompile_contract!(A, pA, B, pB, pAB, α, β, backend, allocator)
    T = promote_type(scalartype(A), scalartype(B))
    C = TO.tensoralloc_contract(T, A, pA, false, B, pB, false, pAB, Val(false))
    TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β, backend, allocator)
    TO.tensorcontract!(C, A, pA, false, B, pB, false, pAB, One(), Zero(), backend, allocator)
    return C
end

if _precompile_workload_enabled()
    @compile_workload begin
        for name in PRECOMPILE_SECTORS
            precompile_contract(_precompile_space(name); eltypes = PRECOMPILE_ELTYPES)
        end
    end
end
