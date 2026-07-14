# Precompilation

The first tensor operation in a Julia session — a contraction, an index manipulation, or a factorization — incurs a significant compilation latency (time-to-first-execution, TTFX).
To mitigate this, TensorKit ships a precompilation workload that is **enabled by default**:
it runs a small set of representative index manipulations, contractions and factorizations for the `Trivial`, `Z2Irrep`, `SU2Irrep` and `FermionParity` symmetries over `Float64` and `ComplexF64`.

Because the numerically-heavy kernels are mostly *sector-agnostic* (they depend only on the element type, the arity, and `sectorscalartype`),
this workload also speeds up the first operation on symmetries that are *not* in the workload — including user-defined ones.

The workload can be tuned or disabled through [Preferences](https://github.com/JuliaPackaging/Preferences.jl):

```julia
using TensorKit, Preferences, PrecompileTools
# disable the workload entirely (fastest precompilation, no TTFX benefit)
PrecompileTools.set_preferences!(TensorKit, "precompile_workloads" => false; force=true)
# disable individual suites (each defaults to `true`)
set_preferences!(TensorKit, "precompile_contract" => false; force=true)
set_preferences!(TensorKit, "precompile_indexmanipulations" => false; force=true)
set_preferences!(TensorKit, "precompile_factorizations" => false; force=true)
# restrict the element types
set_preferences!(TensorKit, "precompile_eltypes" => ["Float64"]; force=true)
# restrict / change the symmetries that are precompiled
set_preferences!(TensorKit, "precompile_sectors" => ["Trivial", "Z2Irrep"]; force=true)
# highest tensor arity (leg count) to precompile, i.e. arities `1:n`; e.g. rank-6 for PEPS/PEPO
set_preferences!(TensorKit, "precompile_ndims" => 6; force=true)
```

Changing a preference triggers recompilation of TensorKit the next time it is loaded.

Downstream packages or startup files that define or heavily use their own symmetry can precompile TensorKit's operations for it by
calling the `precompile_*` helpers inside their own `PrecompileTools.@compile_workload`:

```julia
using TensorKit, PrecompileTools
@compile_workload begin
    TensorKit.precompile_contract(Vect[MyAnyon](s₀ => 1, s₁ => 1))
end
```

```@docs; canonical=false
TensorKit.Precompilation.precompile_contract
TensorKit.Precompilation.precompile_indexmanipulations
TensorKit.Precompilation.precompile_factorizations
```
