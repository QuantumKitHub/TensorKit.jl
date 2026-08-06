# TensorKit.jl test suite

Tests use [ParallelTestRunner.jl](https://github.com/vchuravy/ParallelTestRunner.jl) for parallel
execution. Each test file runs in its own worker process. Shared helpers are loaded automatically
via `init_code` — test files do not need to include `setup.jl` themselves.

Test content which is sector-dependent (i.e. for graded spaces, fusion trees, (diagonal) tensors and factorizations) lives in the reusable, standalone `TensorKitTestSuite` module (see below). `TensorKitTestSuite` can be `include`d directly by downstream
packages to test a custom `Sector` type.

## Running tests

```julia
# Run all tests: requires Julia 1.12+ due to Mooncake
using Pkg; Pkg.test()
```

```bash
# Direct invocation — requires Julia 1.12+ (workspace support)
julia --project=test test/runtests.jl

# Run only a specific group (directory prefix)
julia --project=test test/runtests.jl symmetries

# Run only a specific file
julia --project=test test/runtests.jl tensors/factorizations

# Fast mode: fewer sectors, fewer scalar types, AD tests skipped
julia --project=test test/runtests.jl --fast

# Combine --fast with a group or file filter
julia --project=test test/runtests.jl --fast tensors

# Control parallelism
julia --project=test test/runtests.jl --jobs=4
```

`Pkg.test()`/`Pkg.test("TensorKit"; test_args = [...])` accept the same `--fast`/group/file
arguments via `test_args` (e.g. `test_args = ["--fast", "tensors/diagonal"]`).

## Test groups

| Group | Contents |
|-------|----------|
| `symmetries` | Spaces and fusion trees (call into `TensorKitTestSuite`'s `:spaces`, `:single_fusiontrees`, `:double_fusiontrees` groups) |
| `tensors` | Core tensor operations, planar tensors, diagonal tensors (call into `:tensors`, `:diagonal_tensors`) |
| `factorizations` | Eigendecomposition, QR/LQ, projections, polar, (truncated) SVD (call into `:factorizations`) |
| `other` | Aqua code-quality checks, bug-fix regressions |
| `chainrules` | ChainRulesCore AD tests |
| `mooncake` | Mooncake AD tests |
| `cuda` / `amd` | GPU tests (only run when a functional GPU is present) |

## `TensorKitTestSuite`

`test/testsuite/TensorKitTestSuite.jl` is a self-contained module. It registers individually-runnable test entries via an internal `@testsuite` macro.

Run a single registered entry with:

```julia
TensorKitTestSuite.run_testsuite(group::Symbol, name::String, arg)
```

where `arg` is a `Type{<:Sector}` (for `:single_fusiontrees`, `:double_fusiontrees`, `:spaces`), a
single space (for `:diagonal_tensors`), or a 5-tuple of mutually compatible spaces (for `:tensors`
and `:factorizations`).

Downstream packages can include it standalone, without any of `TensorKit.jl`'s own test
dependencies:

```julia
import TensorKit
testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKit))), # TensorKit root
    "test", "testsuite", "TensorKitTestSuite.jl"
)
include(testsuite_path)
using .TensorKitTestSuite

TensorKitTestSuite.run_testsuite(:single_fusiontrees, "printing", MySector)
TensorKitTestSuite.run_testsuite(:tensors, "full trace", (V1, V2, V3, V4, V5))
```

Sector-level helpers (`smallset`, `randsector`, `hasfusiontensor`, `random_fusion`, `can_fuse`, `F_unitarity_test`, `R_unitarity_test`) are reused internally from `TensorKitSectors.SectorTestSuite`, but deliberately **not** re-exported from `TensorKitTestSuite`. A downstream package can thus directly `include`s `TensorKitSectors`'s own `test/testsuite.jl` as well without export ambiguities.

More information can be found in the docstrings of the `TensorKitTestSuite` module and its exported functions.

## Fast mode (`--fast`)

Skips `chainrules` and `mooncake` groups entirely, and reduces coverage in the remaining tests:

- **Sector types**: tests only `Z2Irrep`, `SU2Irrep`, `FermionParity ⊠ U1Irrep ⊠ SU2Irrep`,
  and `FibonacciAnyon` (instead of the full `sectorlist`)
- **Space lists**: tests only `(Vtr, Vℤ₂, VSU₂)` (trivial, abelian, non-abelian)

Scalar type coverage within an individual `TensorKitTestSuite` entry is generally hardcoded per entry and does not shrink in fast mode, except where an entry explicitly reads `TensorKitTestSuite.fast_tests[]`. This is so that entries remain fully self-contained for standalone/downstream use.

## `setup.jl`

Defines the `TestSetup` module, which is loaded into every worker sandbox automatically. It `include`s `test/testsuite/TensorKitTestSuite.jl` and re-exports:

- **Spaces**: `Vtr`, `Vℤ₂`, `Vfℤ₂`, `Vℤ₃`, `VU₁`, `VfU₁`, `VCU₁`, `VSU₂`, `VfSU₂`,
  `VSU₂U₁`, `Vfib`, `VIB_diag`, `VIB_M`
- **Sector lists**: `sectorlist` (full), `fast_sectorlist` (reduced)
- **Utilities**: `randsector`, `smallset`, `hasfusiontensor`, `force_planar`, `random_fusion`,
  `eval_show`, `randindextuple`, `randcircshift`, `_repartition`, `trivtuple`, `default_tol`

The `fast_tests::Bool` constant is also available in every test file (injected alongside
`TestSetup` via `init_code` in `runtests.jl`).

## Adding a new test file

Create a `.jl` file anywhere under `test/`. It is auto-discovered by `ParallelTestRunner` and
must be self-contained (worker processes have no shared state). `TestSetup` and `fast_tests` are
already in scope — no include needed.

```julia
using Test, TestExtras
using TensorKit

@testset "My tests" begin
    # fast_tests and all TestSetup exports (Vtr, sectorlist, …) are available here
end
```

## Adding a new `TensorKitTestSuite` entry

To add a new check to the exported test suite, register it with `@testsuite` in the corresponding file under `test/testsuite/`. For example, to add a new check to the `:tensors` group, add within `test/testsuite/tensors.jl`:

```julia
@testsuite :tensors "my new check" V -> begin
    V1, V2, V3, V4, V5 = V
    # ... @test ...
end
```

Then call it from the relevant worker file e.g. `test/tensors/xxx.jl` via `TensorKitTestSuite.run_testsuite(:tensors, "my new check", V)`.
