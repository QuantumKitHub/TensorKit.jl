# TensorKit.jl test suite

Tests use [ParallelTestRunner.jl](https://github.com/vchuravy/ParallelTestRunner.jl) for parallel
execution. Each test file runs in its own worker process. Shared helpers are loaded automatically
via `init_code` — test files do not need to include `setup.jl` themselves.

## Running tests

```julia
# Standard — works on all supported Julia versions
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

## Test groups

| Group | Contents |
|-------|----------|
| `symmetries` | Spaces and fusion trees |
| `tensors` | Core tensor operations, factorizations, planar tensors, diagonal tensors |
| `other` | Aqua code-quality checks, bug-fix regressions |
| `chainrules` | ChainRulesCore AD tests |
| `mooncake` | Mooncake AD tests |
| `cuda`/`amd` | GPU tests (only run when a functional GPU is present) |

## `TensorKitTestSuite`

`test/testsuite/TensorKitTestSuite.jl` is a self-contained module.
It registers individually-runnable tests through plain functions, wrapping its checks in `@testset`s.
Nothing is exported, so call these with the module
prefix.

Downstream packages can include it standalone, without any of `TensorKit.jl`'s other test
dependencies:

```julia
import TensorKit
testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKit))), # TensorKit root
    "test", "testsuite", "TensorKitTestSuite.jl"
)
include(testsuite_path)

TensorKitTestSuite.test_graded_space(MySector)
```

Sector-level helpers (`randsector`, `hasfusiontensor`) are reused internally from
`TensorKitSectors.SectorTestSuite`, but deliberately **not** re-exported from
`TensorKitTestSuite`, so a downstream package can also `include` `TensorKitSectors`'s own
`test/testsuite.jl` without name clashes.

To add a new check, define a new `test_*` function in the corresponding file under
`test/testsuite/`, and call it from the relevant test file, e.g. `test/symmetries/spaces.jl`.

## Fast mode (`--fast`)

Skips `chainrules` and `mooncake` groups entirely, and reduces coverage in the remaining tests:

- **Sector types**: tests only `Z2Irrep`, `SU2Irrep`, `FermionParity ⊠ U1Irrep ⊠ SU2Irrep`,
  and `FibonacciAnyon` (instead of the full `sectorlist`)
- **Space lists**: tests only `(Vtr, Vℤ₂, VSU₂)` (trivial, abelian, non-abelian)
- **Scalar types**: tests only `Float64` and `ComplexF64` (instead of all integer/float variants)

## `setup.jl`

Defines the `TestSetup` module, which is loaded into every worker sandbox automatically. It
`include`s `test/testsuite/TensorKitTestSuite.jl` and exports:

- **Spaces**: `Vtr`, `Vℤ₂`, `Vfℤ₂`, `Vℤ₃`, `VU₁`, `VfU₁`, `VCU₁`, `VSU₂`, `VfSU₂`,
  `VSU₂U₁`, `Vfib`, `VIB_diag`, `VIB_M`
- **Sector lists**: `sectorlist` (full), `fast_sectorlist` (reduced)
- **Utilities**: `randsector`, `smallset`, `hasfusiontensor`, `force_planar`, `random_fusion`,
  `randindextuple`, `randcircshift`, `_repartition`, `trivtuple`, `test_dim_isapprox`, `default_tol`
- **Test suite**: the `TensorKitTestSuite` module itself, and its `eval_show` utility

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
