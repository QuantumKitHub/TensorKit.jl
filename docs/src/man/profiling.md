# [Profiling and timers](@id s_profiling)

TensorKit's index manipulations, tensor contractions and factorizations are instrumented with [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) sections that are compiled away by default, so they incur no runtime cost.
They can be enabled to obtain a detailed breakdown of where time is spent inside these operations, in particular the split between the different kinds of work involved in manipulating symmetric tensors:

| category | contents |
|:---|:---|
| `symmetry` | fusion tree manipulations and recoupling coefficients (braiding, transposing, F- and R-symbols) |
| `bookkeeping` | block structure computations, cache lookups, contraction planning |
| `alloc` | allocation of output tensors and temporary buffers |
| `dense` | dense tensor kernels (BLAS/LAPACK calls, strided permutations and additions) |
| `other` | remaining time within an instrumented operation (dispatch, argument checking, uncovered overhead) |

The canonical workflow looks as follows:

```julia
using TensorKit

TensorKit.enable_timers!()      # triggers recompilation of the instrumented methods

# warm up first, so that compilation does not pollute the timings
V = SU2Space(0 => 4, 1//2 => 4, 1 => 2)
t = rand(V ⊗ V ← V ⊗ V)
permute(t, ((1, 3), (2, 4)))
@tensor t2[a b; c d] := t[a x; c y] * t[y b; x d]
svd_compact(t)

TensorKit.reset_timers!()
# ... run the workload of interest ...
TensorKit.print_timers()        # full nested call tree
TensorKit.timer_summary()       # symmetry / bookkeeping / alloc / dense / other totals

TensorKit.disable_timers!()
```

[`TensorKit.print_timers`](@ref) displays the accumulated timings as a nested call tree, with sections for the top-level operations (`"permute!"`, `"contract!"`, `"svd_compact!"`, ...) and nested sections labeled by their category prefix (`"symmetry: ..."`, `"bookkeeping: ..."`, `"alloc: ..."`, `"dense: ..."`).
[`TensorKit.timer_summary`](@ref) aggregates the *exclusive* time of each section (its own time minus that of its timed children) into per-category totals, such that every nanosecond is counted exactly once and the totals sum to the total measured time.

A few caveats to keep in mind:

*   Enabling or disabling the timers redefines internal functions, so instrumented methods recompile on first use afterwards.
    This is a debug-session switch, not a runtime option.
*   While timers are enabled, TensorKit-internal task parallelism is disabled, since the timer object may only be manipulated from a single task.
    As a consequence, multi-threaded speedups are not measurable while timing, and TensorKit functions should not be called concurrently from multiple user tasks.
*   Each section entry costs roughly 100–200 ns, which can distort measurements of workloads on very small tensors.
*   The construction of fusion tree transformers and block structures is cached (see `empty_globalcaches!`), so their cost only shows up the first time a given structure is encountered.
    Call `TensorKit.empty_globalcaches!()` before the measurement if you want the construction cost to be included, or after the warm-up if you want to measure the steady-state behavior with warm caches.
*   For GPU tensors, the timings only reflect host-side dispatch of asynchronous kernels, unless the workload is explicitly synchronized.

## Library documentation

```@docs
TensorKit.GLOBAL_TIMER
TensorKit.enable_timers!
TensorKit.disable_timers!
TensorKit.reset_timers!
TensorKit.print_timers
TensorKit.timer_summary
TensorKit.timer
TensorKit.timers_enabled
```
