# define the `@timeit_debug` switch explicitly (instead of letting the first macro
# expansion do it) so that ordinary code can branch on it, see `timers_enabled`
timeit_debug_enabled() = false

"""
    TensorKit.GLOBAL_TIMER

The global `TimerOutput` object into which all timer sections of TensorKit accumulate.
See [`enable_timers!`](@ref) for how to activate them, [`print_timers`](@ref) for
displaying the resulting call tree and [`timer_summary`](@ref) for aggregating it into
per-category totals.
"""
const GLOBAL_TIMER = TimerOutput("TensorKit")

"""
    TensorKit.timer() -> TimerOutputs.TimerOutput

Return the global timer object [`GLOBAL_TIMER`](@ref) into which all TensorKit timer
sections accumulate.
"""
timer() = GLOBAL_TIMER

"""
    TensorKit.timers_enabled() -> Bool

Return whether the `@timeit_debug` timer sections of TensorKit are currently compiled in,
i.e. whether [`enable_timers!`](@ref) has been called.

This is used internally to force serial execution of parallel regions while timing, since
a `TimerOutput` may only be manipulated from a single task. When timers are disabled this
check const-folds to `false`, so it has no runtime cost.
"""
timers_enabled() = timeit_debug_enabled()

"""
    TensorKit.enable_timers!()

Enable all timer sections of TensorKit (including its submodules), which accumulate
timings of the internal kernels into [`GLOBAL_TIMER`](@ref). Undone by
[`disable_timers!`](@ref).

!!! warning
    Enabling or disabling timers redefines internal functions and therefore triggers
    recompilation of the instrumented methods on first use. This is a debug-session
    operation, not a runtime switch.

!!! warning
    While timers are enabled, TensorKit-internal task parallelism is disabled (threaded
    regions run serially), so multi-threaded speedups are not measurable. Additionally,
    TensorKit functions should not be called concurrently from multiple user tasks while
    timing, as the timer object is not thread-safe.

Note that each timer section adds an overhead of roughly 100-200 ns per entry, which can
distort measurements of very small workloads. For GPU tensors, timings only reflect
host-side dispatch of asynchronous kernels unless the workload is explicitly synchronized.
"""
function enable_timers!()
    TimerOutputs.enable_debug_timings(TensorKit)
    return nothing
end

"""
    TensorKit.disable_timers!()

Disable all timer sections of TensorKit again; the inverse of [`enable_timers!`](@ref).
Also triggers recompilation of the instrumented methods on first use.
"""
function disable_timers!()
    TimerOutputs.disable_debug_timings(TensorKit)
    return nothing
end

"""
    TensorKit.reset_timers!()

Reset the accumulated timings in [`GLOBAL_TIMER`](@ref).
"""
function reset_timers!()
    TimerOutputs.reset_timer!(GLOBAL_TIMER)
    return nothing
end

"""
    TensorKit.print_timers(io::IO = stdout; kwargs...)

Print the accumulated timings in [`GLOBAL_TIMER`](@ref) as a nested table. Keyword
arguments are forwarded to `TimerOutputs.print_timer`.
"""
print_timers(io::IO = stdout; kwargs...) = TimerOutputs.print_timer(io, GLOBAL_TIMER; kwargs...)

const TIMER_CATEGORIES = (:symmetry, :bookkeeping, :alloc, :dense, :other)

# "symmetry: recoupling" -> :symmetry; unprefixed or unknown prefix -> nothing
function _timer_category(label::String)
    i = findfirst(':', label)
    i === nothing && return nothing
    prefix = Symbol(label[1:prevind(label, i)])
    return prefix in TIMER_CATEGORIES ? prefix : nothing
end

# category of the miss-path (construction) section of an `@cached` function
function _cached_category(fname::Symbol)
    return fname in (:fsbraid, :fstranspose, :treebraider, :treetransposer) ?
        "symmetry" : "bookkeeping"
end

const TimerSummary = Dict{Symbol, @NamedTuple{time::Int64, allocated::Int64, ncalls::Int64}}

"""
    TensorKit.timer_summary([io::IO]; to = GLOBAL_TIMER)
        -> Dict{Symbol, @NamedTuple{time::Int64, allocated::Int64, ncalls::Int64}}

Aggregate the timer tree into per-category totals (time in ns, allocated bytes, number of
section entries) for the categories `$(TIMER_CATEGORIES)`.

Each section's *exclusive* time (its own time minus that of its timed children) is
attributed to the category given by its label prefix or, for unprefixed labels, to the
category of the nearest categorized ancestor (`:other` at the root). Every nanosecond is
thus counted exactly once and the totals sum to the total measured time.

When `io` is given (default `stdout`), a small table is printed; pass `nothing` to skip
printing and only return the totals.
"""
function timer_summary(io::Union{IO, Nothing} = stdout; to::TimerOutput = GLOBAL_TIMER)
    totals = TimerSummary(c => (time = 0, allocated = 0, ncalls = 0) for c in TIMER_CATEGORIES)
    for child in to.root.children
        _accumulate_summary!(totals, child, :other)
    end
    io === nothing || _print_timer_summary(io, totals)
    return totals
end

function _accumulate_summary!(totals::TimerSummary, s, inherited::Symbol)
    cat = something(_timer_category(s.name), inherited)
    t = TimerOutputs.time(s)
    b = TimerOutputs.allocated(s)
    for child in s.children
        t -= TimerOutputs.time(child)
        b -= TimerOutputs.allocated(child)
        _accumulate_summary!(totals, child, cat)
    end
    old = totals[cat]
    totals[cat] = (
        time = old.time + max(t, 0), allocated = old.allocated + max(b, 0),
        ncalls = old.ncalls + TimerOutputs.ncalls(s),
    )
    return nothing
end

function _print_timer_summary(io::IO, totals::TimerSummary)
    total_time = sum(x -> x.time, values(totals))
    println(io, "TensorKit timer summary:")
    for cat in TIMER_CATEGORIES
        (; time, allocated, ncalls) = totals[cat]
        percentage = total_time == 0 ? 0.0 : 100 * time / total_time
        @printf(
            io, "%12s: %s (%5.1f%%)  %s  %d sections\n",
            cat, TimerOutputs.prettytime(time), percentage,
            TimerOutputs.prettymemory(allocated), ncalls
        )
    end
    return nothing
end
