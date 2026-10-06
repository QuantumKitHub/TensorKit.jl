using ParallelTestRunner
using TensorKit

testsuite = ParallelTestRunner.find_tests(@__DIR__)

# Exclude non-test files
delete!(testsuite, "setup")          # shared setup module

# CUDA tests: only run if CUDA is functional
using CUDA: CUDA
CUDA.functional() || filter!(!startswith("cuda") ∘ first, testsuite)
# AMDGPU tests: only run if AMDGPU is functional
using AMDGPU
AMDGPU.functional() || filter!(!startswith("amd") ∘ first, testsuite)

# JET tests: JET ≥ 0.12 (Julia 1.12+) signals through `JET_AVAILABLE` whether it is functional,
# and loads empty stubs when it is not. Older JET versions, still needed on Julia < 1.12, do not
# define it, hence the `isdefined` check instead of `using JET: JET_AVAILABLE`.
using JET: JET
const jet_new_generation = isdefined(JET, :JET_AVAILABLE)
const jet_available = !jet_new_generation || JET.JET_AVAILABLE
# whole-package analysis is pinned to the JET 0.12 generation, so its reports need not be curated for several JET/Julia combinations
(jet_new_generation && jet_available) || delete!(testsuite, "other/jet")
# Mooncake's tangent tests reach JET through Mooncake's extension: any functional JET works, the empty stubs do not
jet_available || delete!(testsuite, "mooncake/tangent")

# On Buildkite (GPU CI runner): only run CUDA and AMDGPU tests
if get(ENV, "BUILDKITE", "false") == "true"
    f(str) = startswith(first(str), "cuda") || startswith(first(str), "amd")
    filter!(f, testsuite)
end

# ChainRules / Mooncake: skip on Apple CI and on Julia prerelease builds
if (Sys.isapple() && get(ENV, "CI", "false") == "true") || !isempty(VERSION.prerelease)
    filter!(!startswith("chainrules") ∘ first, testsuite)
    filter!(!startswith("mooncake") ∘ first, testsuite)
    filter!(!startswith("enzyme") ∘ first, testsuite)
end

args = parse_args(ARGS; custom = ["fast"])
# --fast: skip AD tests and inject fast_tests=true into each worker sandbox
fast = !isnothing(args.custom["fast"])

# Enzyme workers end up needing ~4 GB, but the default job count assumes ~2 GB per worker.
# Cap the number of jobs by increasing the per-worker memory budget unless --jobs was given.
selected = copy(testsuite)
ParallelTestRunner.filter_tests!(selected, args)
has_enzyme = any(startswith("enzyme"), keys(selected))
if isnothing(args.jobs) && has_enzyme
    njobs = clamp(Int(Sys.free_memory() ÷ (4 * Int64(2)^30)), 1, Sys.CPU_THREADS)
    # On Windows with Julia 1.10, memory pressure makes recycled workers too slow to start:
    # Malt gives up if a new worker does not connect within 30 s, which aborts the whole run.
    Sys.iswindows() && VERSION < v"1.11" && (njobs = min(njobs, 2))
    args = ParallelTestRunner.ParsedArgs(
        Some(njobs), args.verbose, args.quickfail, args.list, args.custom, args.positionals
    )
end
# Enzyme-compiled code is never freed, so worker RSS keeps growing with each test. Recycle
# workers well before the default 3.8 GB threshold unless JULIA_TEST_MAXRSS_MB is set.
max_worker_rss = if has_enzyme && !haskey(ENV, "JULIA_TEST_MAXRSS_MB")
    2500 * 2^20
else
    ParallelTestRunner.get_max_worker_rss()
end

setup_path = joinpath(@__DIR__, "setup.jl")
const init_worker_code = quote
    const fast_tests = $fast
    include($setup_path)
    using .TestSetup
end
const init_code = quote
    using ..TestSetup
    const fast_tests = $fast
end

ParallelTestRunner.runtests(TensorKit, args; testsuite, init_worker_code, init_code, max_worker_rss)
