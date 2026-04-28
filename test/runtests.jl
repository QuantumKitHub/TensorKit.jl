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

# On Buildkite (GPU CI runner): only run CUDA and AMDGPU tests
if get(ENV, "BUILDKITE", "false") == "true"
    f(str) = startswith(first(str), "cuda") || startswith(first(str), "amd")
    filter!(f, testsuite)
end

# ChainRules / Mooncake: skip on Apple CI and on Julia prerelease builds
if (Sys.isapple() && get(ENV, "CI", "false") == "true") || !isempty(VERSION.prerelease)
    filter!(!startswith("chainrules") ∘ first, testsuite)
    filter!(!startswith("mooncake") ∘ first, testsuite)
end

args = parse_args(ARGS; custom = ["fast"])
# --fast: skip AD tests and inject fast_tests=true into each worker sandbox
fast = !isnothing(args.custom["fast"])

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

ParallelTestRunner.runtests(TensorKit, args; init_worker_code, init_code)
