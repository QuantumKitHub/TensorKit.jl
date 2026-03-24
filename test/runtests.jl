using ParallelTestRunner
using TensorKit

testsuite = ParallelTestRunner.find_tests(@__DIR__)

# Exclude non-test files
delete!(testsuite, "setup")          # shared setup module
delete!(testsuite, "braidingtensor") # not part of the testsuite (see file header)

# CUDA tests: only run if CUDA is functional
using CUDA: CUDA
CUDA.functional() || filter!(k -> !startswith(k, "cuda"), testsuite)

# On Buildkite (GPU CI runner): only run CUDA tests
get(ENV, "BUILDKITE", "false") == "true" && filter!(k -> startswith(k, "cuda"), testsuite)

# ChainRules / Mooncake: skip on Apple CI and on Julia prerelease builds
if (Sys.isapple() && get(ENV, "CI", "false") == "true") || !isempty(VERSION.prerelease)
    filter!(k -> !startswith(k, "chainrules") && !startswith(k, "mooncake"), testsuite)
end

# --fast: skip AD tests and inject fast_tests=true into each worker sandbox
fast = "--fast" in ARGS
filtered_args = filter(!=("--fast"), ARGS)
if fast
    filter!(k -> !startswith(k, "chainrules") && !startswith(k, "mooncake"), testsuite)
end
setup_path = joinpath(@__DIR__, "setup.jl")
init_code = quote
    const fast_tests = $fast
    include($setup_path)
    using .TestSetup
end

ParallelTestRunner.runtests(TensorKit, filtered_args; testsuite, init_code)
