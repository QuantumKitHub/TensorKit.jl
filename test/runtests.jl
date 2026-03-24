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

ParallelTestRunner.runtests(TensorKit, ARGS; testsuite)
