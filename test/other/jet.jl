using TensorKit
using Test
using JET

# also gated in runtests.jl; JET < 0.12 does not define `JET_AVAILABLE`
if isdefined(JET, :JET_AVAILABLE) && JET.JET_AVAILABLE
    JET.test_package(TensorKit; target_modules = (TensorKit,))
else
    @info "Full JET functionality is unavailable on Julia $VERSION; skipping JET analysis"
end
