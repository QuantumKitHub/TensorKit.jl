using Test, TestExtras
using TensorKit
using Mooncake
using Random
using JET, AllocCheck

mode = Mooncake.ReverseMode
rng = Random.default_rng()

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

# `test_data` reaches JET through Mooncake's extension, so it needs a functional JET; this is also
# gated in runtests.jl. JET < 0.12 does not define `JET_AVAILABLE`.
jet_available = !isdefined(JET, :JET_AVAILABLE) || JET.JET_AVAILABLE

# only run on Linux since allocation tests are broken on other versions
jet_available && Sys.islinux() && @timedtestset "Mooncake - Tangent type: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    A = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← (V[4] ⊗ V[5])')
    Mooncake.TestUtils.test_data(rng, A)

    D = DiagonalTensorMap{T}(undef, V[1])
    Mooncake.TestUtils.test_data(rng, D)
end
