using Test, TestExtras
using TensorKit
using TensorOperations
using Mooncake
using Random
using VectorInterface

rng = Random.default_rng()

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Mooncake - VectorInterface: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    C = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
    A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
    for α in (randn(T), One(), Zero()), β in (randn(T), One(), Zero())
        Mooncake.TestUtils.test_rule(rng, scale!, C, α; atol, rtol)
        Mooncake.TestUtils.test_rule(rng, scale!, C', α; atol, rtol)
        Mooncake.TestUtils.test_rule(rng, scale!, C, A, α; atol, rtol)
        Mooncake.TestUtils.test_rule(rng, scale!, C', A', α; atol, rtol)
        Mooncake.TestUtils.test_rule(rng, scale!, copy(C'), A', α; atol, rtol)
        Mooncake.TestUtils.test_rule(rng, scale!, C', copy(A'), α; atol, rtol)

        Mooncake.TestUtils.test_rule(rng, add!, C, A; atol, rtol, is_primitive = false)
        Mooncake.TestUtils.test_rule(rng, add!, C, A, α; atol, rtol, is_primitive = false)
        Mooncake.TestUtils.test_rule(rng, add!, C, A, α, β; atol, rtol)

        Mooncake.TestUtils.test_rule(rng, inner, C, A; atol, rtol)
        Mooncake.TestUtils.test_rule(rng, inner, C', A'; atol, rtol)
    end
end
