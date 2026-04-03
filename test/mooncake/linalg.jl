using Test, TestExtras
using TensorKit
using Mooncake
using Random


mode = Mooncake.ReverseMode
rng = Random.default_rng()

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Mooncake - LinearAlgebra: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    C = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
    A = randn(T, codomain(C) ← V[5]' ⊗ V[4]')
    B = randn(T, domain(A) ← domain(C))
    α = randn(T)
    β = randn(T)

    Mooncake.TestUtils.test_rule(rng, mul!, C, A, B, α, β; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, mul!, C, A, B; atol, rtol, mode, is_primitive = false)

    Mooncake.TestUtils.test_rule(rng, norm, C, 2; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, norm, C', 2; atol, rtol, mode)

    D1 = randn(T, V[1] ← V[1])
    D2 = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
    D3 = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← V[1] ⊗ V[2] ⊗ V[3])

    Mooncake.TestUtils.test_rule(rng, tr, D1; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, tr, D2; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, tr, D3; atol, rtol, mode)

    Mooncake.TestUtils.test_rule(rng, inv, D1; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, inv, D2; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, inv, D3; atol, rtol, mode)
end
