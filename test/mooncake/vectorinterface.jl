using Test, TestExtras
using TensorKit
using TensorOperations
using Mooncake
using Random

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

mode = Mooncake.ReverseMode
rng = Random.default_rng()

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
    (
        Vect[Z2Irrep](0 => 1, 1 => 1),
        Vect[Z2Irrep](0 => 1, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 3),
        Vect[Z2Irrep](0 => 2, 1 => 2),
    ),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2)',
        Vect[FermionParity](0 => 2, 1 => 1)',
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
    (
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
        Vect[U1Irrep](0 => 1, 1 => 1, -1 => 2),
        Vect[U1Irrep](0 => 1, 1 => 2, -1 => 1)',
    ),
    (
        Vect[SU2Irrep](0 => 2, 1 // 2 => 1),
        Vect[SU2Irrep](0 => 1, 1 => 1),
        Vect[SU2Irrep](1 // 2 => 1, 1 => 1)',
        Vect[SU2Irrep](1 // 2 => 2),
        Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
    ),
    (
        Vect[FibonacciAnyon](:I => 2, :τ => 1),
        Vect[FibonacciAnyon](:I => 1, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 3),
        Vect[FibonacciAnyon](:I => 2, :τ => 2),
    ),
)
eltypes = (Float64, ComplexF64)

@timedtestset "Mooncake - VectorInterface: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
    A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
    α = randn(T)
    β = randn(T)

    Mooncake.TestUtils.test_rule(rng, scale!, C, α; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, scale!, C', α; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, scale!, C, A, α; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, scale!, C', A', α; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, scale!, copy(C'), A', α; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, scale!, C', copy(A'), α; atol, rtol, mode)

    Mooncake.TestUtils.test_rule(rng, add!, C, A; atol, rtol, mode, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, add!, C, A, α; atol, rtol, mode, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, add!, C, A, α, β; atol, rtol, mode)

    Mooncake.TestUtils.test_rule(rng, inner, C, A; atol, rtol, mode)
    Mooncake.TestUtils.test_rule(rng, inner, C', A'; atol, rtol, mode)
end
