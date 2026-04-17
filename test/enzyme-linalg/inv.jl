using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

@isdefined(TestSetup) || include("../../setup.jl")
using .TestSetup

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2)',
        Vect[FermionParity](0 => 2, 1 => 1)',
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[FermionParity](0 => 2, 1 => 2),
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

@timedtestset "Enzyme - LinearAlgebra (inv):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        D1 = randn(T, V[1] ← V[1])
        D2 = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
        D3 = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← V[1] ⊗ V[2] ⊗ V[3])
        @testset "inv: TD $TD" for TD in (Const, Duplicated)
            EnzymeTestUtils.test_reverse(inv, TD, (D1, TD); atol, rtol)
            EnzymeTestUtils.test_reverse(inv, TD, (D2, TD); atol, rtol)
            EnzymeTestUtils.test_reverse(inv, TD, (D3, TD); atol, rtol)
        end
    end
end
