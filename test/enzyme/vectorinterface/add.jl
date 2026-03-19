using Test, TestExtras
using TensorKit, Enzyme, EnzymeTestUtils, FiniteDifferences
using TensorOperations
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

@testset "Enzyme - VectorInterface (add!) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
    A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
    α = randn(T)
    β = randn(T)

    for TC in (Duplicated, Const), TA in (Duplicated, Const)
        C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA); atol, rtol, testset_name = "add! TC $TC TA $TA no α no β")
        for Tα in (Active, Const)
            C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA), (α, Tα); atol, rtol, testset_name = "add! TC $TC TA $TA Tα $Tα no β")
            for Tβ in (Active, Const)
                C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
                A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
                EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add! TC $TC TA $TA Tα $Tα Tβ $Tβ")
            end
        end
    end
end
