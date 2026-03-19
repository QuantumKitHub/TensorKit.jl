using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
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

@timedtestset "Enzyme - Index Manipulations (add_transpose!):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T) Tα $Tα Tβ $Tβ" for V in spacelist, T in eltypes, Tα in (Const, Active), Tβ in (Const, Active)
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        α = randn(T)
        β = randn(T)

        # repeat a couple times to get some distribution of arrows
        p = randcircshift(numout(A), numin(A))
        C = randn!(transpose(A, p))
        EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
        EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
        if !(T <: Real)
            EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
            EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
            EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
        end
    end
end
