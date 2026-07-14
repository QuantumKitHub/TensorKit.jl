using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"

rTαs = is_ci ? (Active,) : (Active, Const)
rTβs = is_ci ? (Active,) : (Active, Const)
fTαs = is_ci ? (Duplicated,) : (Duplicated, Const)
fTβs = is_ci ? (Duplicated,) : (Duplicated, Const)

if VERSION > v"1.11.0-rc" # https://github.com/QuantumKitHub/TensorKit.jl/issues/457
    @timedtestset "Enzyme - Index Manipulations (transpose!) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        α = randn(T)
        β = randn(T)

        # repeat a couple times to get some distribution of arrows
        p = randcircshift(numout(A), numin(A))
        C = randn!(transpose(A, p))
        !is_ci && EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
        !is_ci && EnzymeTestUtils.test_forward(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
        @testset for Tα in rTαs, Tβ in rTβs
            EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
            if !(T <: Real) && !is_ci
                EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
            end
        end
        @testset for Tα in fTαs, Tβ in fTβs
            EnzymeTestUtils.test_forward(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
            if !(T <: Real) && !is_ci
                EnzymeTestUtils.test_forward(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                EnzymeTestUtils.test_forward(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                EnzymeTestUtils.test_forward(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
            end
        end
    end
end
