using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"

Tαs = is_ci ? (Active,) : (Active, Const)
Tβs = is_ci ? (Active,) : (Active, Const)

@timedtestset "Enzyme - Index Manipulations (add_transpose!):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        α = randn(T)
        β = randn(T)

        # repeat a couple times to get some distribution of arrows
        p = randcircshift(numout(A), numin(A))
        C = randn!(transpose(A, p))
        EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
        @testset for Tα in Tαs, Tβ in Tβs
            EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
            if !(T <: Real) && !is_ci
                EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
            end
        end
    end
end
