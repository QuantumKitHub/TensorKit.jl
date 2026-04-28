using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@testset "Enzyme - VectorInterface (scale!)" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        α = randn(T)
        @testset for TC in (Duplicated,), Tα in (Active, Const)
            C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            EnzymeTestUtils.test_reverse(scale!, TC, (C, TC), (α, Tα); atol, rtol)
            C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            EnzymeTestUtils.test_reverse(scale!, TC, (C', TC), (α, Tα); atol, rtol)
            @testset for TA in (Duplicated,), (fc, fa) in ((identity, identity), (adjoint, adjoint))
                C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
                A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
                EnzymeTestUtils.test_reverse(scale!, TC, (fc(C), TC), (fa(A), TA), (α, Tα); atol, rtol)
            end
        end
    end
end
