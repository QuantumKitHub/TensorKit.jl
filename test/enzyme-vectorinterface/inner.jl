using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random, FiniteDifferences

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@testset "Enzyme - VectorInterface" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        @testset for TC in (Duplicated, Const), TA in (Duplicated, Const), f in (identity, adjoint)
            atol = default_tol(T)
            rtol = default_tol(T)
            C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            for RT in (Active, Const)
                EnzymeTestUtils.test_reverse(inner, RT, (f(C), TC), (f(A), TA); atol, rtol)
            end
            for RT in (Duplicated, Const)
                EnzymeTestUtils.test_forward(inner, RT, (f(C), TC), (f(A), TA); atol, rtol)
            end
        end
    end
end
