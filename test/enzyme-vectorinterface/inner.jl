using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random, FiniteDifferences

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@testset "Enzyme - VectorInterface" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        @testset for TC in (Duplicated,), TA in (Duplicated,), f in (identity, adjoint)
            atol = default_tol(T)
            rtol = default_tol(T)
            # see https://github.com/QuantumKitHub/TensorKit.jl/issues/457
            @static if VERSION < v"1.11.0-rc"
                CV = V[1] ⊗ V[2] ← V[4] ⊗ V[5]
            else
                CV = V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5]
            end
            C = randn(T, CV)
            A = randn(T, CV)
            for RT in (Active, Const)
                EnzymeTestUtils.test_reverse(inner, RT, (f(C), TC), (f(A), TA); atol, rtol)
            end
            for RT in (Duplicated, Const)
                EnzymeTestUtils.test_forward(inner, RT, (f(C), TC), (f(A), TA); atol, rtol)
            end
        end
    end
end
