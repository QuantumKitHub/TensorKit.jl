using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random, FiniteDifferences

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"

rTs = is_ci ? (Active,) : (Active, Const)
fTs = is_ci ? (Duplicated,) : (Duplicated, Const)

@testset "Enzyme - VectorInterface" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        @testset for TC in (Duplicated,), TA in (Duplicated,), f in (identity, adjoint)
            atol = default_tol(T)
            rtol = default_tol(T)
            CV = V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5]
            C = randn(T, CV)
            A = randn(T, CV)
            for RT in rTs
                EnzymeTestUtils.test_reverse(inner, RT, (f(C), TC), (f(A), TA); atol, rtol)
            end
            for RT in fTs
                EnzymeTestUtils.test_forward(inner, RT, (f(C), TC), (f(A), TA); atol, rtol)
            end
        end
    end
end
