using Test, TestExtras
using TensorKit, Enzyme, EnzymeTestUtils
using TensorOperations
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@testset "Enzyme - VectorInterface (add!) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    α = randn(T)
    β = randn(T)

    # see https://github.com/QuantumKitHub/TensorKit.jl/issues/457
    if VERSION < v"1.11.0-rc" && sectortype(eltype(V)) == Trivial
        CV = V[1] ⊗ V[2] ← V[4] ⊗ V[5]
    else
        CV = V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5]
    end
    C = randn(T, CV)
    A = randn(T, CV)
    for TC in (Duplicated,), TA in (Duplicated,)
        C = randn(T, CV)
        A = randn(T, CV)
        EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA); atol, rtol, testset_name = "add! reverse TC $TC TA $TA no α no β")
        EnzymeTestUtils.test_forward(add!, TC, (C, TC), (A, TA); atol, rtol, testset_name = "add! forward TC $TC TA $TA no α no β")
        for Tα in (Active, Const)
            C = randn(T, CV)
            A = randn(T, CV)
            EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA), (α, Tα); atol, rtol, testset_name = "add! reverse TC $TC TA $TA Tα $Tα no β")
            for Tβ in (Active, Const)
                C = randn(T, CV)
                A = randn(T, CV)
                EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add! reverse TC $TC TA $TA Tα $Tα Tβ $Tβ")
            end
        end
        for Tα in (Duplicated, Const)
            C = randn(T, CV)
            A = randn(T, CV)
            EnzymeTestUtils.test_forward(add!, TC, (C, TC), (A, TA), (α, Tα); atol, rtol, testset_name = "add! forward TC $TC TA $TA Tα $Tα no β")
            for Tβ in (Duplicated, Const)
                C = randn(T, CV)
                A = randn(T, CV)
                EnzymeTestUtils.test_forward(add!, TC, (C, TC), (A, TA), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add! forward TC $TC TA $TA Tα $Tα Tβ $Tβ")
            end
        end
    end
end

@testset "Enzyme - VectorInterface (scale!) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    α = randn(T)
    # see https://github.com/QuantumKitHub/TensorKit.jl/issues/457
    if VERSION < v"1.11.0-rc" && sectortype(eltype(V)) == Trivial
        CV = V[1] ⊗ V[2] ← V[4] ⊗ V[5]
    else
        CV = V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5]
    end
    @testset for TC in (Duplicated,)
        for Tα in (Active, Const)
            C = randn(T, CV)
            EnzymeTestUtils.test_reverse(scale!, TC, (C, TC), (α, Tα); atol, rtol)
            C = randn(T, CV)
            EnzymeTestUtils.test_reverse(scale!, TC, (C', TC), (α, Tα); atol, rtol)
            @testset for TA in (Duplicated,), (fc, fa) in ((identity, identity), (adjoint, adjoint))
                C = randn(T, CV)
                A = randn(T, CV)
                EnzymeTestUtils.test_reverse(scale!, TC, (fc(C), TC), (fa(A), TA), (α, Tα); atol, rtol)
            end
        end
        for Tα in (Duplicated, Const)
            C = randn(T, CV)
            EnzymeTestUtils.test_forward(scale!, TC, (C, TC), (α, Tα); atol, rtol)
            C = randn(T, CV)
            EnzymeTestUtils.test_forward(scale!, TC, (C', TC), (α, Tα); atol, rtol)
            @testset for TA in (Duplicated,), (fc, fa) in ((identity, identity), (adjoint, adjoint))
                C = randn(T, CV)
                A = randn(T, CV)
                EnzymeTestUtils.test_forward(scale!, TC, (fc(C), TC), (fa(A), TA), (α, Tα); atol, rtol)
            end
        end
    end
end

@testset "Enzyme - VectorInterface (inner) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    @testset for TC in (Duplicated,), TA in (Duplicated,), f in (identity, adjoint)
        atol = default_tol(T)
        rtol = default_tol(T)
        # see https://github.com/QuantumKitHub/TensorKit.jl/issues/457
        if VERSION < v"1.11.0-rc" && sectortype(eltype(V)) == Trivial
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
