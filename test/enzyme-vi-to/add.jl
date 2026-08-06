using Test, TestExtras
using TensorKit, Enzyme, EnzymeTestUtils
using TensorOperations
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"

rTαs = is_ci ? (Active,) : (Active, Const)
rTβs = is_ci ? (Active,) : (Active, Const)
fTαs = is_ci ? (Duplicated,) : (Duplicated, Const)
fTβs = is_ci ? (Duplicated,) : (Duplicated, Const)

@testset "Enzyme - VectorInterface (add!) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    α = randn(T)
    β = randn(T)

    CV = V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5]
    C = randn(T, CV)
    A = randn(T, CV)
    for TC in (Duplicated,), TA in (Duplicated,)
        C = randn(T, CV)
        A = randn(T, CV)
        EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA); atol, rtol, testset_name = "add! reverse TC $TC TA $TA no α no β")
        EnzymeTestUtils.test_forward(add!, TC, (C, TC), (A, TA); atol, rtol, testset_name = "add! forward TC $TC TA $TA no α no β")
        for Tα in rTαs
            C = randn(T, CV)
            A = randn(T, CV)
            EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA), (α, Tα); atol, rtol, testset_name = "add! reverse TC $TC TA $TA Tα $Tα no β")
            for Tβ in rTβs
                C = randn(T, CV)
                A = randn(T, CV)
                EnzymeTestUtils.test_reverse(add!, TC, (C, TC), (A, TA), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add! reverse TC $TC TA $TA Tα $Tα Tβ $Tβ")
            end
        end
        for Tα in fTαs
            C = randn(T, CV)
            A = randn(T, CV)
            EnzymeTestUtils.test_forward(add!, TC, (C, TC), (A, TA), (α, Tα); atol, rtol, testset_name = "add! forward TC $TC TA $TA Tα $Tα no β")
            for Tβ in fTβs
                C = randn(T, CV)
                A = randn(T, CV)
                EnzymeTestUtils.test_forward(add!, TC, (C, TC), (A, TA), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add! forward TC $TC TA $TA Tα $Tα Tβ $Tβ")
            end
        end
    end
end
