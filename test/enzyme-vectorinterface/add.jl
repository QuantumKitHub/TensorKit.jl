using Test, TestExtras
using TensorKit, Enzyme, EnzymeTestUtils
using TensorOperations
using Random

spacelist = ad_spacelist(fast_tests)
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
