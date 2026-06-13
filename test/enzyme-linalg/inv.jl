using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
TDs = is_ci ? (Duplicated,) : (Const, Duplicated)

@timedtestset "Enzyme - LinearAlgebra (inv):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        @testset "inv: TD $TD" for TD in TDs
            if !is_ci
                D1 = randn(T, V[1] ← V[1])
                D2 = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
                EnzymeTestUtils.test_reverse(inv, TD, (D1, TD); atol, rtol)
                EnzymeTestUtils.test_reverse(inv, TD, (D2, TD); atol, rtol)
                EnzymeTestUtils.test_forward(inv, TD, (D1, TD); atol, rtol)
                EnzymeTestUtils.test_forward(inv, TD, (D2, TD); atol, rtol)
            end
            D3 = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← V[1] ⊗ V[2] ⊗ V[3])
            EnzymeTestUtils.test_reverse(inv, TD, (D3, TD); atol, rtol)
            EnzymeTestUtils.test_forward(inv, TD, (D3, TD); atol, rtol)
        end
    end
end
