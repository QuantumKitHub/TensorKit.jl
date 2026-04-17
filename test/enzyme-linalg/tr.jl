using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

is_ci = get(ENV, "CI", "false") == "true"

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

RTs = is_ci ? (Active,) : (Const, Active)
TDs = is_ci ? (Duplicated,) : (Const, Duplicated)

@timedtestset verbose = true "Enzyme - LinearAlgebra (tr):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        D1 = randn(T, V[1] ← V[1])
        D2 = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
        D3 = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← V[1] ⊗ V[2] ⊗ V[3])
        @testset "tr: RT $RT, TD $TD" for RT in RTs, TD in TDs
            EnzymeTestUtils.test_reverse(tr, RT, (D1, TD); atol, rtol)
            EnzymeTestUtils.test_reverse(tr, RT, (D2, TD); atol, rtol)
            EnzymeTestUtils.test_reverse(tr, RT, (D3, TD); atol, rtol)
        end
    end
end
