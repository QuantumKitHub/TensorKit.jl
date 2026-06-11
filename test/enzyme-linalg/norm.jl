using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
rRTs = is_ci ? (Active,) : (Const, Active)
fRTs = is_ci ? (Duplicated,) : (Const, Duplicated)

@timedtestset verbose = true "Enzyme - LinearAlgebra (norm):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T), TC $TC" for V in spacelist, T in eltypes, TC in (Const, Duplicated)
        atol = default_tol(T)
        rtol = default_tol(T)
        C = randn(T, V[1] ⊗ V[2] ← V[5])
        for RT in rRTs 
            EnzymeTestUtils.test_reverse(norm, RT, (C, TC), (2, Const); atol, rtol)
            EnzymeTestUtils.test_reverse(norm, RT, (C', TC), (2, Const); atol, rtol)
        end
        for RT in fRTs
            EnzymeTestUtils.test_forward(norm, RT, (C, TC), (2, Const); atol, rtol)
            EnzymeTestUtils.test_forward(norm, RT, (C', TC), (2, Const); atol, rtol)
        end
    end
end
