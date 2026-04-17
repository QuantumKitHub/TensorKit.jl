using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset verbose = true "Enzyme - LinearAlgebra (norm):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T) RT $RT, TC $TC" for V in spacelist, T in eltypes, RT in (Const, Active), TC in (Const, Duplicated)
        atol = default_tol(T)
        rtol = default_tol(T)
        C = randn(T, V[1] ⊗ V[2] ← V[5])
        EnzymeTestUtils.test_reverse(norm, RT, (C, TC), (2, Const); atol, rtol)
        EnzymeTestUtils.test_reverse(norm, RT, (C', TC), (2, Const); atol, rtol)
    end
end
