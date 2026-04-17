using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Index Manipulations (flip):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T) TA ($TA)" for V in spacelist, T in eltypes, TA in (Duplicated,)
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
        EnzymeTestUtils.test_reverse(flip, TA, (A, TA), [1, 3]; atol, rtol, fkwargs = (inv = true,))
        EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol)
        EnzymeTestUtils.test_reverse(flip, TA, (A, TA), ([1, 3], Const); atol, rtol)
    end
end
