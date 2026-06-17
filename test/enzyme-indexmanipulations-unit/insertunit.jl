using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset verbose = true "Enzyme - Index Manipulations (insertunit):" begin
    @timedtestset verbose = true "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, TA in (Duplicated,)
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        @testset for insertunit in (insertleftunit, insertrightunit)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(1), Const); atol, rtol)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(4), Const); atol, rtol)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A', TA), (Val(2), Const); atol, rtol)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(1), Const); atol, rtol, fkwargs = (copy = false,))
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(2), Const); atol, rtol, fkwargs = (copy = true,))
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(3), Const); atol, rtol, fkwargs = (copy = false, dual = true, conj = true))
            EnzymeTestUtils.test_reverse(insertunit, TA, (A', TA), (Val(3), Const); atol, rtol, fkwargs = (copy = false, dual = true, conj = true))
        end
    end
end
