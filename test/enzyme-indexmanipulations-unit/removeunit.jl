using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset verbose = true "Enzyme - Index Manipulations (removeunit):" begin
    @timedtestset verbose = true "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, TB in (Duplicated,)
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        for i in 1:2
            B = insertleftunit(A, i; dual = rand(Bool))
            EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol, fkwargs = (copy = false,))
            EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol, fkwargs = (copy = true,))
        end
    end
end
