using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Index Manipulations (twist):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, TA in (Duplicated,)
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        if !(T <: Real && !(sectorscalartype(sectortype(A)) <: Real))
            EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
            EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
            EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), (1, Const); atol, rtol)
            EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol)
        end
    end
end
