using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (PROJECTIONS): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
    atol = default_tol(T)
    rtol = default_tol(T)
    EnzymeTestUtils.test_reverse(project_hermitian, Duplicated, (t, Duplicated); atol, rtol)
    EnzymeTestUtils.test_reverse(project_antihermitian, Duplicated, (t, Duplicated); atol, rtol)
end
