using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_eig_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (EIG): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
    atol = default_tol(T)
    rtol = default_tol(T)
    DV = eig_full(t)
    ΔDV = EnzymeTestUtils.rand_tangent(DV)
    remove_eig_gauge_dependence!(ΔDV[2], DV...)
    EnzymeTestUtils.test_reverse(eig_full, Duplicated, (t, Duplicated); output_tangent = ΔDV, atol, rtol)

    #D = eig_vals(t)
    #EnzymeTestUtils.test_reverse(eig_vals, Duplicated, (t, Duplicated); atol, rtol)

    V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
    trunc = truncspace(V_trunc)
    alg = MatrixAlgebraKit.select_algorithm(eig_trunc_no_error, t, nothing; trunc)
    DVtrunc = eig_trunc_no_error(t, alg)
    ΔDVtrunc = EnzymeTestUtils.rand_tangent(DVtrunc)
    remove_eig_gauge_dependence!(ΔDVtrunc[2], DVtrunc...)
    EnzymeTestUtils.test_reverse(eig_trunc_no_error, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔDVtrunc, atol, rtol)
end
