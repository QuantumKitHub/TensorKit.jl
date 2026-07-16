using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_eigh_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (EIGH): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
    atol = default_tol(T)
    rtol = default_tol(T)
    th = project_hermitian(t)
    DV = eigh_full(th)
    ΔDV = EnzymeTestUtils.rand_tangent(DV)
    remove_eigh_gauge_dependence!(ΔDV[2], DV...)
    proj_eigh_full(t) = eigh_full(project_hermitian(t))
    EnzymeTestUtils.test_reverse(proj_eigh_full, Duplicated, (th, Duplicated); output_tangent = ΔDV, atol, rtol)

    D = eigh_vals(th)
    EnzymeTestUtils.test_reverse(eigh_vals ∘ project_hermitian, Duplicated, (th, Duplicated); atol, rtol)

    V_trunc = spacetype(th)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
    trunc = truncspace(V_trunc)
    alg = MatrixAlgebraKit.select_algorithm(eigh_trunc_no_error, th, nothing; trunc)
    DVtrunc = eigh_trunc_no_error(th, alg)
    ΔDVtrunc = EnzymeTestUtils.rand_tangent(DVtrunc)
    remove_eigh_gauge_dependence!(ΔDVtrunc[2], DVtrunc...)
    proj_eigh(t, alg) = eigh_trunc_no_error(project_hermitian(t), alg)
    EnzymeTestUtils.test_reverse(proj_eigh, Duplicated, (th, Duplicated), (alg, Const); output_tangent = ΔDVtrunc, atol, rtol)
end
