using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_svd_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (SVD): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])'))
    atol = default_tol(T)
    rtol = default_tol(T)

    #S = svd_vals(t)
    #EnzymeTestUtils.test_reverse(svd_vals, Duplicated, (t, Duplicated); atol, rtol)

    USVᴴ = svd_full(t)
    ΔUSVᴴ = EnzymeTestUtils.rand_tangent(USVᴴ)
    remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
    EnzymeTestUtils.test_reverse(svd_full, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)

    # ntuple related bug in Enzyme internals
    @static if VERSION >= v"1.11.0-rc"
        USVᴴ = svd_compact(t)
        ΔUSVᴴ = EnzymeTestUtils.rand_tangent(USVᴴ)
        remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
        EnzymeTestUtils.test_reverse(svd_compact, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)

        V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
        trunc = truncspace(V_trunc)
        alg = MatrixAlgebraKit.select_algorithm(svd_trunc_no_error, t, nothing; trunc)
        USVᴴtrunc = svd_trunc_no_error(t, alg)
        ΔUSVᴴtrunc = EnzymeTestUtils.rand_tangent(USVᴴtrunc)
        remove_svd_gauge_dependence!(ΔUSVᴴtrunc[1], ΔUSVᴴtrunc[3], USVᴴtrunc...)
        EnzymeTestUtils.test_reverse(svd_trunc_no_error, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔUSVᴴtrunc, atol, rtol)
    end
end
