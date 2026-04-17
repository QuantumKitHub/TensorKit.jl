using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

function remove_svdgauge_dependence!(
        ΔU, ΔVᴴ, U, S, Vᴴ; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(S)
    )
    UdU = U' * ΔU
    VdV = Vᴴ * ΔVᴴ'
    gaugepart = project_antihermitian!(UdU + VdV)
    for (c, b) in blocks(gaugepart)
        Sd = diagview(block(S, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Sc)) .- diagview(Sc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Sd[i] - Sd[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end

@timedtestset "Enzyme - Factorizations (SVD): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4]))
    atol = default_tol(T)
    rtol = default_tol(T)

    #S = svd_vals(t)
    #EnzymeTestUtils.test_reverse(svd_vals, Duplicated, (t, Duplicated); atol, rtol)

    USVᴴ = svd_compact(t)
    ΔUSVᴴ = EnzymeTestUtils.rand_tangent(USVᴴ)
    remove_svdgauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
    EnzymeTestUtils.test_reverse(svd_compact, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)

    #=USVᴴ = svd_full(t)
    ΔUSVᴴ = (TensorMap(randn!(similar(USVᴴ[1].data)), space(USVᴴ[1])), TensorMap(randn!(similar(USVᴴ[2].data)), space(USVᴴ[2])), TensorMap(randn!(similar(USVᴴ[3].data)), space(USVᴴ[3])))
    remove_svdgauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
    EnzymeTestUtils.test_reverse(svd_full, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)=#

    V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
    trunc = truncspace(V_trunc)
    alg = MatrixAlgebraKit.select_algorithm(svd_trunc_no_error, t, nothing; trunc)
    USVᴴtrunc = svd_trunc_no_error(t, alg)
    ΔUSVᴴtrunc = EnzymeTestUtils.rand_tangent(USVᴴtrunc)
    remove_svdgauge_dependence!(ΔUSVᴴtrunc[1], ΔUSVᴴtrunc[3], USVᴴtrunc...)
    EnzymeTestUtils.test_reverse(svd_trunc_no_error, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔUSVᴴtrunc, atol, rtol)
end
