using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_svd_gauge_dependence!
using MatrixAlgebraKit: remove_eig_gauge_dependence!
using MatrixAlgebraKit: remove_eigh_gauge_dependence!
using MatrixAlgebraKit: remove_lq_gauge_dependence!, remove_lq_null_gauge_dependence!
using MatrixAlgebraKit: remove_qr_gauge_dependence!, remove_qr_null_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

is_ci = get(ENV, "CI", "false") == "true"

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]), randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])'))
    atol = default_tol(T)
    rtol = default_tol(T)

    @testset "SVD" begin
        if !is_ci
            S = svd_vals(t)
            EnzymeTestUtils.test_reverse(svd_vals, Duplicated, (t, Duplicated); atol, rtol)

            USVᴴ = svd_full(t)
            ΔUSVᴴ = EnzymeTestUtils.rand_tangent(USVᴴ)
            remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
            EnzymeTestUtils.test_reverse(svd_full, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)

            USVᴴ = svd_compact(t)
            ΔUSVᴴ = EnzymeTestUtils.rand_tangent(USVᴴ)
            remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
            EnzymeTestUtils.test_reverse(svd_compact, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)
        end

        V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
        trunc = truncspace(V_trunc)
        alg = MatrixAlgebraKit.select_algorithm(svd_trunc_no_error, t, nothing; trunc)
        USVᴴtrunc = svd_trunc_no_error(t, alg)
        ΔUSVᴴtrunc = EnzymeTestUtils.rand_tangent(USVᴴtrunc)
        remove_svd_gauge_dependence!(ΔUSVᴴtrunc[1], ΔUSVᴴtrunc[3], USVᴴtrunc...)
        EnzymeTestUtils.test_reverse(svd_trunc_no_error, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔUSVᴴtrunc, atol, rtol)
    end

    @testset "LQ" begin
        EnzymeTestUtils.test_reverse(lq_compact, Duplicated, (t, Duplicated); atol, rtol)

        if !is_ci
            # lq_full/lq_null requires being careful with gauges
            LQ = lq_full(t)
            ΔLQ = EnzymeTestUtils.rand_tangent(LQ)
            remove_lq_gauge_dependence!(ΔLQ..., t, LQ...)
            EnzymeTestUtils.test_reverse(lq_full, Duplicated, (t, Duplicated); output_tangent = ΔLQ, atol, rtol)

            Nᴴ = lq_null(t)
            Q = lq_compact(t)[2]
            ΔNᴴ = EnzymeTestUtils.rand_tangent(Nᴴ)
            remove_lq_null_gauge_dependence!(ΔNᴴ, Q, Nᴴ)
            EnzymeTestUtils.test_reverse(lq_null, Duplicated, (t, Duplicated); output_tangent = ΔNᴴ, atol, rtol)
        end
    end

    @testset "QR" begin
        EnzymeTestUtils.test_reverse(qr_compact, Duplicated, (t, Duplicated); atol, rtol)

        if !is_ci
            # qr_full/qr_null requires being careful with gauges
            QR = qr_full(t)
            ΔQR = EnzymeTestUtils.rand_tangent(QR)
            remove_qr_gauge_dependence!(ΔQR..., t, QR...)
            EnzymeTestUtils.test_reverse(qr_full, Duplicated, (t, Duplicated); output_tangent = ΔQR, atol, rtol)

            N = qr_null(t)
            Q = qr_compact(t)[1]
            ΔN = EnzymeTestUtils.rand_tangent(N)
            remove_qr_null_gauge_dependence!(ΔN, t, N)
            EnzymeTestUtils.test_reverse(qr_null, Duplicated, (t, Duplicated); atol, rtol, output_tangent = ΔN)
        end
    end
end

@timedtestset "Enzyme - Factorizations (EIGH/EIG): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
    atol = default_tol(T)
    rtol = default_tol(T)

    @testset "EIG" begin
        if !is_ci
            DV = eig_full(t)
            ΔDV = EnzymeTestUtils.rand_tangent(DV)
            remove_eig_gauge_dependence!(ΔDV[2], DV...)
            EnzymeTestUtils.test_reverse(eig_full, Duplicated, (t, Duplicated); output_tangent = ΔDV, atol, rtol)

            D = eig_vals(t)
            EnzymeTestUtils.test_reverse(eig_vals, Duplicated, (t, Duplicated); atol, rtol)
        end

        V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
        trunc = truncspace(V_trunc)
        alg = MatrixAlgebraKit.select_algorithm(eig_trunc_no_error, t, nothing; trunc)
        DVtrunc = eig_trunc_no_error(t, alg)
        ΔDVtrunc = EnzymeTestUtils.rand_tangent(DVtrunc)
        remove_eig_gauge_dependence!(ΔDVtrunc[2], DVtrunc...)
        EnzymeTestUtils.test_reverse(eig_trunc_no_error, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔDVtrunc, atol, rtol)
    end

    @testset "EIGH" begin
        th = project_hermitian(t)
        if !is_ci
            DV = eigh_full(th)
            ΔDV = EnzymeTestUtils.rand_tangent(DV)
            remove_eigh_gauge_dependence!(ΔDV[2], DV...)
            proj_eigh_full(t) = eigh_full(project_hermitian(t))
            EnzymeTestUtils.test_reverse(proj_eigh_full, Duplicated, (th, Duplicated); output_tangent = ΔDV, atol, rtol)

            D = eigh_vals(th)
            EnzymeTestUtils.test_reverse(eigh_vals ∘ project_hermitian, Duplicated, (th, Duplicated); atol, rtol)
        end

        V_trunc = spacetype(th)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
        trunc = truncspace(V_trunc)
        alg = MatrixAlgebraKit.select_algorithm(eigh_trunc_no_error, th, nothing; trunc)
        DVtrunc = eigh_trunc_no_error(th, alg)
        ΔDVtrunc = EnzymeTestUtils.rand_tangent(DVtrunc)
        remove_eigh_gauge_dependence!(ΔDVtrunc[2], DVtrunc...)
        proj_eigh(t, alg) = eigh_trunc_no_error(project_hermitian(t), alg)
        EnzymeTestUtils.test_reverse(proj_eigh, Duplicated, (th, Duplicated), (alg, Const); output_tangent = ΔDVtrunc, atol, rtol)
    end

    @testset "Projections" begin
        EnzymeTestUtils.test_reverse(project_hermitian, Duplicated, (t, Duplicated); atol, rtol)
        EnzymeTestUtils.test_reverse(project_antihermitian, Duplicated, (t, Duplicated); atol, rtol)
        EnzymeTestUtils.test_reverse(project_hermitian!, Duplicated, (t, Duplicated); atol, rtol)
        EnzymeTestUtils.test_reverse(project_antihermitian!, Duplicated, (t, Duplicated); atol, rtol)
    end
end
