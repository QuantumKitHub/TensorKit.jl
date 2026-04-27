using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_qr_gauge_dependence!, remove_lq_gauge_dependence!,
    remove_eigh_gauge_dependence!, remove_eig_gauge_dependence!, remove_svd_gauge_dependence!
using Mooncake
using Random


mode = Mooncake.ReverseMode
rng = Random.default_rng()

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Mooncake - Factorizations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    @timedtestset "QR" begin
        A = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])

        Mooncake.TestUtils.test_rule(rng, qr_compact, A; atol, rtol, mode, is_primitive = false)

        # qr_full/qr_null requires being careful with gauges
        QR = qr_full(A)
        ΔQR = Mooncake.randn_tangent(rng, QR)
        remove_qr_gauge_dependence!(ΔQR..., A, QR...)
        Mooncake.TestUtils.test_rule(rng, qr_full, A; output_tangent = ΔQR, atol, rtol, mode, is_primitive = false)
        # TODO:
        # Mooncake.TestUtils.test_rule(rng, qr_null, A; atol, rtol, mode, is_primitive = false)

        A = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← (V[4] ⊗ V[5])')

        Mooncake.TestUtils.test_rule(rng, qr_compact, A; atol, rtol, mode, is_primitive = false)

        # qr_full/qr_null requires being careful with gauges
        QR = qr_full(A)
        ΔQR = Mooncake.randn_tangent(rng, QR)
        remove_qr_gauge_dependence!(ΔQR..., A, QR...)
        Mooncake.TestUtils.test_rule(rng, qr_full, A; output_tangent = ΔQR, atol, rtol, mode, is_primitive = false)
        # TODO:
        # Mooncake.TestUtils.test_rule(rng, qr_null, A; atol, rtol, mode, is_primitive = false)
    end

    @timedtestset "LQ" begin
        A = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])

        Mooncake.TestUtils.test_rule(rng, lq_compact, A; atol, rtol, mode, is_primitive = false)

        # qr_full/qr_null requires being careful with gauges
        LQ = lq_full(A)
        ΔLQ = Mooncake.randn_tangent(rng, LQ)
        remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
        Mooncake.TestUtils.test_rule(rng, lq_full, A; output_tangent = ΔLQ, atol, rtol, mode, is_primitive = false)
        # TODO:
        # Mooncake.TestUtils.test_rule(rng, lq_null, A; atol, rtol, mode, is_primitive = false)

        A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')

        Mooncake.TestUtils.test_rule(rng, lq_compact, A; atol, rtol, mode, is_primitive = false)

        # qr_full/qr_null requires being careful with gauges
        LQ = lq_full(A)
        ΔLQ = Mooncake.randn_tangent(rng, LQ)
        remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
        Mooncake.TestUtils.test_rule(rng, lq_full, A; output_tangent = ΔLQ, atol, rtol, mode, is_primitive = false)
        # TODO:
        # Mooncake.TestUtils.test_rule(rng, lq_null, A; atol, rtol, mode, is_primitive = false)
    end

    @timedtestset "Eigenvalue decomposition" begin
        for t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
            DV = eig_full(t)
            ΔDV = Mooncake.randn_tangent(rng, DV)
            remove_eig_gauge_dependence!(ΔDV[2], DV...)
            Mooncake.TestUtils.test_rule(rng, eig_full, t; output_tangent = ΔDV, atol, rtol, mode, is_primitive = false)

            th = project_hermitian(t)
            DV = eigh_full(th)
            ΔDV = Mooncake.randn_tangent(rng, DV)
            remove_eigh_gauge_dependence!(ΔDV[2], DV...)
            Mooncake.TestUtils.test_rule(rng, eigh_full ∘ project_hermitian, th; output_tangent = ΔDV, atol, rtol, mode, is_primitive = false)
        end
    end

    @timedtestset "Singular value decomposition" begin
        for t in (randn(T, V[1] ← V[1]), randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])'))
            USVᴴ = svd_compact(t)
            ΔUSVᴴ = Mooncake.randn_tangent(rng, USVᴴ)
            remove_svd_gauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
            Mooncake.TestUtils.test_rule(rng, svd_compact, t; output_tangent = ΔUSVᴴ, atol, rtol, mode, is_primitive = false)

            # USVᴴ = svd_full(t)
            # ΔUSVᴴ = Mooncake.randn_tangent(rng, USVᴴ)
            # remove_svdgauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
            # Mooncake.TestUtils.test_rule(rng, svd_full, t; output_tangent = ΔUSVᴴ, atol, rtol, mode, is_primitive = false)

            V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
            trunc = truncspace(V_trunc)
            alg = MatrixAlgebraKit.select_algorithm(svd_trunc, t, nothing; trunc)
            USVᴴtrunc = svd_trunc(t, alg)
            ΔUSVᴴtrunc = (Mooncake.randn_tangent(rng, Base.front(USVᴴtrunc))..., zero(last(USVᴴtrunc)))
            remove_svd_gauge_dependence!(ΔUSVᴴtrunc[1], ΔUSVᴴtrunc[3], Base.front(USVᴴtrunc)...)
            Mooncake.TestUtils.test_rule(rng, svd_trunc, t, alg; output_tangent = ΔUSVᴴtrunc, atol, rtol, mode)
        end
    end
end
