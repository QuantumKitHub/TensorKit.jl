using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_qr_gauge_dependence!, remove_qr_null_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

is_ci = get(ENV, "CI", "false") == "true"

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (QR): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]), randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])'))
    atol = default_tol(T)
    rtol = default_tol(T)

    EnzymeTestUtils.test_reverse(qr_compact, Duplicated, (t, Duplicated); atol, rtol)
    EnzymeTestUtils.test_forward(qr_compact, Duplicated, (t, Duplicated); atol, rtol)

    if !is_ci
        # qr_full/qr_null requires being careful with gauges
        QR = qr_full(t)
        ΔQR = EnzymeTestUtils.rand_tangent(QR)
        remove_qr_gauge_dependence!(ΔQR..., t, QR...)
        EnzymeTestUtils.test_reverse(qr_full, Duplicated, (t, Duplicated); output_tangent = ΔQR, atol, rtol)
        EnzymeTestUtils.test_forward(qr_full, Duplicated, (t, Duplicated); atol, rtol)

        N = qr_null(t)
        Q = qr_compact(t)[1]
        ΔN = EnzymeTestUtils.rand_tangent(N)
        remove_qr_null_gauge_dependence!(ΔN, t, N)
        EnzymeTestUtils.test_reverse(qr_null, Duplicated, (t, Duplicated); atol, rtol, output_tangent = ΔN)
        EnzymeTestUtils.test_forward(qr_null, Duplicated, (t, Duplicated); atol, rtol)
    end
end
