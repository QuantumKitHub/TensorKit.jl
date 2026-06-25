using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_qr_gauge_dependence!, remove_qr_null_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (QR): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, A in (randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]), randn(T, V[1] ⊗ V[2] ⊗ V[3] ← (V[4] ⊗ V[5])'))
    atol = default_tol(T)
    rtol = default_tol(T)

    EnzymeTestUtils.test_reverse(qr_compact, Duplicated, (A, Duplicated); atol, rtol)

    # qr_full/qr_null requires being careful with gauges
    QR = qr_full(A)
    ΔQR = EnzymeTestUtils.rand_tangent(QR)
    remove_qr_gauge_dependence!(ΔQR..., A, QR...)
    EnzymeTestUtils.test_reverse(qr_full, Duplicated, (A, Duplicated); output_tangent = ΔQR, atol, rtol)

    N = qr_null(A)
    Q = qr_compact(A)[1]
    ΔN = EnzymeTestUtils.rand_tangent(N)
    remove_qr_null_gauge_dependence!(ΔN, A, N)
    EnzymeTestUtils.test_reverse(qr_null, Duplicated, (A, Duplicated); atol, rtol, output_tangent = ΔN)
end
