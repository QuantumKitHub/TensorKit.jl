using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_lq_gauge_dependence!, remove_lq_null_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (LQ): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, A in (randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]), randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])'))
    atol = default_tol(T)
    rtol = default_tol(T)
    EnzymeTestUtils.test_reverse(lq_compact, Duplicated, (A, Duplicated); atol, rtol)

    # lq_full/lq_null requires being careful with gauges
    LQ = lq_full(A)
    ΔLQ = EnzymeTestUtils.rand_tangent(LQ)
    remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
    EnzymeTestUtils.test_reverse(lq_full, Duplicated, (A, Duplicated); output_tangent = ΔLQ, atol, rtol)

    Nᴴ = lq_null(A)
    Q = lq_compact(A)[2]
    ΔNᴴ = EnzymeTestUtils.rand_tangent(Nᴴ)
    remove_lq_null_gauge_dependence!(ΔNᴴ, Q, Nᴴ)
    EnzymeTestUtils.test_reverse(lq_null, Duplicated, (A, Duplicated); output_tangent = ΔNᴴ, atol, rtol)
end
