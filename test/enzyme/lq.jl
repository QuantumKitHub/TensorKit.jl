using Test, TestExtras
using TensorKit
using TensorOperations
using MatrixAlgebraKit
using MatrixAlgebraKit: remove_lq_gauge_dependence!, remove_lq_null_gauge_dependence!
using Enzyme, EnzymeTestUtils
using Random

is_ci = get(ENV, "CI", "false") == "true"

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - Factorizations (LQ): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]), randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])'))
    atol = default_tol(T)
    rtol = default_tol(T)

    EnzymeTestUtils.test_reverse(lq_compact, Duplicated, (t, Duplicated); atol, rtol)
    EnzymeTestUtils.test_forward(lq_compact, Duplicated, (t, Duplicated); atol, rtol)

    if !is_ci
        # lq_full/lq_null requires being careful with gauges
        LQ = lq_full(t)
        ΔLQ = EnzymeTestUtils.rand_tangent(LQ)
        remove_lq_gauge_dependence!(ΔLQ..., t, LQ...)
        EnzymeTestUtils.test_reverse(lq_full, Duplicated, (t, Duplicated); output_tangent = ΔLQ, atol, rtol)
        EnzymeTestUtils.test_forward(lq_full, Duplicated, (t, Duplicated); atol, rtol)

        Nᴴ = lq_null(t)
        Q = lq_compact(t)[2]
        ΔNᴴ = EnzymeTestUtils.rand_tangent(Nᴴ)
        remove_lq_null_gauge_dependence!(ΔNᴴ, Q, Nᴴ)
        EnzymeTestUtils.test_reverse(lq_null, Duplicated, (t, Duplicated); output_tangent = ΔNᴴ, atol, rtol)
        EnzymeTestUtils.test_forward(lq_null, Duplicated, (t, Duplicated); atol, rtol)
    end
end
