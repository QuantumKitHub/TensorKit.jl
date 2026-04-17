using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using MatrixAlgebraKit
using Enzyme, EnzymeTestUtils
using Random

@isdefined(TestSetup) || include("../../setup.jl")
using .TestSetup

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2)',
        Vect[FermionParity](0 => 2, 1 => 1)',
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
    (
        Vect[SU2Irrep](0 => 2, 1 // 2 => 1),
        Vect[SU2Irrep](0 => 1, 1 => 1),
        Vect[SU2Irrep](1 // 2 => 1, 1 => 1)',
        Vect[SU2Irrep](1 // 2 => 2),
        Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
    ),
    (
        Vect[FibonacciAnyon](:I => 2, :τ => 1),
        Vect[FibonacciAnyon](:I => 1, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 3),
        Vect[FibonacciAnyon](:I => 2, :τ => 2),
    ),
)
eltypes = (Float64, ComplexF64)

function remove_lqgauge_dependence!(ΔQ, t, Q)
    for (c, b) in blocks(ΔQ)
        m, n = size(block(t, c))
        minmn = min(m, n)
        Qc = block(Q, c)
        Q1 = view(Qc, 1:minmn, 1:n)
        ΔQ2 = view(b, (minmn + 1):n, :)
        mul!(ΔQ2, ΔQ2 * Q1', Q1)
    end
    return ΔQ
end

function remove_lq_null_gauge_dependence!(ΔNᴴ, Q)
    for (c, b) in blocks(ΔNᴴ)
        Qc = block(Q, c)
        ΔNᴴQᴴ = b * Qc'
        mul!(b, ΔNᴴQᴴ, Qc)
    end
    return ΔNᴴ
end

@timedtestset "Enzyme - Factorizations (LQ): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, A in (randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]), randn(T, V[1] ⊗ V[2] ← V[1]))
    atol = default_tol(T)
    rtol = default_tol(T)
    EnzymeTestUtils.test_reverse(lq_compact, Duplicated, (A, Duplicated); atol, rtol)

    # lq_full/lq_null requires being careful with gauges
    LQ = lq_full(A)
    ΔLQ = EnzymeTestUtils.rand_tangent(LQ)
    remove_lqgauge_dependence!(ΔLQ[2], A, LQ[2])
    EnzymeTestUtils.test_reverse(lq_full, Duplicated, (A, Duplicated); output_tangent = ΔLQ, atol, rtol)

    Nᴴ = lq_null(A)
    Q = lq_compact(A)[2]
    ΔNᴴ = EnzymeTestUtils.rand_tangent(Nᴴ)
    remove_lq_null_gauge_dependence!(ΔNᴴ, Q)
    EnzymeTestUtils.test_reverse(lq_null, Duplicated, (A, Duplicated); output_tangent = ΔNᴴ, atol, rtol)
end
