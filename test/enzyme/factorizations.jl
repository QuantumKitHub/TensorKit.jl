using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using MatrixAlgebraKit
using Enzyme, EnzymeTestUtils
using Random
Enzyme.Compiler.VERBOSE_ERRORS[] = true
@isdefined(TestSetup) || include("../setup.jl")
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
    ), # TODO problem in svdgauge_dep
    (
        Vect[FibonacciAnyon](:I => 2, :τ => 1),
        Vect[FibonacciAnyon](:I => 1, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 3),
        Vect[FibonacciAnyon](:I => 2, :τ => 2),
    ),
)
eltypes = (Float64, ComplexF64)

function remove_qrgauge_dependence!(ΔQ, t, Q)
    for (c, b) in blocks(ΔQ)
        m, n = size(block(t, c))
        minmn = min(m, n)
        Qc = block(Q, c)
        Q1 = view(Qc, 1:m, 1:minmn)
        ΔQ2 = view(b, :, (minmn + 1):m)
        mul!(ΔQ2, Q1, Q1' * ΔQ2)
    end
    return ΔQ
end
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
function remove_eiggauge_dependence!(
        ΔV, D, V; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(D)
    )
    gaugepart = V' * ΔV
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_dependence!(
        ΔV, D, V; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(D)
    )
    gaugepart = project_antihermitian!(V' * ΔV)
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end
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

@timedtestset "Enzyme - Factorizations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)

    @timedtestset "QR" begin
        A = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])

        EnzymeTestUtils.test_reverse(qr_compact, Duplicated, (A, Duplicated); atol, rtol)

        # qr_full/qr_null requires being careful with gauges
        QR = qr_full(A)
        ΔQR = randn!.(similar.(QR))
        remove_qrgauge_dependence!(ΔQR[1], A, QR[1])
        EnzymeTestUtils.test_reverse(qr_full, Duplicated, (A, Duplicated); output_tangent = ΔQR, atol, rtol)
        #EnzymeTestUtils.test_reverse(qr_null, Duplicated, (A, Duplicated); atol, rtol)

        A = randn(T, V[1] ⊗ V[2] ← V[1])

        EnzymeTestUtils.test_reverse(qr_compact, Duplicated, (A, Duplicated); atol, rtol)

        # qr_full/qr_null requires being careful with gauges
        QR = qr_full(A)
        ΔQR = randn!.(similar.(QR))
        remove_qrgauge_dependence!(ΔQR[1], A, QR[1])
        EnzymeTestUtils.test_reverse(qr_full, Duplicated, (A, Duplicated); output_tangent = ΔQR, atol, rtol)
        #EnzymeTestUtils.test_reverse(qr_null, Duplicated, (A, Duplicated); atol, rtol)
    end

    @timedtestset "LQ" begin
        A = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])

        EnzymeTestUtils.test_reverse(lq_compact, Duplicated, (A, Duplicated); atol, rtol)

        # lq_full/lq_null requires being careful with gauges
        LQ = lq_full(A)
        ΔLQ = randn!.(similar.(LQ))
        remove_lqgauge_dependence!(ΔLQ[2], A, LQ[2])
        EnzymeTestUtils.test_reverse(lq_full, Duplicated, (A, Duplicated); output_tangent = ΔLQ, atol, rtol)
        #EnzymeTestUtils.test_reverse(lq_null, Duplicated, (A, Duplicated); atol, rtol)

        A = randn(T, V[1] ⊗ V[2] ← V[1])

        EnzymeTestUtils.test_reverse(lq_compact, Duplicated, (A, Duplicated); atol, rtol)

        # lq_full/lq_null requires being careful with gauges
        LQ = lq_full(A)
        ΔLQ = randn!.(similar.(LQ))
        remove_lqgauge_dependence!(ΔLQ[2], A, LQ[2])
        EnzymeTestUtils.test_reverse(lq_full, Duplicated, (A, Duplicated); output_tangent = ΔLQ, atol, rtol)
        #EnzymeTestUtils.test_reverse(lq_null, Duplicated, (A, Duplicated); atol, rtol)
    end

    @timedtestset "Eigenvalue decomposition" begin
        for t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
            DV = eig_full(t)
            ΔDV = (DiagonalTensorMap(randn!(similar(DV[1].data)), space(DV[1])), randn!(similar(DV[2])))
            remove_eiggauge_dependence!(ΔDV[2], DV...)
            EnzymeTestUtils.test_reverse(eig_full, Duplicated, (t, Duplicated); output_tangent = ΔDV, atol, rtol)

            th = project_hermitian(t)
            DV = eigh_full(th)
            ΔDV = (DiagonalTensorMap(randn!(similar(DV[1].data)), space(DV[1])), randn!(similar(DV[2])))
            remove_eighgauge_dependence!(ΔDV[2], DV...)
            EnzymeTestUtils.test_reverse(eigh_full ∘ project_hermitian, Duplicated, (t, Duplicated); output_tangent = ΔDV, atol, rtol)
        end
    end

    @timedtestset "Singular value decomposition" begin
        for t in (randn(T, V[1] ← V[1]), randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4]))
            USVᴴ = svd_compact(t)
            ΔUSVᴴ = (TensorMap(randn!(similar(USVᴴ[1].data)), space(USVᴴ[1])), DiagonalTensorMap(randn!(similar(USVᴴ[2].data)), space(USVᴴ[2], 1)), TensorMap(randn!(similar(USVᴴ[3].data)), space(USVᴴ[3])))
            remove_svdgauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
            EnzymeTestUtils.test_reverse(svd_compact, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)

            USVᴴ = svd_full(t)
            ΔUSVᴴ = (TensorMap(randn!(similar(USVᴴ[1].data)), space(USVᴴ[1])), TensorMap(randn!(similar(USVᴴ[2].data)), space(USVᴴ[2])), TensorMap(randn!(similar(USVᴴ[3].data)), space(USVᴴ[3])))
            remove_svdgauge_dependence!(ΔUSVᴴ[1], ΔUSVᴴ[3], USVᴴ...)
            EnzymeTestUtils.test_reverse(svd_full, Duplicated, (t, Duplicated); output_tangent = ΔUSVᴴ, atol, rtol)

            #=V_trunc = spacetype(t)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
            trunc = truncspace(V_trunc)
            alg = MatrixAlgebraKit.select_algorithm(svd_trunc, t, nothing; trunc)
            USVᴴtrunc = svd_trunc(t, alg)
            ΔUSVᴴtrunc = (Enzyme.randn_tangent(Base.front(USVᴴtrunc))..., zero(last(USVᴴtrunc)))
            remove_svdgauge_dependence!(ΔUSVᴴtrunc[1], ΔUSVᴴtrunc[3], Base.front(USVᴴtrunc)...)
            EnzymeTestUtils.test_reverse(svd_trunc, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔUSVᴴtrunc, atol, rtol)=#
            # TODO
        end
    end
end
