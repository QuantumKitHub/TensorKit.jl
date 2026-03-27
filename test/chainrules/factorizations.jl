using Test, TestExtras
using TensorKit
using TensorKit: type_repr, SectorDict
using TensorOperations
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences: FiniteDifferences, central_fdm, forward_fdm
using Random
using LinearAlgebra
using Zygote
using MatrixAlgebraKit
using MatrixAlgebraKit: LAPACK_HouseholderQR, LAPACK_HouseholderLQ, diagview

# Tests
# -----

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
    (
        Vect[Z2Irrep](0 => 1, 1 => 1),
        Vect[Z2Irrep](0 => 1, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 3),
        Vect[Z2Irrep](0 => 2, 1 => 2),
    ),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2)',
        Vect[FermionParity](0 => 2, 1 => 1)',
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
    (
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
        Vect[U1Irrep](0 => 1, 1 => 1, -1 => 2),
        Vect[U1Irrep](0 => 1, 1 => 2, -1 => 1)',
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

for V in spacelist
    I = sectortype(eltype(V))
    Istr = type_repr(I)
    eltypes = isreal(sectortype(eltype(V))) ? (Float64, ComplexF64) : (ComplexF64,)
    println("---------------------------------------")
    println("Auto-diff with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "AD with symmetry $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2

        @timedtestset "Factorizations" begin
            @testset "QR" begin
                for T in eltypes,
                        t in (
                            randn(T, W, W), randn(T, W, W)',
                            randn(T, W, V[1]), randn(T, V[1], W),
                            randn(T, W, V[1])', randn(T, V[1], W)',
                            DiagonalTensorMap(randn(T, reduceddim(V[1])), V[1]),
                        )

                    atol = rtol = default_tol(T) * dim(space(t))
                    fkwargs = (; positive = true) # make FiniteDifferences happy

                    test_ad_rrule(qr_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(first ∘ qr_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(last ∘ qr_compact, t; fkwargs, atol, rtol)

                    # qr_full/qr_null requires being careful with gauges
                    Q, R = qr_full(t)
                    ΔQ = rand_tangent(Q)
                    ΔR = rand_tangent(R)

                    if fuse(domain(t)) ≺ fuse(codomain(t))
                        _, full_pb = Zygote.pullback(qr_full, t)
                        @test_logs (:warn, r"^`qr") match_mode = :any full_pb((ΔQ, ΔR))
                    end

                    remove_qrgauge_dependence!(ΔQ, t, Q)

                    test_ad_rrule(qr_full, t; fkwargs, atol, rtol, output_tangent = (ΔQ, ΔR))
                    test_ad_rrule(
                        first ∘ qr_full, t;
                        fkwargs, atol, rtol, output_tangent = ΔQ
                    )
                    test_ad_rrule(last ∘ qr_full, t; fkwargs, atol, rtol, output_tangent = ΔR)
                end
            end

            @testset "LQ" begin
                for T in eltypes,
                        t in (
                            randn(T, W, W), randn(T, W, W)',
                            randn(T, W, V[1]), randn(T, V[1], W),
                            randn(T, W, V[1])', randn(T, V[1], W)',
                            DiagonalTensorMap(randn(T, reduceddim(V[1])), V[1]),
                        )

                    atol = rtol = default_tol(T) * dim(space(t))
                    fkwargs = (; positive = true) # make FiniteDifferences happy

                    test_ad_rrule(lq_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(first ∘ lq_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(last ∘ lq_compact, t; fkwargs, atol, rtol)

                    # lq_full/lq_null requires being careful with gauges
                    L, Q = lq_full(t)
                    ΔQ = rand_tangent(Q)
                    ΔL = rand_tangent(L)

                    if fuse(codomain(t)) ≺ fuse(domain(t))
                        _, full_pb = Zygote.pullback(lq_full, t)
                        # broken due to typo in MAK
                        # @test_logs (:warn, r"^`lq") match_mode = :any full_pb((ΔL, ΔQ))
                    end

                    remove_lqgauge_dependence!(ΔQ, t, Q)

                    test_ad_rrule(lq_full, t; fkwargs, atol, rtol, output_tangent = (ΔL, ΔQ))
                    test_ad_rrule(
                        first ∘ lq_full, t;
                        fkwargs, atol, rtol, output_tangent = ΔL
                    )
                    test_ad_rrule(last ∘ lq_full, t; fkwargs, atol, rtol, output_tangent = ΔQ)
                end
            end

            @testset "Eigenvalue decomposition" begin
                for T in eltypes,
                        t in (
                            rand(T, V[1], V[1]), rand(T, W, W), rand(T, W, W)',
                            # DiagonalTensorMap(rand(T, reduceddim(V[1])), V[1]), # broken in MatrixAlgebraKit
                        )

                    atol = rtol = default_tol(T) * dim(space(t))

                    d, v = eig_full(t)
                    Δv = rand_tangent(v)
                    Δd = rand_tangent(d)
                    Δd2 = randn!(similar(d, space(d)))
                    remove_eiggauge_dependence!(Δv, d, v)

                    test_ad_rrule(eig_full, t; output_tangent = (Δd, Δv), atol, rtol)
                    test_ad_rrule(first ∘ eig_full, t; output_tangent = Δd, atol, rtol)
                    test_ad_rrule(last ∘ eig_full, t; output_tangent = Δv, atol, rtol)
                    test_ad_rrule(eig_full, t; output_tangent = (Δd2, Δv), atol, rtol)

                    t += t'
                    d, v = eigh_full(t)
                    Δv = rand_tangent(v)
                    Δd = rand_tangent(d)
                    Δd2 = randn!(similar(d, space(d)))
                    remove_eighgauge_dependence!(Δv, d, v)

                    # necessary for FiniteDifferences to not complain
                    eigh_full′ = eigh_full ∘ _project_hermitian

                    test_ad_rrule(eigh_full′, t; output_tangent = (Δd, Δv), atol, rtol)
                    test_ad_rrule(first ∘ eigh_full′, t; output_tangent = Δd, atol, rtol)
                    test_ad_rrule(last ∘ eigh_full′, t; output_tangent = Δv, atol, rtol)
                    test_ad_rrule(eigh_full′, t; output_tangent = (Δd2, Δv), atol, rtol)
                end
            end

            @testset "Singular value decomposition" begin
                for T in eltypes,
                        t in (randn(T, V[1], V[1]), randn(T, W, W), randn(T, W, W))
                    # TODO: fix diagonaltensormap case
                    #   DiagonalTensorMap(rand(T, reduceddim(V1)), V1))

                    atol = rtol = degeneracy_atol = default_tol(T) * dim(space(t))
                    USVᴴ = svd_compact(t)
                    ΔU, ΔS, ΔVᴴ = rand_tangent.(USVᴴ)
                    ΔS2 = randn!(similar(ΔS, space(ΔS)))
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, USVᴴ...; degeneracy_atol)

                    test_ad_rrule(svd_full, t; output_tangent = (ΔU, ΔS, ΔVᴴ), atol, rtol)
                    test_ad_rrule(svd_full, t; output_tangent = (ΔU, ΔS2, ΔVᴴ), atol, rtol)
                    test_ad_rrule(svd_compact, t; output_tangent = (ΔU, ΔS, ΔVᴴ), atol, rtol)
                    test_ad_rrule(svd_compact, t; output_tangent = (ΔU, ΔS2, ΔVᴴ), atol, rtol)

                    # Testing truncation with finitedifferences is RNG-prone since the
                    # Jacobian changes size if the truncation space changes, causing errors.
                    # So, first test the fixed space case, then do more limited testing on
                    # some gradients and compare to the fixed space case
                    V_trunc = spacetype(t)(c => div(min(size(b)...), 2) for (c, b) in blocks(t))
                    trunc = truncspace(V_trunc)
                    USVᴴ_trunc = svd_trunc(t; trunc)
                    ΔUSVᴴ_trunc = (rand_tangent.(Base.front(USVᴴ_trunc))..., zero(last(USVᴴ_trunc)))
                    remove_svdgauge_dependence!(
                        ΔUSVᴴ_trunc[1], ΔUSVᴴ_trunc[3], Base.front(USVᴴ_trunc)...; degeneracy_atol
                    )
                    test_ad_rrule(
                        svd_trunc, t;
                        fkwargs = (; trunc), output_tangent = ΔUSVᴴ_trunc, atol, rtol
                    )

                    # attempt to construct a loss function that doesn't depend on the gauges
                    function f(t; trunc)
                        Utr, Str, Vᴴtr, ϵ = svd_trunc(t; trunc)
                        return LinearAlgebra.tr(Str) + LinearAlgebra.norm(Utr * Vᴴtr)
                    end

                    trunc = truncrank(ceil(Int, dim(V_trunc)))
                    USVᴴ_trunc′ = svd_trunc(t; trunc)
                    g1, = Zygote.gradient(x -> f(x; trunc), t)
                    g2, = Zygote.gradient(x -> f(x; trunc = truncspace(space(USVᴴ_trunc′[2], 1))), t)
                    @test g1 ≈ g2

                    trunc = truncerror(; atol = last(USVᴴ_trunc))
                    USVᴴ_trunc′ = svd_trunc(t; trunc)
                    g1, = Zygote.gradient(x -> f(x; trunc), t)
                    g2, = Zygote.gradient(x -> f(x; trunc = truncspace(space(USVᴴ_trunc′[2], 1))), t)
                    @test g1 ≈ g2

                    tol = minimum(((c, b),) -> minimum(diagview(b)), blocks(USVᴴ_trunc[2]); init = zero(scalartype(USVᴴ_trunc[2])))
                    trunc = trunctol(; atol = 10 * tol)
                    USVᴴ_trunc′ = svd_trunc(t; trunc)
                    g1, = Zygote.gradient(x -> f(x; trunc), t)
                    g2, = Zygote.gradient(x -> f(x; trunc = truncspace(space(USVᴴ_trunc′[2], 1))), t)
                    @test g1 ≈ g2
                end
            end
        end
    end
end

# https://github.com/quantumkithub/TensorKit.jl/issues/201
@testset "Issue #201" begin
    function f(A::AbstractTensorMap)
        U, S, V, = svd_compact(A)
        return tr(S)
    end
    function f(A::AbstractMatrix)
        S = LinearAlgebra.svdvals(A)
        return sum(S)
    end
    A₀ = randn(Z2Space(4, 4) ← Z2Space(4, 4))
    grad1, = Zygote.gradient(f, A₀)
    grad2, = Zygote.gradient(f, convert(Array, A₀))
    @test convert(Array, grad1) ≈ grad2

    function g(A::AbstractTensorMap)
        U, S, V, = svd_compact(A)
        return tr(U * V)
    end
    function g(A::AbstractMatrix)
        U, S, V, = LinearAlgebra.svd(A)
        return tr(U * V')
    end
    B₀ = randn(ComplexSpace(4) ← ComplexSpace(4))
    grad3, = Zygote.gradient(g, B₀)
    grad4, = Zygote.gradient(g, convert(Array, B₀))
    @test convert(Array, grad3) ≈ grad4
end
