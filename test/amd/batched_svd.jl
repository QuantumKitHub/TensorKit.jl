using Adapt, AMDGPU
using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: MatrixAlgebraKit, DivideAndConquer, DivideAndConquerBatched,
    QRIterationBatched, JacobiBatched, QRIteration, svd_compact, svd_compact!, svd_vals,
    svd_vals!, svd_full, svd_full!

const Factorizations = TensorKit.Factorizations

# SU(2) space with enough sectors to exceed the batching threshold
Vsu2 = Vect[SU2Irrep](0 => 2, 1 // 2 => 2, 1 => 2, 3 // 2 => 1)

# spectra are returned either
# as `Diagonal` blocks (`svd_compact`) or
# as plain vectors (`svd_vals`)
_spec(b) = Array(b isa AbstractMatrix ? TensorKit.diagview(b) : b)

function specdiff(S1, S2)
    d = 0.0
    for (c, b) in TensorKit.blocks(S1)
        d = max(d, maximum(abs, _spec(b) .- _spec(TensorKit.block(S2, c))))
    end
    return d
end

_id(t, V) = TensorKit.id(TensorKit.storagetype(t), V)

@timedtestset "batched SVD on ROCArray" verbose = true begin
    @testset "block-count dispatch" begin
        @test Factorizations.BATCHED_SVD_THRESHOLD == 4
        @test Factorizations.unbatched(DivideAndConquerBatched()) isa DivideAndConquer
    end

    @testset "many blocks: $T" for T in (Float64, ComplexF64)
        t_cpu = randn(T, Vsu2 ⊗ Vsu2 ← Vsu2)
        nblocks = length(TensorKit.blocksectors(t_cpu))
        @test nblocks >= Factorizations.BATCHED_SVD_THRESHOLD
        t = adapt(ROCArray, t_cpu)

        U, S, Vᴴ = svd_compact(t; alg = DivideAndConquerBatched())
        Ur, Sr, Vr = svd_compact(t; alg = DivideAndConquer())

        # singular values agree with the unbatched path
        @test specdiff(S, Sr) < 1.0e-10
        @test norm(U * S * Vᴴ - t) / norm(t) < 1.0e-10
        @test norm(U' * U - _id(U, domain(U))) < 1.0e-10
        @test norm(Vᴴ * Vᴴ' - _id(Vᴴ, codomain(Vᴴ))) < 1.0e-10

        Sv = svd_vals(t; alg = DivideAndConquerBatched())
        @test specdiff(Sv, Sr) < 1.0e-10
    end

    @testset "few blocks fall back: $T" for T in (Float64, ComplexF64)
        # trivial sector -> a single block, below the threshold
        V = ComplexSpace(6)
        t = adapt(ROCArray, randn(T, V ⊗ V ← V))
        @test length(TensorKit.blocksectors(t)) < Factorizations.BATCHED_SVD_THRESHOLD
        U, S, Vᴴ = svd_compact(t; alg = DivideAndConquerBatched())
        Ur, Sr, Vr = svd_compact(t; alg = DivideAndConquer())
        @test specdiff(S, Sr) < 1.0e-10
        @test norm(U * S * Vᴴ - t) / norm(t) < 1.0e-10
    end

    @testset "other batched algorithms" for alg in (QRIterationBatched(), JacobiBatched())
        t_cpu = randn(Float64, Vsu2 ⊗ Vsu2 ← Vsu2)
        t = adapt(ROCArray, t_cpu)
        U, S, Vᴴ = svd_compact(t; alg)
        _, Sr, _ = svd_compact(t; alg = DivideAndConquer())
        @test specdiff(S, Sr) < 1.0e-8
        @test norm(U * S * Vᴴ - t) / norm(t) < 1.0e-8
    end

    # `svd_full!` returns U (m, m) and Vᴴ (n, n) rather than the compact (m, minmn) / (minmn, n)
    @testset "svd_full: uniform blocks batch: $T" for T in (Float64, ComplexF64)
        Vu = Vect[Z4Irrep](0 => 3, 1 => 3, 2 => 3, 3 => 3)
        t_cpu = randn(T, Vu ← Vu)
        t = adapt(ROCArray, t_cpu)
        szs = [size(TensorKit.block(t, c)) for c in TensorKit.blocksectors(t)]
        @test length(szs) >= Factorizations.BATCHED_SVD_THRESHOLD
        @test all(isequal(first(szs)), szs)
        cs, _ = Factorizations._batchable(t, QRIterationBatched(), true)
        @test !isempty(cs)

        U, S, Vᴴ = svd_full(t; alg = QRIterationBatched())
        Ur, Sr, Vr = svd_full(t; alg = QRIteration())
        @test specdiff(S, Sr) < 1.0e-10
        @test norm(U * S * Vᴴ - t) / norm(t) < 1.0e-10
        @test norm(U' * U - _id(U, domain(U))) < 1.0e-10
        @test norm(U * U' - _id(U, codomain(U))) < 1.0e-10
        @test norm(Vᴴ' * Vᴴ - _id(Vᴴ, domain(Vᴴ))) < 1.0e-10
    end

    @testset "svd_full: ragged blocks fall back: $T" for T in (Float64,)
        t = adapt(ROCArray, randn(T, Vsu2 ⊗ Vsu2 ← Vsu2))
        szs = [size(TensorKit.block(t, c)) for c in TensorKit.blocksectors(t)]
        @test !all(isequal(first(szs)), szs)
        cs, _ = Factorizations._batchable(t, QRIterationBatched(), true)
        @test isempty(cs)
        U, S, Vᴴ = svd_full(t; alg = QRIterationBatched())
        Ur, Sr, Vr = svd_full(t; alg = QRIteration())
        @test specdiff(S, Sr) < 1.0e-10
        @test norm(U * S * Vᴴ - t) / norm(t) < 1.0e-10
    end

    # tall uniform blocks: hits the full mode where U (m, m) and Vᴴ (n, n) have different sizes
    @testset "svd_full: tall uniform blocks: $T" for T in (Float64,)
        Vbig = Vect[Z4Irrep](0 => 4, 1 => 4, 2 => 4, 3 => 4)
        Vsml = Vect[Z4Irrep](0 => 2, 1 => 2, 2 => 2, 3 => 2)
        t = adapt(ROCArray, randn(T, Vbig ← Vsml))
        szs = [size(TensorKit.block(t, c)) for c in TensorKit.blocksectors(t)]
        @test all(isequal((4, 2)), szs)
        cs, _ = Factorizations._batchable(t, QRIterationBatched(), true)
        @test !isempty(cs)
        U, S, Vᴴ = svd_full(t; alg = QRIterationBatched())
        Ur, Sr, Vr = svd_full(t; alg = QRIteration())
        @test specdiff(S, Sr) < 1.0e-10
        @test norm(U * S * Vᴴ - t) / norm(t) < 1.0e-10
        @test norm(U' * U - _id(U, domain(U))) < 1.0e-10
        @test norm(Vᴴ * Vᴴ' - _id(Vᴴ, codomain(Vᴴ))) < 1.0e-10
    end
end
