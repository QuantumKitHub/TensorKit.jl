using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: DefaultAlgorithm, diagview


spacelist = factorization_spacelist(fast_tests)

eltypes = (Float32, ComplexF64)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("QR and LQ decompositions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "QR and LQ decompositions with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2 ⊗ V3
        Vd = fuse(V1 ⊗ V2)

        @testset "QR decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)')',
                        rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5)), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
                    )

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)

                Q, R = @constinferred qr_full(t, DefaultAlgorithm())
                @test Q * R ≈ t
                @test isunitary(Q)

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred qr_compact(t, DefaultAlgorithm())
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred left_orth(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred left_orth(t, DefaultAlgorithm())
                @test Q * R ≈ t
                @test isisometric(Q)

                # Compute only R by passing an empty basis output.
                V0 = zerospace(spacetype(t))
                Q0 = t isa DiagonalTensorMap ? DiagonalTensorMap(T[], V0) : similar(t, V0 ← V0)
                for positive in (false, true), alg in ((; positive), DefaultAlgorithm(; positive))
                    Q, R = @constinferred left_orth(t; positive)
                    R0 = similar(R)
                    Q2, R2 = @constinferred left_orth!(copy(t), (Q0, R0); alg)
                    @test dim(Q2) == 0
                    @test space(R2) == space(R)
                    @test R2 ≈ R
                    @test Q * R2 ≈ t
                end

                N = @constinferred qr_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred qr_null(t, DefaultAlgorithm())
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t, DefaultAlgorithm())
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes
                t = rand(T, V1 ⊗ V2, zerospace(V1))

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)
                @test dim(R) == dim(t) == 0

                Q, R = @constinferred qr_full(t, DefaultAlgorithm())
                @test Q * R ≈ t
                @test isunitary(Q)
                @test dim(R) == dim(t) == 0

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred qr_compact(t, DefaultAlgorithm())
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred left_orth(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred left_orth(t, DefaultAlgorithm())
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                N = @constinferred qr_null(t)
                @test isunitary(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred qr_null(t, DefaultAlgorithm())
                @test isunitary(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end
        end

        @testset "LQ decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)')',
                        rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5)), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
                    )

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)

                L, Q = @constinferred lq_full(t, DefaultAlgorithm())
                @test L * Q ≈ t
                @test isunitary(Q)

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred lq_compact(t, DefaultAlgorithm())
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred right_orth(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred right_orth(t, DefaultAlgorithm())
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                # Compute only L by passing an empty basis output.
                V0 = zerospace(spacetype(t))
                Q0 = t isa DiagonalTensorMap ? DiagonalTensorMap(T[], V0) : similar(t, V0 ← V0)
                for positive in (false, true), alg in ((; positive), DefaultAlgorithm(; positive))
                    L, Q = @constinferred right_orth(t; positive)
                    L0 = similar(L)
                    L2, Q2 = @constinferred right_orth!(copy(t), (L0, Q0); alg)
                    @test dim(Q2) == 0
                    @test space(L2) == space(L)
                    @test L2 ≈ L
                    @test L2 * Q ≈ t
                end

                Nᴴ = @constinferred lq_null(t)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred lq_null(t, DefaultAlgorithm())
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            for T in eltypes
                # empty tensor
                t = rand(T, zerospace(V1), V1 ⊗ V2)

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)
                @test dim(L) == dim(t) == 0

                L, Q = @constinferred lq_full(t, DefaultAlgorithm())
                @test L * Q ≈ t
                @test isunitary(Q)
                @test dim(L) == dim(t) == 0

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred lq_compact(t, DefaultAlgorithm())
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred right_orth(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred right_orth(t, DefaultAlgorithm())
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                Nᴴ = @constinferred lq_null(t)
                @test isunitary(Nᴴ)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred lq_null(t, DefaultAlgorithm())
                @test isunitary(Nᴴ)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end
        end
    end
end
