using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: diagview


spacelist = if fast_tests
    (Vtr, Vℤ₃, VSU₂)
elseif get(ENV, "CI", "false") == "true"
    println("Detected running on CI")
    if Sys.iswindows()
        (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VIB_diag)
    elseif Sys.isapple()
        (Vtr, Vℤ₃, VfU₁, VfSU₂, VIB_M)
    else
        (Vtr, VU₁, VCU₁, VSU₂, VfSU₂, VIB_diag, VIB_M)
    end
else
    (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VIB_diag, VIB_M)
end

eltypes = (Float32, ComplexF64)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("Factorizations with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Factorizations with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2
        @assert !isempty(blocksectors(W))
        @assert !isempty(intersect(blocksectors(V4), blocksectors(W)))

        @testset "QR decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V4), rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred left_orth(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                N = @constinferred qr_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t)
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

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred left_orth(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                N = @constinferred qr_null(t)
                @test isunitary(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end
        end

        @testset "LQ decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V4), rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred right_orth(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                Nᴴ = @constinferred lq_null(t)
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

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred right_orth(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                Nᴴ = @constinferred lq_null(t)
                @test isunitary(Nᴴ)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end
        end
    end
end
