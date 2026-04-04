using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: diagview


spacelist = factorization_spacelist(fast_tests)

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

        @testset "Hermitian projections" begin
            for T in eltypes,
                    t in (
                        rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )
                normalize!(t)
                noisefactor = eps(real(T))^(3 / 4)

                th = (t + t') / 2
                ta = (t - t') / 2
                tc = copy(t)

                th′ = @constinferred project_hermitian(t)
                @test ishermitian(th′)
                @test th′ ≈ th
                @test t == tc
                th_approx = th + noisefactor * ta
                @test !ishermitian(th_approx) || (T <: Real && t isa DiagonalTensorMap)
                @test ishermitian(th_approx; atol = 10 * noisefactor)

                ta′ = project_antihermitian(t)
                @test isantihermitian(ta′)
                @test ta′ ≈ ta
                @test t == tc
                ta_approx = ta + noisefactor * th
                @test !isantihermitian(ta_approx)
                @test isantihermitian(ta_approx; atol = 10 * noisefactor) || (T <: Real && t isa DiagonalTensorMap)
            end
        end

        @testset "Isometric projections" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                    )
                t2 = project_isometric(t)
                @test isisometric(t2)
                t3 = project_isometric(t2)
                @test t3 ≈ t2 # stability of the projection
                @test t2 * (t2' * t) ≈ t

                tc = similar(t)
                t3 = @constinferred project_isometric!(copy!(tc, t), t2)
                @test t3 === t2
                @test isisometric(t2)

                # test that t2 is closer to A then any other isometry
                for k in 1:10
                    δt = randn!(similar(t))
                    t3 = project_isometric(t + δt / 100)
                    @test norm(t - t3) > norm(t - t2)
                end
            end
        end
    end
end
