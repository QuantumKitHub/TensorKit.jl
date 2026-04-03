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

        @testset "Eigenvalue decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                d, v = @constinferred eig_full(t)
                @test t * v ≈ v * d

                d′ = @constinferred eig_vals(t)
                @test d′ ≈ diagview(d)
                @test d′ isa TensorKit.SectorVector

                d2 = @constinferred DiagonalTensorMap(d′)
                @test d2 ≈ d

                vdv = project_hermitian!(v' * v)
                @test @constinferred isposdef(vdv)
                t isa DiagonalTensorMap || @test !isposdef(t) # unlikely for non-hermitian map

                nvals = round(Int, dim(domain(t)) / 2)
                d, v = @constinferred eig_trunc(t; trunc = truncrank(nvals))
                @test t * v ≈ v * d
                test_dim_isapprox(domain(d), nvals)

                t2 = @constinferred project_hermitian(t)
                D, V = eigen(t2)
                @test isisometric(V)
                D̃, Ṽ = @constinferred eigh_full(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                λ = minimum(real, diagview(D))
                @test cond(Ṽ) ≈ one(real(T))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))

                d, v = @constinferred eigh_full(t2)
                @test t2 * v ≈ v * d
                @test isunitary(v)

                d′ = @constinferred eigh_vals(t2)
                @test d′ ≈ diagview(d)
                @test d′ isa TensorKit.SectorVector

                λ = minimum(real, diagview(d))
                @test cond(v) ≈ one(real(T))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ * one(t) + 0.1 * one(t2))
                @test !isposdef(t2 - λ * one(t) - 0.1 * one(t2))

                d, v = @constinferred eigh_trunc(t2; trunc = truncrank(nvals))
                @test t2 * v ≈ v * d
                test_dim_isapprox(domain(d), nvals)
            end
        end
    end
end
