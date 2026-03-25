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

        @testset "Polar decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)', rand(T, W, V4), rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                @assert domain(t) ≾ codomain(t)
                w, p = @constinferred left_polar(t)
                @test w * p ≈ t
                @test isisometric(w)
                @test isposdef(p)

                w, p = @constinferred left_orth(t; alg = :polar)
                @test w * p ≈ t
                @test isisometric(w)
            end

            for T in eltypes,
                    t in (rand(T, W, W), rand(T, W, W)', rand(T, V4, W), rand(T, W, V4)')

                @assert codomain(t) ≾ domain(t)
                p, wᴴ = @constinferred right_polar(t)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
                @test isposdef(p)

                p, wᴴ = @constinferred right_orth(t; alg = :polar)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
            end
        end

        @testset "SVD" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, W, V4), rand(T, V4, W),
                        rand(T, W, V4)', rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                u, s, vᴴ = @constinferred svd_full(t)
                @test u * s * vᴴ ≈ t
                @test isunitary(u)
                @test isunitary(vᴴ)

                u, s, vᴴ = @constinferred svd_compact(t)
                @test u * s * vᴴ ≈ t
                @test isisometric(u)
                @test isposdef(s)
                @test isisometric(vᴴ; side = :right)

                s′ = @constinferred svd_vals(t)
                @test s′ ≈ diagview(s)
                @test s′ isa TensorKit.SectorVector

                s2 = @constinferred DiagonalTensorMap(s′)
                @test s2 ≈ s

                v, c = @constinferred left_orth(t; alg = :svd)
                @test v * c ≈ t
                @test isisometric(v)

                c, vᴴ = @constinferred right_orth(t; alg = :svd)
                @test c * vᴴ ≈ t
                @test isisometric(vᴴ; side = :right)

                N = @constinferred left_null(t; alg = :svd)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t; trunc = (; atol = 100 * eps(norm(t))))
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred right_null(t; alg = :svd)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred right_null(t; trunc = (; atol = 100 * eps(norm(t))))
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes, t in (rand(T, W, zerospace(V1)), rand(T, zerospace(V1), W))
                U, S, Vᴴ = @constinferred svd_full(t)
                @test U * S * Vᴴ ≈ t
                @test isunitary(U)
                @test isunitary(Vᴴ)

                U, S, Vᴴ = @constinferred svd_compact(t)
                @test U * S * Vᴴ ≈ t
                @test dim(U) == dim(S) == dim(Vᴴ) == dim(t) == 0
            end
        end

        @testset "truncated SVD" begin
            for T in eltypes,
                    t in (
                        randn(T, W, W), randn(T, W, W)',
                        randn(T, W, V4), randn(T, V4, W),
                        randn(T, W, V4)', randn(T, V4, W)',
                        DiagonalTensorMap(randn(T, reduceddim(V1)), V1),
                    )

                @constinferred normalize!(t)

                U, S, Vᴴ, ϵ = @constinferred svd_trunc(t; trunc = notrunc())
                @test U * S * Vᴴ ≈ t
                @test ϵ ≈ 0
                @test isisometric(U)
                @test isisometric(Vᴴ; side = :right)

                # when rank of t is already smaller than truncrank
                t_rank = ceil(Int, min(dim(codomain(t)), dim(domain(t))))
                U, S, Vᴴ, ϵ = @constinferred svd_trunc(t; trunc = truncrank(t_rank + 1))
                @test U * S * Vᴴ ≈ t
                @test ϵ ≈ 0
                @test isisometric(U)
                @test isisometric(Vᴴ; side = :right)

                # dimension of S is a float for IsingBimodule
                nvals = round(Int, dim(domain(S)) / 2)
                trunc = truncrank(nvals)
                U1, S1, Vᴴ1, ϵ1 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ1' ≈ U1 * S1
                @test isisometric(U1)
                @test isisometric(Vᴴ1; side = :right)
                @test norm(t - U1 * S1 * Vᴴ1) ≈ ϵ1 atol = eps(real(T))^(4 / 5)
                test_dim_isapprox(domain(S1), nvals)

                λ = minimum(diagview(S1))
                trunc = trunctol(; atol = λ - 10eps(λ))
                U2, S2, Vᴴ2, ϵ2 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ2' ≈ U2 * S2
                @test isisometric(U2)
                @test isisometric(Vᴴ2; side = :right)
                @test norm(t - U2 * S2 * Vᴴ2) ≈ ϵ2 atol = eps(real(T))^(4 / 5)
                @test minimum(diagview(S1)) >= λ
                @test U2 ≈ U1
                @test S2 ≈ S1
                @test Vᴴ2 ≈ Vᴴ1
                @test ϵ1 ≈ ϵ2

                trunc = truncspace(space(S2, 1))
                U3, S3, Vᴴ3, ϵ3 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ3' ≈ U3 * S3
                @test isisometric(U3)
                @test isisometric(Vᴴ3; side = :right)
                @test norm(t - U3 * S3 * Vᴴ3) ≈ ϵ3 atol = eps(real(T))^(4 / 5)
                @test space(S3, 1) ≾ space(S2, 1)

                for trunc in (truncerror(; atol = ϵ2), truncerror(; rtol = ϵ2 / norm(t)))
                    U4, S4, Vᴴ4, ϵ4 = @constinferred svd_trunc(t; trunc)
                    @test t * Vᴴ4' ≈ U4 * S4
                    @test isisometric(U4)
                    @test isisometric(Vᴴ4; side = :right)
                    @test norm(t - U4 * S4 * Vᴴ4) ≈ ϵ4 atol = eps(real(T))^(4 / 5)
                    @test ϵ4 ≤ ϵ2
                end

                trunc = truncrank(nvals) & trunctol(; atol = λ - 10eps(λ))
                U5, S5, Vᴴ5, ϵ5 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ5' ≈ U5 * S5
                @test isisometric(U5)
                @test isisometric(Vᴴ5; side = :right)
                @test norm(t - U5 * S5 * Vᴴ5) ≈ ϵ5 atol = eps(real(T))^(4 / 5)
                @test minimum(diagview(S5)) >= λ
                test_dim_isapprox(domain(S5), nvals)

                trunc = truncrank(nvals) | trunctol(; atol = λ - 10eps(λ))
                U5, S5, Vᴴ5, ϵ5 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ5' ≈ U5 * S5
                @test isisometric(U5)
                @test isisometric(Vᴴ5; side = :right)
                @test norm(t - U5 * S5 * Vᴴ5) ≈ ϵ5 atol = eps(real(T))^(4 / 5)
                @test minimum(diagview(S5)) >= λ
                test_dim_isapprox(domain(S5), nvals)
            end
        end
    end
end
