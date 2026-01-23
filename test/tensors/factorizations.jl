using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: diagview

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

spacelist = try
    if ENV["CI"] == "true"
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
catch
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
            end
        end

        @testset "Eigenvalue decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                        # DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                d, v = @constinferred eig_full(t)
                @test t * v ≈ v * d

                d′ = @constinferred eig_vals(t)
                @test d′ ≈ diagview(d)
                @test d′ isa TensorKit.SectorVector

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
                D̃, Ṽ = @constinferred eigh_full(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                λ = minimum(real, diagview(D))
                @test cond(Ṽ) ≈ one(real(T))
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

        @testset "Condition number and rank" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, W, V4), rand(T, V4, W),
                        rand(T, W, V4)', rand(T, V4, W)',
                        DiagonalTensorMap(rand(T, reduceddim(V1)), V1),
                    )

                d1, d2 = dim(codomain(t)), dim(domain(t))
                r = rank(t)
                @test r == min(d1, d2)
                @test typeof(r) == typeof(d1)
                M = left_null(t)
                @test @constinferred(rank(M)) + r ≈ d1
                Mᴴ = right_null(t)
                @test rank(Mᴴ) + r ≈ d2
            end
            for T in eltypes
                u = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
                @test @constinferred(cond(u)) ≈ one(real(T))
                @test @constinferred(rank(u)) == dim(V1 ⊗ V2)

                t = rand(T, zerospace(V1), W)
                @test rank(t) == 0
                t2 = rand(T, zerospace(V1) * zerospace(V2), zerospace(V1) * zerospace(V2))
                @test rank(t2) == 0
                @test cond(t2) == 0.0
            end
            for T in eltypes, t in (rand(T, W, W), rand(T, W, W)')
                project_hermitian!(t)
                vals = @constinferred LinearAlgebra.eigvals(t)
                λmax = maximum(s -> maximum(abs, s), values(vals))
                λmin = minimum(s -> minimum(abs, s), values(vals))
                @test cond(t) ≈ λmax / λmin
            end
        end

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
                        randn(T, W, W), randn(T, W, W)',
                        randn(T, W, V4), randn(T, V4, W)',
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
