using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: defaulttol, diagview

spacelist = factorization_spacelist(fast_tests)

eltypes = (Float32, ComplexF64)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------------------------------")
    println("Singular value and polar decompositions with symmetry: $Istr")
    println("---------------------------------------------------------------")
    @timedtestset "Singular value and polar decompositions with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2 ⊗ V3
        Vd = fuse(V1 ⊗ V2)

        @testset "Condition number and rank" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                        rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
                    )

                d1, d2 = dim(codomain(t)), dim(domain(t))
                r = rank(t)
                @test r ≈ min(d1, d2)
                @test typeof(r) == typeof(d1)
                M = left_null(t)
                @test @constinferred(rank(M)) + r ≈ d1
                Mᴴ = right_null(t)
                @test rank(Mᴴ) + r ≈ d2
            end
            for T in eltypes
                u = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
                @test @constinferred(cond(u)) ≈ one(real(T))
                @test @constinferred(rank(u)) ≈ dim(V1 ⊗ V2)

                t = rand(T, zerospace(V1), W)
                @test rank(t) == 0
                t2 = rand(T, zerospace(V1) * zerospace(V2), zerospace(V1) * zerospace(V2))
                @test rank(t2) == 0
                @test cond(t2) == 0.0
            end
            for T in eltypes, t in (rand(T, W, W), rand(T, W, W)')
                project_hermitian!(t)
                vals = @constinferred eigh_vals(t)
                λmax = maximum(abs, vals)
                λmin = minimum(abs, vals)
                @test cond(t) ≈ λmax / λmin
            end
        end

        @testset "Polar decomposition" begin
            for T in eltypes,
                    t in (
                        rand(T, W, W),
                        rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'),
                        rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
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
                    t in (
                        rand(T, W, W),
                        rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'),
                        rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
                    )

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
                        rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                        rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
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

                atol = norm(t) * defaulttol(T) # tol used by `:svd` left_null/right_null

                N = @constinferred left_null(t; alg = :svd)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = atol

                N = @constinferred left_null(t; trunc = (; atol = 6 * atol))
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 10 * atol

                Nᴴ = @constinferred right_null(t; alg = :svd)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = atol

                Nᴴ = @constinferred right_null(t; trunc = (; atol = 6 * atol))
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 10 * atol
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
                        rand(T, W, W), rand(T, W, W)',
                        rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                        rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                        DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
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
                @test abs(dim(domain(S1)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

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
                @test spacetype(typeof(trunc)) == spacetype(W)
                @test sectortype(trunc) == sectortype(W)
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
                @test abs(dim(domain(S5)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

                trunc = truncrank(nvals) | trunctol(; atol = λ - 10eps(λ))
                U5, S5, Vᴴ5, ϵ5 = @constinferred svd_trunc(t; trunc)
                @test t * Vᴴ5' ≈ U5 * S5
                @test isisometric(U5)
                @test isisometric(Vᴴ5; side = :right)
                @test norm(t - U5 * S5 * Vᴴ5) ≈ ϵ5 atol = eps(real(T))^(4 / 5)
                @test minimum(diagview(S5)) >= λ
                @test abs(dim(domain(S5)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)
            end
        end
    end
end
