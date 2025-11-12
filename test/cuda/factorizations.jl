using LinearAlgebra, CUDA, Test, TestExtras, TensorKit, cuTENSOR

const CUDAExt = Base.get_extension(TensorKit, :TensorKitCUDAExt)
@assert !isnothing(CUDAExt)
# const CuTensorMap{T,S,N1,N2,I,A} = CUDAExt.CuTensorMap{T,S,N1,N2,I,A}
const CuTensorMap = getglobal(CUDAExt, :CuTensorMap)
const CuDiagonalTensorMap = getglobal(CUDAExt, :CuDiagonalTensorMap)

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

spacelist = try
    if ENV["CI"] == "true"
        println("Detected running on CI")
        if Sys.iswindows()
            (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂)
        elseif Sys.isapple()
            (Vtr, Vℤ₃, VfU₁, VfSU₂)
        else
            (Vtr, VU₁, VCU₁, VSU₂, VfSU₂)
        end
    else
        (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂)
    end
catch
    (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂)
end

import CUDA: rand as curand, randn as curandn


eltypes = (Float32, ComplexF64)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("CUDA Factorizations with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "CUDA Factorizations with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2
        @testset "QR decomposition" begin
            for T in eltypes,
                    t in (
                        curand(T, W, W), curand(T, W, W)', curand(T, W, V1), curand(T, V1, W)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
                    )

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)

                Q, R = @constinferred left_orth(t; kind = :qr)
                @test Q * R ≈ t
                @test isisometric(Q)

                N = @constinferred qr_null(t)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                N = @constinferred left_null(t; kind = :qr)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes
                t = curand(T, V1 ⊗ V2, zero(V1))

                Q, R = @constinferred qr_full(t)
                @test Q * R ≈ t
                @test isunitary(Q)
                @test dim(R) == dim(t) == 0

                Q, R = @constinferred qr_compact(t)
                @test Q * R ≈ t
                @test isisometric(Q)
                @test dim(Q) == dim(R) == dim(t)

                Q, R = @constinferred left_orth(t; kind = :qr)
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
                        curand(T, W, W), curand(T, W, W)', curand(T, W, V1), curand(T, V1, W)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
                    )

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                L, Q = @constinferred right_orth(t; kind = :lq)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)

                Nᴴ = @constinferred lq_null(t)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            for T in eltypes
                # empty tensor
                t = curand(T, zero(V1), V1 ⊗ V2)

                L, Q = @constinferred lq_full(t)
                @test L * Q ≈ t
                @test isunitary(Q)
                @test dim(L) == dim(t) == 0

                L, Q = @constinferred lq_compact(t)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                L, Q = @constinferred right_orth(t; kind = :lq)
                @test L * Q ≈ t
                @test isisometric(Q; side = :right)
                @test dim(Q) == dim(L) == dim(t)

                Nᴴ = @constinferred lq_null(t)
                @test isunitary(Nᴴ)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end
        end
        @testset "Polar decomposition" begin
            @testset for T in eltypes,
                    t in (
                        curand(T, W, W),
                        curand(T, W, W)',
                        curand(T, W, V1),
                        curand(T, V1, W)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
                    )

                @assert domain(t) ≾ codomain(t)
                w, p = @constinferred left_polar(t)
                @test w * p ≈ t
                @test isisometric(w)
                # broken for T <: Complex
                @test isposdef(p)

                w, p = @constinferred left_orth(t; kind = :polar)
                @test w * p ≈ t
                @test isisometric(w)
            end

            @testset for T in eltypes,
                    t in (
                        curand(T, W, W), curand(T, W, W)', curand(T, V1, W), curand(T, W, V1)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
                    )

                @assert codomain(t) ≾ domain(t)
                p, wᴴ = @constinferred right_polar(t)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
                @test isposdef(p)

                p, wᴴ = @constinferred right_orth(t; kind = :polar)
                @test p * wᴴ ≈ t
                @test isisometric(wᴴ; side = :right)
            end
        end
        @testset "SVD" begin
            for T in eltypes,
                    t in (
                        curand(T, W, W), curand(T, W, W)',
                        curand(T, W, V1), curand(T, V1, W),
                        curand(T, W, V1)', curand(T, V1, W)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
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

                s′ = LinearAlgebra.diag(s)
                for (c, b) in LinearAlgebra.svdvals(t)
                    @test b ≈ s′[c]
                end

                v, c = @constinferred left_orth(t; kind = :svd)
                @test v * c ≈ t
                @test isisometric(v)

                N = @constinferred left_null(t; kind = :svd)
                @test isisometric(N)
                @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

                Nᴴ = @constinferred right_null(t; kind = :svd)
                @test isisometric(Nᴴ; side = :right)
                @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
            end

            # empty tensor
            for T in eltypes, t in (curand(T, W, zero(V1)), curand(T, zero(V1), W))
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
                        curandn(T, W, W), curandn(T, W, W)',
                        curandn(T, W, V1), curandn(T, V1, W),
                        curandn(T, W, V1)', curandn(T, V1, W)',
                        CuDiagonalTensorMap(curandn(T, reduceddim(V1)), V1),
                    )

                @constinferred normalize!(t)

                U, S, Vᴴ = @constinferred svd_trunc(t; trunc = notrunc())
                @test U * S * Vᴴ ≈ t
                @test isisometric(U)
                @test isisometric(Vᴴ; side = :right)

                trunc = truncrank(dim(domain(S)) ÷ 2)
                CUDA.@allowscalar begin
                    U1, S1, Vᴴ1 = @constinferred svd_trunc(t; trunc)
                end
                @test t * Vᴴ1' ≈ U1 * S1
                @test isisometric(U1)
                @test isisometric(Vᴴ1; side = :right)
                @test dim(domain(S1)) <= trunc.howmany

                λ = minimum(minimum, values(LinearAlgebra.diag(S1)))
                trunc = trunctol(; atol = λ - 10eps(λ))
                CUDA.@allowscalar begin
                    U2, S2, Vᴴ2 = @constinferred svd_trunc(t; trunc)
                end
                @test t * Vᴴ2' ≈ U2 * S2
                @test isisometric(U2)
                @test isisometric(Vᴴ2; side = :right)
                @test minimum(minimum, values(LinearAlgebra.diag(S1))) >= λ
                @test U2 ≈ U1
                @test S2 ≈ S1
                @test Vᴴ2 ≈ Vᴴ1

                trunc = truncspace(space(S2, 1))
                CUDA.@allowscalar begin
                    U3, S3, Vᴴ3 = @constinferred svd_trunc(t; trunc)
                end
                @test t * Vᴴ3' ≈ U3 * S3
                @test isisometric(U3)
                @test isisometric(Vᴴ3; side = :right)
                @test space(S3, 1) ≾ space(S2, 1)

                trunc = truncerror(; atol = 0.5)
                CUDA.@allowscalar begin
                    U4, S4, Vᴴ4 = @constinferred svd_trunc(t; trunc)
                end
                @test t * Vᴴ4' ≈ U4 * S4
                @test isisometric(U4)
                @test isisometric(Vᴴ4; side = :right)
                @test norm(t - U4 * S4 * Vᴴ4) <= 0.5
            end
        end

        @testset "Eigenvalue decomposition" begin
            for T in eltypes,
                    t in (
                        curand(T, V1, V1),
                        curand(T, W, W),
                        curand(T, W, W)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
                    )

                d, v = @constinferred eig_full(t)
                @test t * v ≈ v * d

                d′ = LinearAlgebra.diag(d)
                for (c, b) in LinearAlgebra.eigvals(t)
                    @test sort(b; by = abs) ≈ sort(d′[c]; by = abs)
                end

                vdv = v' * v
                vdv = (vdv + vdv') / 2
                @test @constinferred isposdef(vdv)
                t isa CuDiagonalTensorMap || @test !isposdef(t) # unlikely for non-hermitian map

                CUDA.@allowscalar begin # TODO
                    d, v = @constinferred eig_trunc(t; trunc = truncrank(dim(domain(t)) ÷ 2))
                end
                @test t * v ≈ v * d
                @test dim(domain(d)) ≤ dim(domain(t)) ÷ 2

                t2 = (t + t')
                D, V = eigen(t2)
                @test isisometric(V)
                D̃, Ṽ = @constinferred eigh_full(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                λ = minimum(
                    minimum(real(LinearAlgebra.diag(b)))
                        for (c, b) in blocks(D)
                )
                @test cond(Ṽ) ≈ one(real(T))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))

                add!(t, t')

                d, v = @constinferred eigh_full(t)
                @test t * v ≈ v * d
                @test isunitary(v)

                λ = minimum(minimum(real(LinearAlgebra.diag(b))) for (c, b) in blocks(d))
                @test cond(v) ≈ one(real(T))
                @test isposdef(t) == isposdef(λ)
                @test isposdef(t - λ * one(t) + 0.1 * one(t))
                @test !isposdef(t - λ * one(t) - 0.1 * one(t))

                CUDA.@allowscalar begin
                    d, v = @constinferred eigh_trunc(t; trunc = truncrank(dim(domain(t)) ÷ 2))
                end
                @test t * v ≈ v * d
                @test dim(domain(d)) ≤ dim(domain(t)) ÷ 2
            end
        end

        @testset "Condition number and rank" begin
            for T in eltypes,
                    t in (
                        curand(T, W, W), curand(T, W, W)',
                        curand(T, W, V1), curand(T, V1, W),
                        curand(T, W, V1)', curand(T, V1, W)',
                        CuDiagonalTensorMap(curand(T, reduceddim(V1)), V1),
                    )

                d1, d2 = dim(codomain(t)), dim(domain(t))
                @test rank(t) == min(d1, d2)
                M = left_null(t)
                @test @constinferred(rank(M)) + rank(t) == d1
                Mᴴ = right_null(t)
                @test rank(Mᴴ) + rank(t) == d2
            end
            for T in eltypes
                u = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
                @test @constinferred(cond(u)) ≈ one(real(T))
                @test @constinferred(rank(u)) == dim(V1 ⊗ V2)

                t = curand(T, zero(V1), W)
                @test rank(t) == 0
                t2 = curand(T, zero(V1) * zero(V2), zero(V1) * zero(V2))
                @test rank(t2) == 0
                @test cond(t2) == 0.0
            end
            for T in eltypes, t in (curand(T, W, W), curand(T, W, W)')
                add!(t, t')
                vals = @constinferred LinearAlgebra.eigvals(t)
                λmax = maximum(s -> maximum(abs, s), values(vals))
                λmin = minimum(s -> minimum(abs, s), values(vals))
                @test cond(t) ≈ λmax / λmin
            end
        end
    end
end
