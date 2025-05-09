using TestEnv;
TestEnv.activate();

using Test
using TestExtras
using Random
using TensorKit
using Combinatorics
using TensorKit: ProductSector, fusiontensor, pentagon_equation, hexagon_equation
using TensorOperations
using Base.Iterators: take, product
# using SUNRepresentations: SUNIrrep
# const SU3Irrep = SUNIrrep{3}
using LinearAlgebra: LinearAlgebra
using Zygote: Zygote
using MatrixAlgebraKit

const TK = TensorKit

Random.seed!(1234)

smallset(::Type{I}) where {I<:Sector} = take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1,I2}}}) where {I1,I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i, j) in iter if dim(i) * dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1,I2,I3}}}) where {I1,I2,I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i, j, k) in iter if dim(i) * dim(j) * dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I<:Sector}
    s = collect(smallset(I))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(I::Type{<:Sector})
    try
        fusiontensor(one(I), one(I), one(I))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

# spaces
Vtr = (ℂ^3,
       (ℂ^4)',
       ℂ^5,
       ℂ^6,
       (ℂ^7)')
Vℤ₂ = (ℂ[Z2Irrep](0 => 1, 1 => 1),
       ℂ[Z2Irrep](0 => 1, 1 => 2)',
       ℂ[Z2Irrep](0 => 3, 1 => 2)',
       ℂ[Z2Irrep](0 => 2, 1 => 3),
       ℂ[Z2Irrep](0 => 2, 1 => 5))
Vfℤ₂ = (ℂ[FermionParity](0 => 1, 1 => 1),
        ℂ[FermionParity](0 => 1, 1 => 2)',
        ℂ[FermionParity](0 => 3, 1 => 2)',
        ℂ[FermionParity](0 => 2, 1 => 3),
        ℂ[FermionParity](0 => 2, 1 => 5))
Vℤ₃ = (ℂ[Z3Irrep](0 => 1, 1 => 2, 2 => 2),
       ℂ[Z3Irrep](0 => 3, 1 => 1, 2 => 1),
       ℂ[Z3Irrep](0 => 2, 1 => 2, 2 => 1)',
       ℂ[Z3Irrep](0 => 1, 1 => 2, 2 => 3),
       ℂ[Z3Irrep](0 => 1, 1 => 3, 2 => 3)')
VU₁ = (ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2),
       ℂ[U1Irrep](0 => 3, 1 => 1, -1 => 1),
       ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
       ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 3),
       ℂ[U1Irrep](0 => 1, 1 => 3, -1 => 3)')
VfU₁ = (ℂ[FermionNumber](0 => 1, 1 => 2, -1 => 2),
        ℂ[FermionNumber](0 => 3, 1 => 1, -1 => 1),
        ℂ[FermionNumber](0 => 2, 1 => 2, -1 => 1)',
        ℂ[FermionNumber](0 => 1, 1 => 2, -1 => 3),
        ℂ[FermionNumber](0 => 1, 1 => 3, -1 => 3)')
VCU₁ = (ℂ[CU1Irrep]((0, 0) => 1, (0, 1) => 2, 1 => 1),
        ℂ[CU1Irrep]((0, 0) => 3, (0, 1) => 0, 1 => 1),
        ℂ[CU1Irrep]((0, 0) => 1, (0, 1) => 0, 1 => 2)',
        ℂ[CU1Irrep]((0, 0) => 2, (0, 1) => 2, 1 => 1),
        ℂ[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 2)')
VSU₂ = (ℂ[SU2Irrep](0 => 3, 1 // 2 => 1),
        ℂ[SU2Irrep](0 => 2, 1 => 1),
        ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
        ℂ[SU2Irrep](0 => 2, 1 // 2 => 2),
        ℂ[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)')
VfSU₂ = (ℂ[FermionSpin](0 => 3, 1 // 2 => 1),
         ℂ[FermionSpin](0 => 2, 1 => 1),
         ℂ[FermionSpin](1 // 2 => 1, 1 => 1)',
         ℂ[FermionSpin](0 => 2, 1 // 2 => 2),
         ℂ[FermionSpin](0 => 1, 1 // 2 => 1, 3 // 2 => 1)')
for V in (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂)#, VSU₃)
    V1, V2, V3, V4, V5 = V

    @assert V3 * V4 * V2 ≿ V1' * V5' # necessary for leftorth tests
    @assert V3 * V4 ≾ V1' * V2' * V5' # necessary for rightorth tests
end

spacelist = try
    if ENV["CI"] == "true"
        println("Detected running on CI")
        if Sys.iswindows()
            (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂)
        elseif Sys.isapple()
            (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VfU₁, VfSU₂)#, VSU₃)
        else
            (Vtr, Vℤ₂, Vfℤ₂, VU₁, VCU₁, VSU₂, VfSU₂)#, VSU₃)
        end
    else
        (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂)#, VSU₃)
    end
catch
    (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂)#, VSU₃)
end


function test_leftorth(t, p, alg)
    Q, R = @inferred leftorth(t, p; alg)
    @test Q * R ≈ permute(t, p)
    @test isisometry(Q)
    if alg isa Polar
        @test isposdef(R)
        @test domain(R) == codomain(R) == domain(permute(space(t), p))
    end
end
function test_leftnull(t, p, alg)
    N = @inferred leftnull(t, p; alg)
    @test isisometry(N)
    @test norm(N' * permute(t, p)) ≈ 0  atol= 100 * eps(norm(t))
end

# @timedtestset "Factorizations with symmetry: $(sectortype(first(V)))" for V in spacelist
    V = collect(spacelist)[2]
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
    for T in (Float32, ComplexF64), adj in (false, true)
        t = adj ? rand(T, W)' : rand(T, W);
        @testset "leftorth with $alg" for alg in (TensorKit.QR(), TensorKit.QRpos(), TensorKit.QL(), TensorKit.QLpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
            test_leftorth(t, ((3, 4, 2), (1, 5)), alg)
        end
        @testset "leftnull with $alg" for alg in (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
            test_leftnull(t, ((3, 4, 2), (1, 5)), alg)
        end
        @testset "rightorth with $alg" for alg in
                                           (TensorKit.RQ(), TensorKit.RQpos(),
                                            TensorKit.LQ(), TensorKit.LQpos(),
                                            TensorKit.Polar(), TensorKit.SVD(),
                                            TensorKit.SDD())
            L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg=alg)
            QQd = Q * Q'
            @test QQd ≈ one(QQd)
            @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
            if alg isa Polar
                @test isposdef(L)
                @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
            end
        end
        @testset "rightnull with $alg" for alg in
                                           (TensorKit.LQ(), TensorKit.SVD(),
                                            TensorKit.SDD())
            M = @constinferred rightnull(t, ((3, 4), (2, 1, 5)); alg=alg)
            MMd = M * M'
            @test MMd ≈ one(MMd)
            @test norm(permute(t, ((3, 4), (2, 1, 5))) * M') <
                  100 * eps(norm(t))
        end
        @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
            U, S, V = @constinferred tsvd(t, ((3, 4, 2), (1, 5)); alg=alg)
            UdU = U' * U
            @test UdU ≈ one(UdU)
            VVd = V * V'
            @test VVd ≈ one(VVd)
            t2 = permute(t, ((3, 4, 2), (1, 5)))
            @test U * S * V ≈ t2

            s = LinearAlgebra.svdvals(t2)
            s′ = LinearAlgebra.diag(S)
            for (c, b) in s
                @test b ≈ s′[c]
            end
        end
        @testset "cond and rank" begin
            t2 = permute(t, ((3, 4, 2), (1, 5)))
            d1 = dim(codomain(t2))
            d2 = dim(domain(t2))
            @test rank(t2) == min(d1, d2)
            M = leftnull(t2)
            @test rank(M) == max(d1, d2) - min(d1, d2)
            t3 = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
            @test cond(t3) ≈ one(real(T))
            @test rank(t3) == dim(V1 ⊗ V2)
            t4 = randn(T, V1 ⊗ V2, V1 ⊗ V2)
            t4 = (t4 + t4') / 2
            vals = LinearAlgebra.eigvals(t4)
            λmax = maximum(s -> maximum(abs, s), values(vals))
            λmin = minimum(s -> minimum(abs, s), values(vals))
            @test cond(t4) ≈ λmax / λmin
        end
    end
    @testset "empty tensor" begin
        for T in (Float32, ComplexF64)
            t = randn(T, V1 ⊗ V2, zero(V1))
            @testset "leftorth with $alg" for alg in
                                              (TensorKit.QR(), TensorKit.QRpos(),
                                               TensorKit.QL(), TensorKit.QLpos(),
                                               TensorKit.Polar(), TensorKit.SVD(),
                                               TensorKit.SDD())
                Q, R = @constinferred leftorth(t; alg=alg)
                @test Q == t
                @test dim(Q) == dim(R) == 0
            end
            @testset "leftnull with $alg" for alg in
                                              (TensorKit.QR(), TensorKit.SVD(),
                                               TensorKit.SDD())
                N = @constinferred leftnull(t; alg=alg)
                @test N' * N ≈ id(domain(N))
                @test N * N' ≈ id(codomain(N))
            end
            @testset "rightorth with $alg" for alg in
                                               (TensorKit.RQ(), TensorKit.RQpos(),
                                                TensorKit.LQ(), TensorKit.LQpos(),
                                                TensorKit.Polar(), TensorKit.SVD(),
                                                TensorKit.SDD())
                L, Q = @constinferred rightorth(copy(t'); alg=alg)
                @test Q == t'
                @test dim(Q) == dim(L) == 0
            end
            @testset "rightnull with $alg" for alg in
                                               (TensorKit.LQ(), TensorKit.SVD(),
                                                TensorKit.SDD())
                M = @constinferred rightnull(copy(t'); alg=alg)
                @test M * M' ≈ id(codomain(M))
                @test M' * M ≈ id(domain(M))
            end
            @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                U, S, V = @constinferred tsvd(t; alg=alg)
                @test U == t
                @test dim(U) == dim(S) == dim(V)
            end
            @testset "cond and rank" begin
                @test rank(t) == 0
                W2 = zero(V1) * zero(V2)
                t2 = rand(W2, W2)
                @test rank(t2) == 0
                @test cond(t2) == 0.0
            end
        end
    end
    @testset "eig and isposdef" begin
        for T in (Float32, ComplexF64)
            t = rand(T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
            D, V = eigen(t, ((1, 3), (2, 4)))
            t2 = permute(t, ((1, 3), (2, 4)))
            @test t2 * V ≈ V * D

            d = LinearAlgebra.eigvals(t2; sortby=nothing)
            d′ = LinearAlgebra.diag(D)
            for (c, b) in d
                @test b ≈ d′[c]
            end

            # Somehow moving these test before the previous one gives rise to errors
            # with T=Float32 on x86 platforms. Is this an OpenBLAS issue? 
            VdV = V' * V
            VdV = (VdV + VdV') / 2
            @test isposdef(VdV)

            @test !isposdef(t2) # unlikely for non-hermitian map
            t2 = (t2 + t2')
            D, V = eigen(t2)
            VdV = V' * V
            @test VdV ≈ one(VdV)
            D̃, Ṽ = @constinferred eigh(t2)
            @test D ≈ D̃
            @test V ≈ Ṽ
            λ = minimum(minimum(real(LinearAlgebra.diag(b)))
                        for (c, b) in blocks(D))
            @test cond(Ṽ) ≈ one(real(T))
            @test isposdef(t2) == isposdef(λ)
            @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
            @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))
        end
    end
    @testset "Tensor truncation" begin
        for T in (Float32, ComplexF64), p in (1, 2, 3, Inf), adj in (false, true)
            t = adj ? rand(T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5) : rand(T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)'

            U₀, S₀, V₀, = tsvd(t)
            t = rmul!(t, 1 / norm(S₀, p))
            U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
            # @show p, ϵ
            # @show domain(S)
            # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
            U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
            @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
            U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))),
                                  p=p)
            @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
            U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
            @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
            # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
            U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
            # @show p, ϵ
            # @show domain(S)
            # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
            U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
            @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
        end
    end
end
