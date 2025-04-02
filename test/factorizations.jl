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

@timedtestset "Factorizatios with symmetry: $(sectortype(first(V)))" for V in spacelist
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
    for T in (Float32, ComplexF64), adj in (false, true)
        t = adj ? rand(T, W)' : rand(T, W)
        @testset "leftorth with $alg" for alg in
                                          (TensorKit.QR(), TensorKit.QRpos(),
                                           TensorKit.QL(), TensorKit.QLpos(),
                                           TensorKit.Polar(), TensorKit.SVD(),
                                           TensorKit.SDD())
            Q, R = @constinferred leftorth(t, ((3, 4, 2), (1, 5)); alg=alg)
            QdQ = Q' * Q
            @test QdQ ≈ one(QdQ)
            @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
            if alg isa Polar
                @test isposdef(R)
                @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
            end
        end
        @testset "leftnull with $alg" for alg in
                                          (TensorKit.QR(), TensorKit.SVD(),
                                           TensorKit.SDD())
            N = @constinferred leftnull(t, ((3, 4, 2), (1, 5)); alg=alg)
            NdN = N' * N
            @test NdN ≈ one(NdN)
            @test norm(N' * permute(t, ((3, 4, 2), (1, 5)))) <
                  100 * eps(norm(t))
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
