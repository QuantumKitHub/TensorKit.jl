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

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Factorization" begin
            W = V1 ⊗ V2
            @testset for T in (Float32, ComplexF64)
                # Test both a normal tensor and an adjoint one.
                ts = (rand(T, W, W'), rand(T, W, W')')
                @testset for t in ts
                    # test squares and rectangles here
                    @testset "leftorth with $alg" for alg in
                                                      (TensorKit.LAPACK_HouseholderQR(),
                                                       TensorKit.LAPACK_HouseholderQR(;
                                                                                      positive=true),
                                                       #TensorKit.QL(),
                                                       #TensorKit.QLpos(),
                                                       TensorKit.PolarViaSVD(TensorKit.LAPACK_QRIteration()),
                                                       TensorKit.PolarViaSVD(TensorKit.LAPACK_DivideAndConquer()),
                                                       TensorKit.LAPACK_QRIteration(),
                                                       TensorKit.LAPACK_DivideAndConquer())
                        Q, R = @constinferred leftorth(t; alg=alg)
                        @test isisometry(Q)
                        tQR = Q * R
                        @test tQR ≈ t
                    end
                    @testset "leftnull with $alg" for alg in
                                                      (TensorKit.LAPACK_HouseholderQR(),
                                                       TensorKit.LAPACK_QRIteration(),
                                                       TensorKit.LAPACK_DivideAndConquer())
                        N = @constinferred leftnull(t; alg=alg)
                        @test isisometry(N)
                        @test norm(N' * t) < 100 * eps(norm(t))
                    end
                    @testset "rightorth with $alg" for alg in
                                                       (TensorKit.LAPACK_HouseholderLQ(),
                                                        TensorKit.LAPACK_HouseholderLQ(;
                                                                                       positive=true),
                                                        TensorKit.PolarViaSVD(TensorKit.LAPACK_QRIteration()),
                                                        TensorKit.PolarViaSVD(TensorKit.LAPACK_DivideAndConquer()),
                                                        TensorKit.LAPACK_QRIteration(),
                                                        TensorKit.LAPACK_DivideAndConquer())
                        L, Q = @constinferred rightorth(t; alg=alg)
                        @test isisometry(Q; side=:right)
                        @test L * Q ≈ t
                    end
                    @testset "rightnull with $alg" for alg in
                                                       (TensorKit.LAPACK_HouseholderLQ(),
                                                        TensorKit.LAPACK_QRIteration(),
                                                        TensorKit.LAPACK_DivideAndConquer())
                        M = @constinferred rightnull(t; alg=alg)
                        @test isisometry(M; side=:right)
                        @test norm(t * M') < 100 * eps(norm(t))
                    end
                    @testset "tsvd with $alg" for alg in (TensorKit.LAPACK_QRIteration(),
                                                          TensorKit.LAPACK_DivideAndConquer())
                        U, S, V = @constinferred tsvd(t; alg=alg)
                        @test isisometry(U)
                        @test isisometry(V; side=:right)
                        @test U * S * V ≈ t

                        s = LinearAlgebra.svdvals(t)
                        s′ = LinearAlgebra.diag(S)
                        for (c, b) in s
                            @test b ≈ s′[c]
                        end
                        s = LinearAlgebra.svdvals(t')
                        s′ = LinearAlgebra.diag(S')
                        for (c, b) in s
                            @test b ≈ s′[c]
                        end
                    end
                    @testset "cond and rank" begin
                        d1 = dim(codomain(t))
                        d2 = dim(domain(t))
                        @test rank(t) == min(d1, d2)
                        M = leftnull(t)
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
                        vals = LinearAlgebra.eigvals(t4')
                        λmax = maximum(s -> maximum(abs, s), values(vals))
                        λmin = minimum(s -> minimum(abs, s), values(vals))
                        @test cond(t4') ≈ λmax / λmin
                    end
                end
                @testset "empty tensor" begin
                    t = randn(T, V1 ⊗ V2, zero(V1))
                    @testset "leftorth with $alg" for alg in
                                                      (TensorKit.LAPACK_HouseholderQR(),
                                                       TensorKit.LAPACK_HouseholderQR(;
                                                                                      positive=true),
                                                       #TensorKit.QL(), TensorKit.QLpos(),
                                                       TensorKit.PolarViaSVD(TensorKit.LAPACK_QRIteration()),
                                                       TensorKit.PolarViaSVD(TensorKit.LAPACK_DivideAndConquer()),
                                                       TensorKit.LAPACK_QRIteration(),
                                                       TensorKit.LAPACK_DivideAndConquer())
                        Q, R = @constinferred leftorth(t; alg=alg)
                        @test Q == t
                        @test dim(Q) == dim(R) == 0
                    end
                    @testset "leftnull with $alg" for alg in
                                                      (TensorKit.LAPACK_HouseholderQR(),
                                                       TensorKit.LAPACK_QRIteration(),
                                                       TensorKit.LAPACK_DivideAndConquer())
                        N = @constinferred leftnull(t; alg=alg)
                        @test isunitary(N)
                    end
                    @testset "rightorth with $alg" for alg in
                                                       (TensorKit.LAPACK_HouseholderLQ(),
                                                        TensorKit.LAPACK_HouseholderLQ(;
                                                                                       positive=true),
                                                        TensorKit.PolarViaSVD(TensorKit.LAPACK_QRIteration()),
                                                        TensorKit.PolarViaSVD(TensorKit.LAPACK_DivideAndConquer()),
                                                        TensorKit.LAPACK_QRIteration(),
                                                        TensorKit.LAPACK_DivideAndConquer())
                        L, Q = @constinferred rightorth(copy(t'); alg=alg)
                        @test Q == t'
                        @test dim(Q) == dim(L) == 0
                    end
                    @testset "rightnull with $alg" for alg in
                                                       (TensorKit.LAPACK_HouseholderLQ(),
                                                        TensorKit.LAPACK_QRIteration(),
                                                        TensorKit.LAPACK_DivideAndConquer())
                        M = @constinferred rightnull(copy(t'); alg=alg)
                        @test isunitary(M)
                    end
                    @testset "tsvd with $alg" for alg in (TensorKit.LAPACK_QRIteration(),
                                                          TensorKit.LAPACK_DivideAndConquer())
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
                @testset "eig and isposdef" begin
                    t = rand(T, V1, V1)
                    D, V = eigen(t)
                    @test t * V ≈ V * D

                    d = LinearAlgebra.eigvals(t; sortby=nothing)
                    d′ = LinearAlgebra.diag(D)
                    for (c, b) in d
                        @test b ≈ d′[c]
                    end

                    # Somehow moving these test before the previous one gives rise to errors
                    # with T=Float32 on x86 platforms. Is this an OpenBLAS issue? 
                    VdV = V' * V
                    VdV = (VdV + VdV') / 2
                    @test isposdef(VdV)

                    @test !isposdef(t) # unlikely for non-hermitian map
                    t2 = (t + t')
                    D, V = eigen(t2)
                    @test isisometry(V)
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
        end
    end
end
