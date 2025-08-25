using Adapt
ad = adapt(Array)

const AMDGPUExt = Base.get_extension(TensorKit, :TensorKitAMDGPUExt)
@assert !isnothing(AMDGPUExt)
# const ROCTensorMap{T,S,N1,N2,I,A} = AMDGPUExt.ROCTensorMap{T,S,N1,N2,I,A}
const ROCTensorMap = getglobal(AMDGPUExt, :ROCTensorMap)

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
        (Vtr, VU₁, VSU₂, Vfℤ₂)
    end
catch
    (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂)#, VSU₃)
end

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("AMDGPU Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64)
                t = @constinferred AMDGPU.zeros(T, W)
                # @test @constinferred(hash(t)) == hash(deepcopy(t)) # hash is not defined for CuArray?
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) ==
                      @constinferred tensormaptype(spacetype(t), 5, 0, storagetype(t))
            end
        end
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                t = @constinferred AMDGPU.rand(T, W)
                d = convert(Dict, t)
                @test t == convert(ROCTensorMap, d)
            end
        end
        if hasfusiontensor(I)
            @timedtestset "Tensor Array conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5

                # cuTENSOR does not support Int
                Ts = sectortype(W) === Trivial ? (Int, Float32, ComplexF64) :
                     (Float32, ComplexF64)
                for T in Ts
                    t = @constinferred AMDGPU.rand(T, W)
                    a = @constinferred convert(CuArray, t)
                    @test t ≈ @constinferred TensorMap(a, W)
                    # also test if input is matrix
                    a2 = reshape(a, prod(dim, codomain(t)), prod(dim, domain(t)))
                    @test t ≈ @constinferred TensorMap(a2, codomain(t), domain(t))
                end
            end
        end
        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred AMDGPU.rand(T, W)
                @test scalartype(t) == T
                @test space(t) == W
                @test space(t') == W'
                @test dim(t) == dim(space(t))
                @test codomain(t) == codomain(W)
                @test domain(t) == domain(W)
                @test isa(@constinferred(norm(t)), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
                p = 3 * rand(Float64)
                @test norm(t + t, p) ≈ 2 * norm(t, p)
                @test norm(t) ≈ norm(t')

                t2 = @constinferred rand!(similar(t))
                β = rand(T)
                @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')

                A = storagetype(t)

                i1 = @constinferred(isomorphism(CuMatrix{T}, V1 ⊗ V2, V2 ⊗ V1))
                i2 = @constinferred(isomorphism(CuMatrix{T}, V2 ⊗ V1, V1 ⊗ V2))
                @test i1 * i2 == @constinferred(id(CuMatrix{T}, V1 ⊗ V2))
                @test i2 * i1 == @constinferred(id(CuMatrix{T}, V2 ⊗ V1))

                w = @constinferred(isometry(CuMatrix{T}, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)),
                                            V1))
                @test dim(w) == 2 * dim(V1 ← V1)
                @test w' * w == id(CuMatrix{T}, V1)
                @test w * w' == (w * w')^2
            end
        end
        if hasfusiontensor(I)
            @timedtestset "Basic linear algebra: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for T in (Float32, ComplexF64)
                    t = AMDGPU.rand(T, W)
                    t2 = @constinferred rand!(similar(t))
                    @test norm(t, 2) ≈ norm(ad(t), 2)
                    @test dot(t2, t) ≈ dot(ad(t2), ad(t))
                    α = rand(T)
                    @test ad(α * t) ≈ α * ad(t)
                    @test ad(t + t) ≈ 2 * ad(t)
                end
            end
            @timedtestset "Real and imaginary parts" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64, ComplexF32)
                    t = @constinferred AMDGPU.randn(T, W, W)
                    @test real(ad(t)) == ad(@constinferred real(t))
                    @test imag(ad(t)) == ad(@constinferred imag(t))
                end
            end
        end
        @timedtestset "Tensor conversion" begin
            W = V1 ⊗ V2
            t = @constinferred AMDGPU.randn(W ← W)
            @test typeof(convert(TensorMap, t')) == typeof(t)
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            @test typeof(convert(typeof(tc), t')) == typeof(tc)
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
        end
        @timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            t = AMDGPU.randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end
        @timedtestset "Permutations: test via inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = AMDGPU.rand(ComplexF64, W)
            t′ = randn!(similar(t))
            for k in 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n -> p[n], k)
                    p2 = ntuple(n -> p[k + n], 5 - k)
                    t2 = @constinferred permute(t, (p1, p2))
                    @test norm(t2) ≈ norm(t)
                    t2′ = permute(t′, (p1, p2))
                    @test dot(t2′, t2) ≈ dot(t′, t) ≈ dot(transpose(t2′), transpose(t2))
                end

                t3 = VERSION < v"1.7" ? repartition(t, k) :
                     @constinferred repartition(t, $k)
                @test norm(t3) ≈ norm(t)
                t3′ = @constinferred repartition!(similar(t3), t′)
                @test norm(t3′) ≈ norm(t′)
                @test dot(t′, t) ≈ dot(t3′, t3)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Permutations: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
                t = AMDGPU.rand(ComplexF64, W)
                a = ad(t)
                for k in 0:5
                    for p in permutations(1:5)
                        p1 = ntuple(n -> p[n], k)
                        p2 = ntuple(n -> p[k + n], 5 - k)
                        t2 = permute(t, (p1, p2))
                        a2 = ad(t2)
                        @test a2 ≈ permute(a, (p1, p2))
                        @test ad(transpose(t2)) ≈ transpose(a2)
                    end

                    t3 = repartition(t, k)
                    a3 = ad(t3)
                    @test a3 ≈ repartition(a, k)
                end
            end
        end
        @timedtestset "Full trace: test self-consistency" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
            t2 = permute(t, ((1, 2), (4, 3)))
            s = @constinferred tr(t2)
            @test conj(s) ≈ tr(t2')
            if !isdual(V1)
                t2 = twist!(t2, 1)
            end
            if isdual(V2)
                t2 = twist!(t2, 2)
            end
            ss = tr(t2)
            @tensor s2 = t[a, b, b, a]
            @tensor t3[a, b] := t[a, c, c, b]
            @tensor s3 = t3[a, a]
            @test ss ≈ s2
            @test ss ≈ s3
        end
        @timedtestset "Partial trace: test self-consistency" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            @tensor t2[a, b] := t[c, d, b, d, c, a]
            @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
            @tensor t5[a, b] := t4[a, b, c, c]
            @test t2 ≈ t5
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Trace: test via conversion" begin
                t = AMDGPU.rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
                @tensor t2[a, b] := t[c, d, b, d, c, a]
                @tensor t3[a, b] := ad(t)[c, d, b, d, c, a]
                @test t3 ≈ ad(t2)
            end
        end
        @timedtestset "Trace and contraction" begin
            t1 = AMDGPU.rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
            t2 = AMDGPU.rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
            t3 = t1 ⊗ t2
            @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
            @tensor tb[a, b] := t3[x, y, a, y, b, x]
            @test ta ≈ tb
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor contraction: test via conversion" begin
                A1 = AMDGPU.randn(ComplexF64, V1' * V2', V3')
                A2 = AMDGPU.randn(ComplexF64, V3 * V4, V5)
                rhoL = AMDGPU.randn(ComplexF64, V1, V1)
                rhoR = AMDGPU.randn(ComplexF64, V5, V5)' # test adjoint tensor
                H = AMDGPU.randn(ComplexF64, V2 * V4, V2 * V4)
                @tensor HrA12[a, s1, s2, c] := rhoL[a, a'] * conj(A1[a', t1, b]) *
                                               A2[b, t2, c'] * rhoR[c', c] *
                                               H[s1, s2, t1, t2]

                @tensor HrA12array[a, s1, s2, c] := ad(rhoL)[a, a'] *
                                                    conj(ad(A1)[a', t1, b]) *
                                                    ad(A2)[b, t2, c'] *
                                                    ad(rhoR)[c', c] *
                                                    ad(H)[s1, s2, t1, t2]

                @test HrA12array ≈ ad(HrA12)
            end
        end
        @timedtestset "Multiplication and inverse: test compatibility" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = V4 ⊗ V5
            for T in (Float64, ComplexF64)
                t1 = AMDGPU.rand(T, W1, W1)
                t2 = AMDGPU.rand(T, W2, W2)
                t = AMDGPU.rand(T, W1, W2)
                @test t1 * (t1 \ t) ≈ t
                @test (t / t2) * t2 ≈ t
                @test t1 \ one(t1) ≈ inv(t1)
                @test one(t1) / t1 ≈ pinv(t1)
                @test_throws SpaceMismatch inv(t)
                @test_throws SpaceMismatch t2 \ t
                @test_throws SpaceMismatch t / t1
                tp = pinv(t) * t
                @test tp ≈ tp * tp
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Multiplication and inverse: test via conversion" begin
                W1 = V1 ⊗ V2 ⊗ V3
                W2 = V4 ⊗ V5
                for T in (Float32, Float64, ComplexF32, ComplexF64)
                    t1 = AMDGPU.rand(T, W1, W1)
                    t2 = AMDGPU.rand(T, W2, W2)
                    t = AMDGPU.rand(T, W1, W2)
                    d1 = dim(W1)
                    d2 = dim(W2)
                    At1 = reshape(convert(Array, t1), d1, d1)
                    At2 = reshape(convert(Array, t2), d2, d2)
                    At = reshape(convert(Array, t), d1, d2)
                    @test ad(t1 * t) ≈ ad(t1) * ad(t)
                    @test ad(t1' * t) ≈ ad(t1)' * ad(t)
                    @test ad(t2 * t') ≈ ad(t2) * ad(t)'
                    @test ad(t2' * t') ≈ ad(t2)' * ad(t)'
                    @test ad(inv(t1)) ≈ inv(ad(t1))
                    @test ad(pinv(t)) ≈ pinv(ad(t))

                    if T == Float32 || T == ComplexF32
                        continue
                    end

                    @test ad(t1 \ t) ≈ ad(t1) \ ad(t)
                    @test ad(t1' \ t) ≈ ad(t1)' \ ad(t)
                    @test ad(t2 \ t') ≈ ad(t2) \ ad(t)'
                    @test ad(t2' \ t') ≈ ad(t2)' \ ad(t)'

                    @test ad(t2 / t) ≈ ad(t2) / ad(t)
                    @test ad(t2' / t) ≈ ad(t2)' / ad(t)
                    @test ad(t1 / t') ≈ ad(t1) / ad(t)'
                    @test ad(t1' / t') ≈ ad(t1)' / ad(t)'
                end
            end
        end
        @timedtestset "Factorization" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                # Test both a normal tensor and an adjoint one.
                ts = (AMDGPU.rand(T, W), AMDGPU.rand(T, W)')
                for t in ts
                    @testset "leftorth with $alg" for alg in
                                                      (TensorKit.QR(), TensorKit.QRpos(),
                                                       TensorKit.QL(), TensorKit.QLpos(),
                                                       TensorKit.Polar(), # TensorKit.SVD(),
                                                       TensorKit.SDD())
                        Q, R = @constinferred leftorth(t, ((3, 4, 2), (1, 5)); alg=alg)
                        QdQ = Q' * Q
                        @test QdQ ≈ one(QdQ)
                        @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
                        if alg isa Polar
                            # @test isposdef(R) # not defined for AMDGPU
                            @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
                        end
                    end
                    @testset "leftnull with $alg" for alg in
                                                      (TensorKit.QR(), # TensorKit.SVD(),
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
                                                        TensorKit.Polar(), #TensorKit.SVD(),
                                                        TensorKit.SDD())
                        # cusolver SVD requires m >= n for some reason
                        L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg=alg)
                        QQd = Q * Q'
                        @test QQd ≈ one(QQd)
                        @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
                        if alg isa Polar
                            # @test isposdef(L) # not defined for AMDGPU
                            @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
                        end
                    end
                    @testset "rightnull with $alg" for alg in
                                                       (TensorKit.LQ(), # TensorKit.SVD(),
                                                        TensorKit.SDD())
                        M = @constinferred rightnull(t, ((3, 4), (2, 1, 5)); alg=alg)
                        MMd = M * M'
                        @test MMd ≈ one(MMd)
                        @test norm(permute(t, ((3, 4), (2, 1, 5))) * M') <
                              100 * eps(norm(t))
                    end
                    @testset "tsvd with $alg" for alg in (TensorKit.SDD(),)
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
                end
                @testset "empty tensor" begin
                    t = AMDGPU.randn(T, V1 ⊗ V2, typeof(V1)())
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
                        @test N' * N ≈ id(storagetype(t), domain(N))
                        @test N * N' ≈ id(storagetype(t), codomain(N))
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
                        @test M * M' ≈ id(storagetype(t), codomain(M))
                        @test M' * M ≈ id(storagetype(t), domain(M))
                    end
                    @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                        U, S, V = @constinferred tsvd(t; alg=alg)
                        @test U == t
                        @test dim(U) == dim(S) == dim(V)
                    end
                end

                # AMDGPU only supports symmetric/hermitian eigen
                @testset "eig and isposdef" begin
                    t = AMDGPU.rand(T, V1 ⊗ V2 ← V1 ⊗ V2)
                    t += t'
                    D, V = eigen(t)
                    @test t * V ≈ V * D

                    # d = LinearAlgebra.eigvals(t; sortby=nothing)
                    # d′ = LinearAlgebra.diag(D)
                    # for (c, b) in d
                    #     @test b ≈ d′[c]
                    # end

                    # Somehow moving these test before the previous one gives rise to errors
                    # with T=Float32 on x86 platforms. Is this an OpenBLAS issue? 
                    # VdV = V' * V
                    # VdV = (VdV + VdV') / 2
                    # @test isposdef(VdV)
                    #
                    # @test !isposdef(t2) # unlikely for non-hermitian map
                    # t2 = (t2 + t2')
                    # D, V = eigen(t2)
                    # VdV = V' * V
                    # @test VdV ≈ one(VdV)
                    # D̃, Ṽ = @constinferred eigh(t2)
                    # @test D ≈ D̃
                    # @test V ≈ Ṽ
                    # λ = minimum(minimum(real(LinearAlgebra.diag(b)))
                    #             for (c, b) in blocks(D))
                    # @test isposdef(t2) == isposdef(λ)
                    # @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
                    # @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))
                end
            end
        end
        @timedtestset "Tensor truncation" begin
            for T in (Float32, ComplexF64)
                for p in (1, 2, 3, Inf)
                    # Test both a normal tensor and an adjoint one.
                    ts = (AMDGPU.randn(T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
                          AMDGPU.randn(T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
                    for t in ts
                        U₀, S₀, V₀, = tsvd(t)
                        t = rmul!(t, 1 / norm(S₀, p))
                        # Probably shouldn't allow truncerr and truncdim, as these require scalar indexing?
                        U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    end
                end
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor functions" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64)
                    t = AMDGPU.randn(T, W, W)
                    s = dim(W)
                    expt = @constinferred exp(t)
                    @test ad(expt) ≈ exp(ad(t))

                    @test (@constinferred sqrt(t))^2 ≈ t
                    @test ad(sqrt(t)) ≈ sqrt(ad(t))

                    @test exp(@constinferred log(expt)) ≈ expt
                    @test ad(log(expt)) ≈ log(ad(expt))

                    @test (@constinferred cos(t))^2 + (@constinferred sin(t))^2 ≈
                          id(storagetype(t), W)
                    @test (@constinferred tan(t)) ≈ sin(t) / cos(t)
                    @test (@constinferred cot(t)) ≈ cos(t) / sin(t)
                    @test (@constinferred cosh(t))^2 - (@constinferred sinh(t))^2 ≈
                          id(storagetype(t), W)
                    @test (@constinferred tanh(t)) ≈ sinh(t) / cosh(t)
                    @test (@constinferred coth(t)) ≈ cosh(t) / sinh(t)

                    t1 = sin(t)
                    @test sin(@constinferred asin(t1)) ≈ t1
                    t2 = cos(t)
                    @test cos(@constinferred acos(t2)) ≈ t2
                    t3 = sinh(t)
                    @test sinh(@constinferred asinh(t3)) ≈ t3
                    t4 = cosh(t)
                    @test cosh(@constinferred acosh(t4)) ≈ t4
                    t5 = tan(t)
                    @test tan(@constinferred atan(t5)) ≈ t5
                    t6 = cot(t)
                    @test cot(@constinferred acot(t6)) ≈ t6
                    t7 = tanh(t)
                    @test tanh(@constinferred atanh(t7)) ≈ t7
                    t8 = coth(t)
                    @test coth(@constinferred acoth(t8)) ≈ t8
                end
            end
        end
        # Sylvester not defined for AMDGPU
        # @timedtestset "Sylvester equation" begin
        #     for T in (Float32, ComplexF64)
        #         tA = AMDGPU.rand(T, V1 ⊗ V3, V1 ⊗ V3)
        #         tB = AMDGPU.rand(T, V2 ⊗ V4, V2 ⊗ V4)
        #         tA = 3 // 2 * leftorth(tA; alg=Polar())[1]
        #         tB = 1 // 5 * leftorth(tB; alg=Polar())[1]
        #         tC = AMDGPU.rand(T, V1 ⊗ V3, V2 ⊗ V4)
        #         t = @constinferred sylvester(tA, tB, tC)
        #         @test codomain(t) == V1 ⊗ V3
        #         @test domain(t) == V2 ⊗ V4
        #         @test norm(tA * t + t * tB + tC) <
        #               (norm(tA) + norm(tB) + norm(tC)) * eps(real(T))^(2 / 3)
        #         if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
        #             matrix(x) = reshape(convert(Array, x), dim(codomain(x)), dim(domain(x)))
        #             @test matrix(t) ≈ sylvester(matrix(tA), matrix(tB), matrix(tC))
        #         end
        #     end
        # end
        @timedtestset "Tensor product: test via norm preservation" begin
            for T in (Float32, ComplexF64)
                t1 = AMDGPU.rand(T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
                t2 = AMDGPU.rand(T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
                t = @constinferred (t1 ⊗ t2)
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor product: test via conversion" begin
                for T in (Float32, ComplexF64)
                    t1 = AMDGPU.rand(T, V2 ⊗ V3 ⊗ V1, V1)
                    t2 = AMDGPU.rand(T, V2 ⊗ V1 ⊗ V3, V2)
                    t = @constinferred (t1 ⊗ t2)
                    d1 = dim(codomain(t1))
                    d2 = dim(codomain(t2))
                    d3 = dim(domain(t1))
                    d4 = dim(domain(t2))
                    At = ad(t)
                    @test ad(t) ≈ ad(t1) ⊗ ad(t2)
                end
            end
        end
        @timedtestset "Tensor product: test via tensor contraction" begin
            for T in (Float32, ComplexF64)
                t1 = AMDGPU.rand(T, V2 ⊗ V3 ⊗ V1)
                t2 = AMDGPU.rand(T, V2 ⊗ V1 ⊗ V3)
                t = @constinferred (t1 ⊗ t2)
                @tensor t′[1, 2, 3, 4, 5, 6] := t1[1, 2, 3] * t2[4, 5, 6]
                @test t ≈ t′
            end
        end
    end
end

@timedtestset "Deligne tensor product: test via conversion" begin
    Vlists1 = (Vtr,) # VSU₂)
    Vlists2 = (Vtr,) # Vℤ₂)
    @testset for Vlist1 in Vlists1, Vlist2 in Vlists2
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = AMDGPU.rand(T, V1 ⊗ V2, V3' ⊗ V4)
            t2 = AMDGPU.rand(T, W2, W1 ⊗ W1')
            t = @constinferred (t1 ⊠ t2)
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            @test ad(t1) ⊠ ad(t2) ≈ ad(t1 ⊠ t2)
        end
    end
end

