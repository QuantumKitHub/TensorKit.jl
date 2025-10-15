using Adapt, CUDA, cuTENSOR
ad = adapt(Array)
const CUDAExt = Base.get_extension(TensorKit, :TensorKitCUDAExt)
@assert !isnothing(CUDAExt)
const CuTensorMap = getglobal(CUDAExt, :CuTensorMap)
const CuDiagonalTensorMap = getglobal(CUDAExt, :CuDiagonalTensorMap)

for V in (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂) #, VSU₃)
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
            (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VfU₁, VfSU₂) #, VSU₃)
        else
            (Vtr, Vℤ₂, Vfℤ₂, VU₁, VCU₁, VSU₂, VfSU₂) #, VSU₃)
        end
    else
        (Vtr, VU₁, VSU₂, Vfℤ₂)
    end
catch
    (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂) #, VSU₃)
end

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("CUDA Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64)
                t = @constinferred CUDA.zeros(T, W)
                CUDA.@allowscalar begin
                    @test @constinferred(hash(t)) == hash(deepcopy(t))
                end
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, CuVector{T, CUDA.DeviceMemory}}
                # blocks
                bs = @constinferred blocks(t)
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t, first(blocksectors(t)))
                @test b1 == b2
                @test_broken eltype(bs) === Pair{typeof(c), typeof(b1)}
                @test_broken typeof(b1) === TensorKit.blocktype(t)
                @test typeof(c) === sectortype(t)
            end
        end
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                t = @constinferred CUDA.rand(T, W)
                d = convert(Dict, t)
                @test t == convert(CuTensorMap, d)
            end
        end
        if hasfusiontensor(I) || I == Trivial
            @timedtestset "Tensor Array conversion" begin
                W1 = V1 ← one(V1)
                W2 = one(V2) ← V2
                W3 = V1 ⊗ V2 ← one(V1)
                W4 = V1 ← V2
                W5 = one(V1) ← V1 ⊗ V2
                W6 = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for W in (W1, W2, W3, W4, W5, W6)
                    for T in (Int, Float32, ComplexF64)
                        if T == Int
                            t = CuTensorMap{T}(undef, W)
                            for (_, b) in blocks(t)
                                CUDA.@allowscalar CUDA.rand!(b, -20:20)
                            end
                        else
                            t = @constinferred CUDA.randn(T, W)
                        end
                        #=CUDA.@allowscalar begin
                            a = @constinferred convert(CuArray, t)
                        end
                        b = reshape(a, dim(codomain(W)), dim(domain(W)))
                        @test t ≈ @constinferred CuTensorMap(a, W)
                        @test t ≈ @constinferred CuTensorMap(b, W)
                        @test t === @constinferred CuTensorMap(t.data, W)=# # TODO convert(CuArray broken
                    end
                end
                for T in (Int, Float32, ComplexF64)
                    t = CUDA.randn(T, V1 ⊗ V2 ← zero(V1))
                    CUDA.@allowscalar begin
                        a = convert(CuArray, t)
                    end
                    @test norm(a) == 0
                end
            end
        end
        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred CUDA.rand(T, W)
                @test scalartype(t) == T
                @test space(t) == W
                @test space(t') == W'
                @test dim(t) == dim(space(t))
                @test codomain(t) == codomain(W)
                @test domain(t) == domain(W)
                # blocks for adjoint
                bs = @constinferred blocks(t')
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W'))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t', first(blocksectors(t')))
                @test b1 == b2
                @test_broken eltype(bs) === Pair{typeof(c), typeof(b1)}
                @test_broken typeof(b1) === TensorKit.blocktype(t')
                @test typeof(c) === sectortype(t)
                # linear algebra
                @test isa(@constinferred(norm(t)), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
                p = 3 * rand(Float64)
                CUDA.@allowscalar begin
                    @test norm(t + t, p) ≈ 2 * norm(t, p)
                end
                @test norm(t) ≈ norm(t')

                t2 = @constinferred rand!(similar(t))
                β = rand(T)
                #@test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2)) # broken for Irrep[CU₁]
                @test dot(β * t2, α * t) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')

                i1 = @constinferred(isomorphism(T, V1 ⊗ V2, V2 ⊗ V1))
                i2 = @constinferred(isomorphism(CuVector{T}, V2 ⊗ V1, V1 ⊗ V2))
                CUDA.@allowscalar begin
                    #@test i1 * i2 == @constinferred(id(T, V1 ⊗ V2)) # TODO
                    #@test i2 * i1 == @constinferred(id(CuVector{T}, V2 ⊗ V1)) # TODO
                    w = @constinferred(isometry(T, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)), V1))
                    @test dim(w) == 2 * dim(V1 ← V1)
                    @test w' * w == id(CuVector{T}, V1)
                    @test w * w' == (w * w')^2
                end
            end
        end
        @timedtestset "Trivial space insertion and removal" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred CUDA.rand(T, W)
                t2 = @constinferred insertleftunit(t)
                @test t2 == @constinferred insertrightunit(t)
                @test numind(t2) == numind(t) + 1
                @test space(t2) == insertleftunit(space(t))
                @test scalartype(t2) === T
                @test t.data === t2.data
                @test @constinferred(removeunit(t2, $(numind(t2)))) == t
                t3 = @constinferred insertleftunit(t; copy = true)
                @test t3 == @constinferred insertrightunit(t; copy = true)
                @test t.data !== t3.data
                for (c, b) in blocks(t)
                    @test b == block(t3, c)
                end
                @test @constinferred(removeunit(t3, $(numind(t3)))) == t
                t4 = @constinferred insertrightunit(t, 3; dual = true)
                @test numin(t4) == numin(t) && numout(t4) == numout(t) + 1
                for (c, b) in blocks(t)
                    @test b == block(t4, c)
                end
                @test @constinferred(removeunit(t4, 4)) == t
                t5 = @constinferred insertleftunit(t, 4; dual = true)
                @test numin(t5) == numin(t) + 1 && numout(t5) == numout(t)
                for (c, b) in blocks(t)
                    @test b == block(t5, c)
                end
                @test @constinferred(removeunit(t5, 4)) == t
            end
        end
        if hasfusiontensor(I)
            #=@timedtestset "Basic linear algebra: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for T in (Float32, ComplexF64)
                    t = CUDA.rand(T, W)
                    t2 = @constinferred CUDA.rand!(similar(t))
                    @test norm(t, 2) ≈ norm(convert(CuArray, t), 2)
                    @test dot(t2, t) ≈ dot(convert(CuArray, t2), convert(CuArray, t))
                    α = rand(T)
                    @test convert(CuArray, α * t) ≈ α * convert(CuArray, t)
                    @test convert(CuArray, t + t) ≈ 2 * convert(CuArray, t)
                end
            end
            @timedtestset "Real and imaginary parts" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64, ComplexF32)
                    t = @constinferred CUDA.randn(T, W, W)

                    tr = @constinferred real(t)
                    @test scalartype(tr) <: Real
                    @test real(convert(CuArray, t)) == convert(CuArray, tr)

                    ti = @constinferred imag(t)
                    @test scalartype(ti) <: Real
                    @test imag(convert(CuArray, t)) == convert(CuArray, ti)

                    tc = @inferred complex(t)
                    @test scalartype(tc) <: Complex
                    @test complex(convert(CuArray, t)) == convert(CuArray, tc)

                    tc2 = @inferred complex(tr, ti)
                    @test tc2 ≈ tc
                end
            end=# # TODO convert(CuArray is broken
        end
        @timedtestset "Tensor conversion" begin
            W = V1 ⊗ V2
            t = @constinferred CUDA.randn(W ← W)
            @test typeof(convert(CuTensorMap, t')) == typeof(t)
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            @test typeof(convert(typeof(tc), t')) == typeof(tc)
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
        end
        #=@timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            t = CUDA.randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            # TODO find a way to use CUDA here
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end=#
        @timedtestset "Permutations: test via inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = CUDA.rand(ComplexF64, W)
            t′ = CUDA.randn!(similar(t))
            for k in 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n -> p[n], k)
                    p2 = ntuple(n -> p[k + n], 5 - k)
                    CUDA.@allowscalar begin
                        t2 = @constinferred permute(t, (p1, p2))
                        t2 = permute(t, (p1, p2))
                        @test norm(t2) ≈ norm(t)
                        t2′ = permute(t′, (p1, p2))
                        @test dot(t2′, t2) ≈ dot(t′, t) ≈ dot(transpose(t2′), transpose(t2))
                    end
                end

                CUDA.@allowscalar begin
                    t3 = @constinferred repartition(t, $k)
                    t3 = repartition(t, k)
                    @test norm(t3) ≈ norm(t)
                    t3′ = @constinferred repartition!(similar(t3), t′)
                    @test norm(t3′) ≈ norm(t′)
                    @test dot(t′, t) ≈ dot(t3′, t3)
                end
            end
        end
        #=if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Permutations: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
                t = CUDA.rand(ComplexF64, W)
                a = convert(CuArray, t)
                for k in 0:5
                    for p in permutations(1:5)
                        p1 = ntuple(n -> p[n], k)
                        p2 = ntuple(n -> p[k + n], 5 - k)
                        t2 = permute(t, (p1, p2))
                        a2 = convert(CuArray, t2)
                        @test a2 ≈ permutedims(a, (p1..., p2...))
                        @test convert(CuArray, transpose(t2)) ≈
                            permutedims(a2, (5, 4, 3, 2, 1))
                    end

                    t3 = repartition(t, k)
                    a3 = convert(CuArray, t3)
                    @test a3 ≈ permutedims(
                        a, (ntuple(identity, k)..., reverse(ntuple(i -> i + k, 5 - k))...)
                    )
                end
            end
        end=# # convert(CuArray broken
        @timedtestset "Full trace: test self-consistency" begin
            t = CUDA.rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
            CUDA.@allowscalar begin
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
            end
            @test ss ≈ s2
            @test ss ≈ s3
        end
        @timedtestset "Partial trace: test self-consistency" begin
            t = CUDA.rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            @tensor t2[a, b] := t[c, d, b, d, c, a]
            @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
            @tensor t5[a, b] := t4[a, b, c, c]
            @test t2 ≈ t5
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Trace: test via conversion" begin
                t = CUDA.rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
                CUDA.@allowscalar begin
                    @tensor t2[a, b] := t[c, d, b, d, c, a]
                    @tensor t3[a, b] := ad(t)[c, d, b, d, c, a]
                end
                @test t3 ≈ ad(t2)
            end
        end
        @timedtestset "Trace and contraction" begin
            t1 = CUDA.rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
            t2 = CUDA.rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
            CUDA.@allowscalar begin
                t3 = t1 ⊗ t2
                @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
                @tensor tb[a, b] := t3[x, y, a, y, b, x]
            end
            @test ta ≈ tb
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor contraction: test via conversion" begin
                A1 = CUDA.randn(ComplexF64, V1' * V2', V3')
                A2 = CUDA.randn(ComplexF64, V3 * V4, V5)
                rhoL = CUDA.randn(ComplexF64, V1, V1)
                rhoR = CUDA.randn(ComplexF64, V5, V5)' # test adjoint tensor
                H = CUDA.randn(ComplexF64, V2 * V4, V2 * V4)
                CUDA.@allowscalar begin
                    @tensor HrA12[a, s1, s2, c] := rhoL[a, a'] * conj(A1[a', t1, b]) *
                        A2[b, t2, c'] * rhoR[c', c] *
                        H[s1, s2, t1, t2]

                    @tensor HrA12array[a, s1, s2, c] := ad(rhoL)[a, a'] *
                        conj(ad(A1)[a', t1, b]) *
                        ad(A2)[b, t2, c'] *
                        ad(rhoR)[c', c] *
                        ad(H)[s1, s2, t1, t2]
                end
                @test HrA12array ≈ ad(HrA12)
            end
        end
        @timedtestset "Index flipping: test flipping inverse" begin
            t = CUDA.rand(ComplexF64, V1 ⊗ V1' ← V1' ⊗ V1)
            for i in 1:4
                CUDA.@allowscalar begin
                    @test t ≈ flip(flip(t, i), i; inv = true)
                    @test t ≈ flip(flip(t, i; inv = true), i)
                end
            end
        end
        #=@timedtestset "Index flipping: test via explicit flip" begin
            t = CUDA.rand(ComplexF64, V1 ⊗ V1' ← V1' ⊗ V1)
            F1 = unitary(flip(V1), V1)

            CUDA.@allowscalar begin
                @tensor tf[a, b; c, d] := F1[a, a'] * t[a', b; c, d]
                @test flip(t, 1) ≈ tf
                @tensor tf[a, b; c, d] := conj(F1[b, b']) * t[a, b'; c, d]
                @test twist!(flip(t, 2), 2) ≈ tf
                @tensor tf[a, b; c, d] := F1[c, c'] * t[a, b; c', d]
                @test flip(t, 3) ≈ tf
                @tensor tf[a, b; c, d] := conj(F1[d, d']) * t[a, b; c, d']
                @test twist!(flip(t, 4), 4) ≈ tf
            end
        end=# # TODO
        @timedtestset "Index flipping: test via contraction" begin
            t1 = CUDA.rand(ComplexF64, V1 ⊗ V2 ⊗ V3 ← V4)
            t2 = CUDA.rand(ComplexF64, V2' ⊗ V5 ← V4' ⊗ V1)
            CUDA.@allowscalar begin
                @tensor ta[a, b] := t1[x, y, a, z] * t2[y, b, z, x]
                @tensor tb[a, b] := flip(t1, 1)[x, y, a, z] * flip(t2, 4)[y, b, z, x]
                @test ta ≈ tb
                @tensor tb[a, b] := flip(t1, (2, 4))[x, y, a, z] * flip(t2, (1, 3))[y, b, z, x]
                @test ta ≈ tb
                @tensor tb[a, b] := flip(t1, (1, 2, 4))[x, y, a, z] * flip(t2, (1, 3, 4))[y, b, z, x]
                @tensor tb[a, b] := flip(t1, (1, 3))[x, y, a, z] * flip(t2, (2, 4))[y, b, z, x]
                @test flip(ta, (1, 2)) ≈ tb
            end
        end
        @timedtestset "Multiplication of isometries: test properties" begin
            W2 = V4 ⊗ V5
            W1 = W2 ⊗ (oneunit(V1) ⊕ oneunit(V1))
            for T in (Float64, ComplexF64)
                t1 = randisometry(T, CuMatrix{T}, W1, W2)
                t2 = randisometry(T, CuMatrix{T}, W2 ← W2)
                @test isisometric(t1)
                @test isunitary(t2)
                P = t1 * t1'
                @test P * P ≈ P
            end
        end
        @timedtestset "Multiplication and inverse: test compatibility" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = V4 ⊗ V5
            for T in (Float64, ComplexF64)
                t1 = CUDA.rand(T, W1, W1)
                t2 = CUDA.rand(T, W2, W2)
                t = CUDA.rand(T, W1, W2)
                @test t1 * (t1 \ t) ≈ t
                @test (t / t2) * t2 ≈ t
                @test t1 \ one(t1) ≈ inv(t1)
                # @test one(t1) / t1 ≈ pinv(t1) # pinv not available in CUDA
                @test_throws SpaceMismatch inv(t)
                @test_throws SpaceMismatch t2 \ t
                @test_throws SpaceMismatch t / t1
                # pinv not available in CUDA
                # tp = pinv(t) * t
                # @test tp ≈ tp * tp
            end
        end
        #=if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Multiplication and inverse: test via conversion" begin
                W1 = V1 ⊗ V2 ⊗ V3
                W2 = V4 ⊗ V5
                for T in (Float32, Float64, ComplexF32, ComplexF64)
                    t1 = CUDA.rand(T, W1, W1)
                    t2 = CUDA.rand(T, W2, W2)
                    t = CUDA.rand(T, W1, W2)
                    d1 = dim(W1)
                    d2 = dim(W2)
                    At1 = reshape(convert(CuArray, t1), d1, d1)
                    At2 = reshape(convert(CuArray, t2), d2, d2)
                    At = reshape(convert(CuArray, t), d1, d2)
                    @test reshape(convert(CuArray, t1 * t), d1, d2) ≈ At1 * At
                    @test reshape(convert(CuArray, t1' * t), d1, d2) ≈ At1' * At
                    @test reshape(convert(CuArray, t2 * t'), d2, d1) ≈ At2 * At'
                    @test reshape(convert(CuArray, t2' * t'), d2, d1) ≈ At2' * At'

                    @test reshape(convert(CuArray, inv(t1)), d1, d1) ≈ inv(At1)
                    @test reshape(convert(CuArray, pinv(t)), d2, d1) ≈ pinv(At)

                    if T == Float32 || T == ComplexF32
                        continue
                    end

                    @test reshape(convert(CuArray, t1 \ t), d1, d2) ≈ At1 \ At
                    @test reshape(convert(CuArray, t1' \ t), d1, d2) ≈ At1' \ At
                    @test reshape(convert(CuArray, t2 \ t'), d2, d1) ≈ At2 \ At'
                    @test reshape(convert(CuArray, t2' \ t'), d2, d1) ≈ At2' \ At'

                    @test reshape(convert(CuArray, t2 / t), d2, d1) ≈ At2 / At
                    @test reshape(convert(CuArray, t2' / t), d2, d1) ≈ At2' / At
                    @test reshape(convert(CuArray, t1 / t'), d1, d2) ≈ At1 / At'
                    @test reshape(convert(CuArray, t1' / t'), d1, d2) ≈ At1' / At'
                end
            end
        end=# # convert(CuArray broken
        @timedtestset "Tensor truncation" begin
            for T in (Float32, ComplexF64)
                for p in (1, 2, 3, Inf)
                    # Test both a normal tensor and an adjoint one.
                    ts = (
                        CUDA.randn(T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
                        CUDA.randn(T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)',
                    )
                    for t in ts
                        U₀, S₀, V₀, = tsvd(t)
                        t = rmul!(t, 1 / norm(S₀, p))
                        # Probably shouldn't allow truncerr and truncdim, as these require scalar indexing?
                        CUDA.@allowscalar begin
                            U, S, V = tsvd(t; trunc = truncbelow(1 / dim(domain(S₀))), p = p)
                            U′, S′, V′ = tsvd(t; trunc = truncspace(space(S, 1)), p = p)
                        end
                        @test (U, S, V) == (U′, S′, V′)
                    end
                end
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor functions" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64)
                    t = CUDA.randn(T, W, W)
                    s = dim(W)
                    CUDA.@allowscalar begin
                        #=@test (@constinferred sqrt(t))^2 ≈ t
                        @test ad(sqrt(t)) ≈ sqrt(ad(t))
                        
                        expt = @constinferred exp(t)
                        @test ad(expt) ≈ exp(ad(t))

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
                        =# #exp not supported for CuArray
                    end
                end
            end
        end
        # Sylvester not defined for CUDA
        # @timedtestset "Sylvester equation" begin
        #     for T in (Float32, ComplexF64)
        #         tA = CUDA.rand(T, V1 ⊗ V3, V1 ⊗ V3)
        #         tB = CUDA.rand(T, V2 ⊗ V4, V2 ⊗ V4)
        #         tA = 3 // 2 * leftorth(tA; alg=Polar())[1]
        #         tB = 1 // 5 * leftorth(tB; alg=Polar())[1]
        #         tC = CUDA.rand(T, V1 ⊗ V3, V2 ⊗ V4)
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
        #
        # TODO
        @timedtestset "Tensor product: test via norm preservation" begin
            for T in (Float32, ComplexF64)
                t1 = CUDA.rand(T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
                t2 = CUDA.rand(T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
                CUDA.@allowscalar begin
                    t = @constinferred (t1 ⊗ t2)
                end
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor product: test via conversion" begin
                for T in (Float32, ComplexF64)
                    t1 = CUDA.rand(T, V2 ⊗ V3 ⊗ V1, V1)
                    t2 = CUDA.rand(T, V2 ⊗ V1 ⊗ V3, V2)
                    d1 = dim(codomain(t1))
                    d2 = dim(codomain(t2))
                    d3 = dim(domain(t1))
                    d4 = dim(domain(t2))
                    CUDA.@allowscalar begin
                        t = @constinferred (t1 ⊗ t2)
                        At = ad(t)
                        @test ad(t) ≈ ad(t1) ⊗ ad(t2)
                    end
                end
            end
        end
        @timedtestset "Tensor product: test via tensor contraction" begin
            for T in (Float32, ComplexF64)
                t1 = CUDA.rand(T, V2 ⊗ V3 ⊗ V1)
                t2 = CUDA.rand(T, V2 ⊗ V1 ⊗ V3)
                CUDA.@allowscalar begin
                    t = @constinferred (t1 ⊗ t2)
                    @tensor t′[1, 2, 3, 4, 5, 6] := t1[1, 2, 3] * t2[4, 5, 6]
                    # @test t ≈ t′ # TODO broken for symmetry: Irrep[ℤ₃]
                end
            end
        end
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "Deligne tensor product: test via conversion" begin
    Vlists1 = (Vtr,) # VSU₂)
    Vlists2 = (Vtr,) # Vℤ₂)
    @testset for Vlist1 in Vlists1, Vlist2 in Vlists2
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = CUDA.rand(T, V1 ⊗ V2, V3' ⊗ V4)
            t2 = CUDA.rand(T, W2, W1 ⊗ W1')
            CUDA.@allowscalar begin
                t = @constinferred (t1 ⊠ t2)
            end
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            CUDA.@allowscalar begin
                @test ad(t1) ⊠ ad(t2) ≈ ad(t1 ⊠ t2)
            end
        end
    end
end
