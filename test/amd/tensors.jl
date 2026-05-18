using Adapt, AMDGPU
using Test, TestExtras
using TensorKit, TensorKitSectors, Combinatorics
ad = adapt(Array)
const AMDGPUExt = Base.get_extension(TensorKit, :TensorKitAMDGPUExt)
@assert !isnothing(AMDGPUExt)
const ROCTensorMap = getglobal(AMDGPUExt, :ROCTensorMap)
const rocrand = getglobal(AMDGPUExt, :rocrand)
const rocrandn = getglobal(AMDGPUExt, :rocrandn)
const rocrand! = getglobal(AMDGPUExt, :rocrand!)
using AMDGPU: rand as rocrand, rand! as rocrand!, randn as rocrandn, randn! as rocrandn!

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("AMDGPU Tensors with symmetry: $Istr")
    println("---------------------------------------")
    symmetricbraiding = BraidingStyle(I) isa SymmetricBraiding
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            # test default pass-throughs
            for f in (AMDGPU.zeros, AMDGPU.ones, rocrand, rocrandn)
                t = @constinferred f(W)
                @test scalartype(t) == Float64
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{Float64, spacetype(t), 5, 0, ROCVector{Float64, AMDGPU.Mem.HIPBuffer}}
            end
            for f in (rand, randn)
                t = @constinferred f(ROCVector{Float64, AMDGPU.Mem.HIPBuffer}, W)
                @test scalartype(t) == Float64
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{Float64, spacetype(t), 5, 0, ROCVector{Float64, AMDGPU.Mem.HIPBuffer}}
            end
            for f! in (rocrand!, rocrandn!)
                t = @constinferred AMDGPU.zeros(W)
                f!(t)
                @test scalartype(t) == Float64
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{Float64, spacetype(t), 5, 0, ROCVector{Float64, AMDGPU.Mem.HIPBuffer}}
            end
        end
        @timedtestset "Conversion to/from host" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                h_t = @constinferred rand(T, W)
                t1 = convert(ROCTensorMap{T}, h_t)
                @test collect(t1.data) == h_t.data
                @test space(t1) == space(h_t)
                @test scalartype(t1) == T
                @test codomain(t1) == W
                @test space(t1) == (W ← one(W))
                @test domain(t1) == one(W)
                t2 = ROCTensorMap(h_t)
                @test collect(t2.data) == h_t.data
                @test space(t2) == space(h_t)
                @test scalartype(t2) == T
                @test codomain(t2) == W
                @test space(t2) == (W ← one(W))
                @test domain(t2) == one(W)
            end
        end
        @timedtestset "Adapt" begin
            W = V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)'
            for T in (Int, Float32, ComplexF64)
                t = rand(T, W)
                t_gpu = @constinferred adapt(ROCArray, t)
                @test storagetype(t_gpu) <: ROCArray{T}
                @test scalartype(t_gpu) === scalartype(t)
                @test collect(t_gpu.data) == t.data

                t_cpu = @constinferred adapt(Array, t_gpu)
                @test t_cpu == t
                @test storagetype(t_cpu) <: Array{T}
            end
        end
        @timedtestset "Trivial space insertion and removal" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred AMDGPU.rand(T, W)
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
        @timedtestset "Tensor conversion" begin # TODO adjoint conversion methods don't work yet
            W = V1 ⊗ V2
            t = @constinferred AMDGPU.randn(W ← W)
            #@test typeof(convert(TensorMap, t')) == typeof(t) # TODO Adjoint not supported yet
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            # @test typeof(convert(typeof(tc), t')) == typeof(tc) # TODO Adjoint not supported yet
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
        end
        symmetricbraiding && @timedtestset "Permutations: test via inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = AMDGPU.rand(ComplexF64, W)
            t′ = AMDGPU.randn!(similar(t))
            for k in 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n -> p[n], k)
                    p2 = ntuple(n -> p[k + n], 5 - k)
                    AMDGPU.@allowscalar begin
                        t2 = @constinferred permute(t, (p1, p2))
                        t2 = permute(t, (p1, p2))
                        @test norm(t2) ≈ norm(t)
                        t2′ = permute(t′, (p1, p2))
                        @test dot(t2′, t2) ≈ dot(t′, t) ≈ dot(transpose(t2′), transpose(t2))
                    end
                end

                AMDGPU.@allowscalar begin
                    t3 = @constinferred repartition(t, $k)
                    @test norm(t3) ≈ norm(t)
                    t3′ = @constinferred repartition!(similar(t3), t′)
                    @test norm(t3′) ≈ norm(t′)
                    @test dot(t′, t) ≈ dot(t3′, t3)
                end
            end
        end
        symmetricbraiding && @timedtestset "Permutations: test via CPU" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = AMDGPU.rand(ComplexF64, W)
            for k in 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n -> p[n], k)
                    p2 = ntuple(n -> p[k + n], 5 - k)
                    dt2 = AMDGPU.@allowscalar permute(t, (p1, p2))
                    ht2 = permute(TensorKit.to_cpu(t), (p1, p2))
                    @test ht2 == TensorKit.to_cpu(dt2)
                end

                dt3 = AMDGPU.@allowscalar repartition(t, k)
                ht3 = repartition(TensorKit.to_cpu(t), k)
                @test ht3 == TensorKit.to_cpu(dt3)
            end
        end
        symmetricbraiding && @timedtestset "Full trace: test self-consistency" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
            AMDGPU.@allowscalar begin
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
        #=symmetricbraiding && @timedtestset "Partial trace: test self-consistency" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            @tensor t2[a, b] := t[c, d, b, d, c, a]
            @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
            @tensor t5[a, b] := t4[a, b, c, c]
            @test t2 ≈ t5
        end
        symmetricbraiding && @timedtestset "Trace: test via conversion" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            AMDGPU.@allowscalar begin
                @tensor t2[a, b] := t[c, d, b, d, c, a]
                @tensor t3[a, b] := ad(t)[c, d, b, d, c, a]
            end
            @test t3 ≈ ad(t2)
        end
        symmetricbraiding && @timedtestset "Trace and contraction" begin
            t1 = AMDGPU.rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
            t2 = AMDGPU.rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
            AMDGPU.@allowscalar begin
                t3 = t1 ⊗ t2
                @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
                @tensor tb[a, b] := t3[x, y, a, y, b, x]
            end
            @test ta ≈ tb
        end
        #=if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor contraction: test via CPU" begin
                dA1 = AMDGPU.randn(ComplexF64, V1' * V2', V3')
                dA2 = AMDGPU.randn(ComplexF64, V3 * V4, V5)
                drhoL = AMDGPU.randn(ComplexF64, V1, V1)
                drhoR = AMDGPU.randn(ComplexF64, V5, V5)' # test adjoint tensor
                dH = AMDGPU.randn(ComplexF64, V2 * V4, V2 * V4)
                @tensor dHrA12[a, s1, s2, c] := drhoL[a, a'] * conj(dA1[a', t1, b]) *
                    dA2[b, t2, c'] * drhoR[c', c] *
                    dH[s1, s2, t1, t2]
                @tensor hHrA12[a, s1, s2, c] := TensorKit.to_cpu(drhoL)[a, a'] * conj(TensorKit.to_cpu(dA1)[a', t1, b]) *
                    TensorKit.to_cpu(dA2)[b, t2, c'] * TensorKit.to_cpu(drhoR)[c', c] *
                    TensorKit.to_cpu(dH)[s1, s2, t1, t2]
                @test TensorKit.to_cpu(dHrA12) ≈ hHrA12
            end
        end=# # doesn't yet work because of AdjointTensor
        BraidingStyle(I) isa HasBraiding && @timedtestset "Index flipping: test flipping inverse" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V1' ← V1' ⊗ V1)
            for i in 1:4
                AMDGPU.@allowscalar begin
                    @test t ≈ flip(flip(t, i), i; inv = true)
                    @test t ≈ flip(flip(t, i; inv = true), i)
                end
            end
        end
        #=@timedtestset "Index flipping: test via explicit flip" begin
            t = AMDGPU.rand(ComplexF64, V1 ⊗ V1' ← V1' ⊗ V1)
            F1 = unitary(flip(V1), V1)

            AMDGPU.@allowscalar begin
                @tensor tf[a, b; c, d] := F1[a, a'] * t[a', b; c, d]
                @test flip(t, 1) ≈ tf
                @tensor tf[a, b; c, d] := conj(F1[b, b']) * t[a, b'; c, d]
                @test twist!(flip(t, 2), 2) ≈ tf
                @tensor tf[a, b; c, d] := F1[c, c'] * t[a, b; c', d]
                @test flip(t, 3) ≈ tf
                @tensor tf[a, b; c, d] := conj(F1[d, d']) * t[a, b; c, d']
                @test twist!(flip(t, 4), 4) ≈ tf
            end
        end
        @timedtestset "Index flipping: test via contraction" begin
            t1 = AMDGPU.rand(ComplexF64, V1 ⊗ V2 ⊗ V3 ← V4)
            t2 = AMDGPU.rand(ComplexF64, V2' ⊗ V5 ← V4' ⊗ V1)
            AMDGPU.@allowscalar begin
                @tensor ta[a, b] := t1[x, y, a, z] * t2[y, b, z, x]
                @tensor tb[a, b] := flip(t1, 1)[x, y, a, z] * flip(t2, 4)[y, b, z, x]
                @test ta ≈ tb
                @tensor tb[a, b] := flip(t1, (2, 4))[x, y, a, z] * flip(t2, (1, 3))[y, b, z, x]
                @test ta ≈ tb
                @tensor tb[a, b] := flip(t1, (1, 2, 4))[x, y, a, z] * flip(t2, (1, 3, 4))[y, b, z, x]
                @tensor tb[a, b] := flip(t1, (1, 3))[x, y, a, z] * flip(t2, (2, 4))[y, b, z, x]
                @test flip(ta, (1, 2)) ≈ tb
            end
        end=# # TODO =# # None of this works due to lack of HIPTensor support
        #=@timedtestset "Tensor product: test via norm preservation" begin
            for T in (Float32, ComplexF64)
                t1 = AMDGPU.rand(T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
                t2 = AMDGPU.rand(T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
                AMDGPU.@allowscalar begin
                    t = @constinferred (t1 ⊗ t2)
                end
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end
        symmetricbraiding && @timedtestset "Tensor product: test via conversion" begin
            for T in (Float32, ComplexF64)
                t1 = AMDGPU.rand(T, V2 ⊗ V3 ⊗ V1, V1)
                t2 = AMDGPU.rand(T, V2 ⊗ V1 ⊗ V3, V2)
                d1 = dim(codomain(t1))
                d2 = dim(codomain(t2))
                d3 = dim(domain(t1))
                d4 = dim(domain(t2))
                AMDGPU.@allowscalar begin
                    t = @constinferred (t1 ⊗ t2)
                    At = ad(t)
                    @test ad(t) ≈ ad(t1) ⊗ ad(t2)
                end
            end
        end=#
        #=symmetricbraiding && @timedtestset "Tensor product: test via tensor contraction" begin
            for T in (Float32, ComplexF64)
                t1 = AMDGPU.rand(T, V2 ⊗ V3 ⊗ V1)
                t2 = AMDGPU.rand(T, V2 ⊗ V1 ⊗ V3)
                AMDGPU.@allowscalar begin
                    t = @constinferred (t1 ⊗ t2)
                    @tensor t′[1, 2, 3, 4, 5, 6] := t1[1, 2, 3] * t2[4, 5, 6]
                    # @test t ≈ t′ # TODO broken for symmetry: Irrep[ℤ₃]
                end
            end
        end=# # broken due to no HIPTensor
    end
    TensorKit.empty_globalcaches!()
end

#=
@timedtestset "Deligne tensor product: test via conversion" begin
    Vlists1 = (Vtr,) # VSU₂)
    Vlists2 = (Vtr,) # Vℤ₂)
    @testset for Vlist1 in Vlists1, Vlist2 in Vlists2
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = AMDGPU.rand(T, V1 ⊗ V2, V3' ⊗ V4)
            t2 = AMDGPU.rand(T, W2, W1 ⊗ W1')
            AMDGPU.@allowscalar begin
                t = @constinferred (t1 ⊠ t2)
            end
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            AMDGPU.@allowscalar begin
                @test ad(t1) ⊠ ad(t2) ≈ ad(t1 ⊠ t2)
            end
        end
    end
end=#
