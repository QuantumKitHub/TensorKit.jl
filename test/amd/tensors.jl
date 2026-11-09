using Adapt, AMDGPU
using Test, TestExtras
using Random
using TensorKit, Combinatorics
using TensorOperations: TensorOperations as TO
ad = adapt(Array)
const AMDGPUExt = Base.get_extension(TensorKit, :TensorKitAMDGPUExt)
@assert !isnothing(AMDGPUExt)
const ROCTensorMap = getglobal(AMDGPUExt, :ROCTensorMap)
const rocrand = getglobal(AMDGPUExt, :rocrand)
const rocrandn = getglobal(AMDGPUExt, :rocrandn)
const rocrand! = getglobal(AMDGPUExt, :rocrand!)
using AMDGPU: rand as rocrand, rand! as rocrand!, randn as rocrandn, randn! as rocrandn!

Random.seed!(123456)

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("AMDGPU Tensors with symmetry: $Istr")
    println("---------------------------------------")
    hasbraiding = BraidingStyle(I) isa HasBraiding
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
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64)
                t = @constinferred AMDGPU.zeros(T, W)
                AMDGPU.@allowscalar begin
                    @test @constinferred(hash(t)) == hash(deepcopy(t))
                end
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, ROCVector{T, AMDGPU.Mem.HIPBuffer}}
                # blocks
                bs = @constinferred blocks(t)
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t, first(blocksectors(t)))
                @test b1 == b2
                @test eltype(bs) === Pair{typeof(c), typeof(b1)}
                @test typeof(b1) === TensorKit.blocktype(t)
                @test typeof(c) === sectortype(t)
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
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)'
            for T in (Int, Float32, ComplexF64)
                t = @constinferred AMDGPU.rand(T, W)
                d = convert(Dict, t)
                @test convert(Dict, adapt(Array, t)) == d
            end
        end
        #=@timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            t = AMDGPU.randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            # TODO find a way to use AMDGPU here
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end=#
        symmetricbraiding && @timedtestset "Full trace: test self-consistency" begin
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
        @timedtestset "Multiplication and inverse: test compatibility" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = (V4 ⊗ V5)'
            for T in (Float64, ComplexF64)
                t1 = AMDGPU.rand(T, W1, W1)
                t2 = AMDGPU.rand(T, W2, W2)
                t = AMDGPU.rand(T, W1, W2)
                @test t1 * (t1 \ t) ≈ t
                @test (t / t2) * t2 ≈ t
                @test t1 \ one(t1) ≈ inv(t1)
                @test one(t1) / t1 ≈ pinv(t1)
                tp = pinv(t) * t
                @test tp ≈ tp * tp
                @test_throws SpaceMismatch inv(t)
                @test_throws SpaceMismatch t2 \ t
                @test_throws SpaceMismatch t / t1
            end
        end
        @timedtestset "Multiplication and inverse: test via CPU" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = (V4 ⊗ V5)'
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                t1 = AMDGPU.rand(T, W1, W1)
                t2 = AMDGPU.rand(T, W2, W2)
                t = AMDGPU.rand(T, W1, W2)
                ht1 = adapt(Array, t1)
                ht2 = adapt(Array, t2)
                ht = adapt(Array, t)
                @test adapt(Array, t1 * t) ≈ ht1 * ht
                @test adapt(Array, t1' * t) ≈ ht1' * ht
                @test adapt(Array, t2 * t') ≈ ht2 * ht'
                @test adapt(Array, t2' * t') ≈ ht2' * ht'

                #@test adapt(Array, inv(t1)) ≈ inv(ht1) # needs getrf
                #@test adapt(Array, pinv(t)) ≈ pinv(ht)

                if T == Float32 || T == ComplexF32
                    continue
                end

                @test adapt(Array, t1 \ t) ≈ ht1 \ ht
                @test adapt(Array, t1' \ t) ≈ ht1' \ ht
                @test adapt(Array, t2 \ t') ≈ ht2 \ ht'
                @test adapt(Array, t2' \ t') ≈ ht2' \ ht'

                @test adapt(Array, t2 / t) ≈ ht2 / ht
                @test adapt(Array, t2' / t) ≈ ht2' / ht
                @test adapt(Array, t1 / t') ≈ ht1 / ht'
                @test adapt(Array, t1' / t') ≈ ht1' / ht'
            end
        end
        symmetricbraiding && @timedtestset "Tensor functions" begin
            W = V1 ⊗ V2
            for T in (Float64, ComplexF64)
                #=t = project_hermitian!(AMDGPU.randn(T, W, W))
                s = dim(W)
                @test (@constinferred sqrt(t))^2 ≈ t
                @test adapt(Array, sqrt(t)) ≈ sqrt(adapt(Array, t))
                expt = @constinferred exp(t)
                @test adapt(Array, expt) ≈ exp(adapt(Array, t))
                @test exp(@constinferred log(project_hermitian!(expt))) ≈ expt
                @test adapt(Array, log(project_hermitian!(expt))) ≈ log(adapt(Array, expt))

                @test (@constinferred cos(t))^2 + (@constinferred sin(t))^2 ≈
                      id(storagetype(t), W)
                @test (@constinferred tan(t)) ≈ sin(t) / cos(t)
                @test (@constinferred cot(t)) ≈ cos(t) / sin(t)
                @test (@constinferred cosh(t))^2 - (@constinferred sinh(t))^2 ≈
                      id(storagetype(t), W)
                @test (@constinferred tanh(t)) ≈ sinh(t) / cosh(t)
                @test (@constinferred coth(t)) ≈ cosh(t) / sinh(t)=# # TODO in AMDGPU

                #=t1 = sin(t)
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
                @test coth(@constinferred acoth(t8)) ≈ t8=#
                # TODO in AMDGPU
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
        #
        # TODO
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "DiagonalTensorMap index manipulations with domain $V: test via CPU" for V in (
        (ℂ^4)',
        Vect[Z2Irrep](0 => 2, 1 => 3),
        Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1),
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[SU2Irrep](0 => 2, 1 => 1)',
    )
    function test_tensoradd(dst, d, p, conjd, α, β)
        cpu = TO.tensoradd!(adapt(Array, dst), adapt(Array, d), p, conjd, α, β, TO.DefaultBackend(), TO.DefaultAllocator())
        gpu = TO.tensoradd!(copy(dst), d, p, conjd, α, β, TO.DefaultBackend(), TO.DefaultAllocator())
        @test gpu isa typeof(dst)
        @test adapt(Array, gpu) ≈ cpu
        return nothing
    end
    diagonal(T, W) = DiagonalTensorMap(ROCVector{T}(rand(T, reduceddim(W))), W)

    @testset "eltype $T" for T in (Float64, ComplexF64)
        d = diagonal(T, V)
        dcpu = adapt(Array, d)
        scalars = ((TensorKit.One(), TensorKit.Zero()), (T(2), T(3)))

        for p in (((), (1, 2)), ((1, 2), ()), ((2, 1), ()), ((), (2, 1)), ((1,), (2,)), ((2,), (1,))), conjd in (false, true)
            W = TO.tensoradd_structure(dcpu, p, conjd)
            dst = adapt(ROCVector{T}, randn(T, W))
            for (α, β) in scalars
                test_tensoradd(dst, d, p, conjd, α, β)
            end
        end
        for p in (((1,), (2,)), ((2,), (1,))), conjd in (false, true)
            W = domain(TO.tensoradd_structure(dcpu, p, conjd))[1]
            for (α, β) in scalars
                test_tensoradd(diagonal(T, W), d, p, conjd, α, β)
            end
        end
        for p in (((), (1, 2)), ((1, 2), ()), ((2, 1), ()), ((2,), (1,)))
            @test adapt(Array, permute(d, p)) ≈ permute(dcpu, p)
            @test adapt(Array, permute(d', p)) ≈ permute(dcpu', p)
        end
        for k in (0, 2)
            @test adapt(Array, repartition(d, k)) ≈ repartition(dcpu, k)
            @test adapt(Array, repartition(d', k)) ≈ repartition(dcpu', k)
        end
        for levels in ((1, 2), (2, 1))
            @test adapt(Array, braid(d, ((1, 2), ()), levels)) ≈ braid(dcpu, ((1, 2), ()), levels)
        end
        for inds in ((1,), (2,), (1, 2)), inv in (false, true)
            ref = twist(dcpu, inds; inv)
            @test twist(d, inds; inv) isa typeof(d)
            @test adapt(Array, twist(d, inds; inv)) ≈ ref
            @test adapt(Array, twist(d', inds; inv)) ≈ twist(dcpu', inds; inv)
            @test adapt(Array, twist!(copy(d), inds; inv)) ≈ ref
            dst = adapt(ROCVector{T}, randn(T, space(dcpu)))
            @test adapt(Array, twist!(copy(dst), d, inds, T(2), T(3); inv)) ≈ 3 * adapt(Array, dst) + 2 * ref
            ddst = diagonal(T, V)
            @test adapt(Array, twist!(copy(ddst), d, inds, T(2), T(3); inv)) ≈ 3 * adapt(Array, ddst) + 2 * ref
        end
    end

    @testset "real source, complex destination" begin
        d = diagonal(Float64, V)
        p = ((1,), (2,))
        W = domain(TO.tensoradd_structure(adapt(Array, d), p, true))[1]
        test_tensoradd(diagonal(ComplexF64, W), d, p, true, TensorKit.One(), TensorKit.Zero())
        test_tensoradd(diagonal(ComplexF64, W), d, p, true, ComplexF64(2), ComplexF64(3))
    end

    @testset "contractions" begin
        d = diagonal(Float64, V)
        t = adapt(ROCVector{ComplexF64}, randn(ComplexF64, V ⊗ V ← V))
        s = adapt(ROCVector{Float64}, randn(Float64, V ← V))
        dcpu, tcpu, scpu = adapt(Array, d), adapt(Array, t), adapt(Array, s)

        @tensor gpu[-1 -2; -3] := t[-1 -2; 1] * d[1; -3]
        @tensor cpu[-1 -2; -3] := tcpu[-1 -2; 1] * dcpu[1; -3]
        @test adapt(Array, gpu) ≈ cpu

        @tensor gpu[-1 -2; -3] := d[-1; 1] * t[1 -2; -3]
        @tensor cpu[-1 -2; -3] := dcpu[-1; 1] * tcpu[1 -2; -3]
        @test adapt(Array, gpu) ≈ cpu

        @test (@tensor s[1; 2] * d[2; 1]) ≈ (@tensor scpu[1; 2] * dcpu[2; 1])
    end
end
