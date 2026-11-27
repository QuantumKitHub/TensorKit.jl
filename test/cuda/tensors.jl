using Adapt, CUDA, CUDA.cuRAND, cuTENSOR
using Test, TestExtras
using Random
using TensorKit, Combinatorics
using TensorOperations: TensorOperations as TO
ad = adapt(Array)
const CUDAExt = Base.get_extension(TensorKit, :TensorKitCUDAExt)
@assert !isnothing(CUDAExt)
const CuTensorMap = getglobal(CUDAExt, :CuTensorMap)
const GPUArraysExt = Base.get_extension(TensorKit, :TensorKitGPUArraysExt)

Random.seed!(123456)

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("CUDA Tensors with symmetry: $Istr")
    println("---------------------------------------")
    hasbraiding = BraidingStyle(I) isa HasBraiding
    symmetricbraiding = BraidingStyle(I) isa SymmetricBraiding
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            # test default pass-throughs
            for f in (CUDA.zeros, CUDA.ones, cuRAND.rand, cuRAND.randn)
                t = @constinferred f(W)
                @test scalartype(t) == Float64
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{Float64, spacetype(t), 5, 0, CuVector{Float64, CUDA.DeviceMemory}}
            end
            for f! in (cuRAND.rand!, cuRAND.randn!)
                t = @constinferred CUDA.zeros(W)
                f!(t)
                @test scalartype(t) == Float64
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{Float64, spacetype(t), 5, 0, CuVector{Float64, CUDA.DeviceMemory}}
            end
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64)
                t = @constinferred CUDA.zeros(T, W)
                @test scalartype(t) == T
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, CuVector{T, CUDA.DeviceMemory}}
            end
        end
        @timedtestset "Conversion to/from host" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                h_t = @constinferred rand(T, W)
                t1 = convert(CuTensorMap{T}, h_t)
                @test collect(t1.data) == h_t.data
                @test space(t1) == space(h_t)
                @test scalartype(t1) == T
                @test codomain(t1) == W
                @test space(t1) == (W ← one(W))
                @test domain(t1) == one(W)
                t2 = CuTensorMap(h_t)
                @test collect(t2.data) == h_t.data
                @test space(t2) == space(h_t)
                @test scalartype(t2) == T
                @test codomain(t2) == W
                @test space(t2) == (W ← one(W))
                @test domain(t2) == one(W)
            end
        end
        symmetricbraiding && @timedtestset "Full trace: test self-consistency" begin
            t = cuRAND.rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
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
                t1 = cuRAND.rand(T, W1, W1)
                t2 = cuRAND.rand(T, W2, W2)
                t = cuRAND.rand(T, W1, W2)
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
        @timedtestset "Multiplication and inverse: test via CPU" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = (V4 ⊗ V5)'
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                t1 = cuRAND.rand(T, W1, W1)
                t2 = cuRAND.rand(T, W2, W2)
                t = cuRAND.rand(T, W1, W2)
                ht1 = adapt(Array, t1)
                ht2 = adapt(Array, t2)
                ht = adapt(Array, t)
                @test adapt(Array, t1 * t) ≈ ht1 * ht
                @test adapt(Array, t1' * t) ≈ ht1' * ht
                @test adapt(Array, t2 * t') ≈ ht2 * ht'
                @test adapt(Array, t2' * t') ≈ ht2' * ht'

                @test adapt(Array, inv(t1)) ≈ inv(ht1)
                @test adapt(Array, pinv(t)) ≈ pinv(ht)

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
                t = project_hermitian!(cuRAND.randn(T, W, W))
                s = dim(W)
                #@test (@constinferred sqrt(t))^2 ≈ t
                #@test adapt(Array, sqrt(t)) ≈ sqrt(adapt(Array, t))

                expt = @constinferred exp(t)
                @test adapt(Array, expt) ≈ exp(adapt(Array, t))

                # log doesn't work on CUDA yet (scalar indexing)
                #@test exp(@constinferred log(project_hermitian!(expt))) ≈ expt
                #@test adapt(Array, log(project_hermitian!(expt))) ≈ log(adapt(Array, expt))

                #=@test (@constinferred cos(t))^2 + (@constinferred sin(t))^2 ≈
                          id(storagetype(t), W)
                    @test (@constinferred tan(t)) ≈ sin(t) / cos(t)
                    @test (@constinferred cot(t)) ≈ cos(t) / sin(t)
                    @test (@constinferred cosh(t))^2 - (@constinferred sinh(t))^2 ≈
                          id(storagetype(t), W)
                    @test (@constinferred tanh(t)) ≈ sinh(t) / cosh(t)
                    @test (@constinferred coth(t)) ≈ cosh(t) / sinh(t)=# # TODO in CUDA

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
                # TODO in CUDA
            end
        end
        # Sylvester not defined for CUDA
        # @timedtestset "Sylvester equation" begin
        #     for T in (Float32, ComplexF64)
        #         tA = cuRAND.rand(T, V1 ⊗ V3, V1 ⊗ V3)
        #         tB = cuRAND.rand(T, V2 ⊗ V4, V2 ⊗ V4)
        #         tA = 3 // 2 * leftorth(tA; alg=Polar())[1]
        #         tB = 1 // 5 * leftorth(tB; alg=Polar())[1]
        #         tC = cuRAND.rand(T, V1 ⊗ V3, V2 ⊗ V4)
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

@timedtestset "DiagonalTensorMap with domain $V: test via CPU" for V in (
        (ℂ^4)',
        Vect[Z2Irrep](0 => 2, 1 => 3),
        Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1),
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[SU2Irrep](0 => 2, 1 => 1)',
    )
    diagonal(T, W) = DiagonalTensorMap(CuVector{T}(rand(T, reduceddim(W))), W)

    @testset "normalize, eltype $T" for T in (Float64, ComplexF64)
        d = diagonal(T, V)
        dcpu = adapt(Array, d)
        for p in (1, 2, 3 * rand(Float64), Inf)
            dn = @constinferred normalize(d, p)
            @test dn isa typeof(d)
            @test adapt(Array, dn) ≈ normalize(dcpu, p)
            @test adapt(Array, normalize!(copy(d), p)) ≈ normalize(dcpu, p)
        end
    end

    @testset "scalar contraction" begin
        d = diagonal(Float64, V)
        s = adapt(CuVector{Float64}, randn(Float64, V ← V))
        dcpu, scpu = adapt(Array, d), adapt(Array, s)
        @test (@tensor s[1; 2] * d[2; 1]) ≈ (@tensor scpu[1; 2] * dcpu[2; 1])
    end
end

@timedtestset "Contraction outside `@tensor` rewinds a CUDA buffer" begin
    using .TestSetup: VRepU₁, VfRepSU₂
    for V in (VRepU₁, VfRepSU₂)
        V1, V2, V3, V4 = V
        A = cuRAND.randn(Float64, V1 ⊗ V2 ⊗ V3)
        B = cuRAND.randn(Float64, V3' ⊗ V2' ⊗ V4)
        @tensor Cref[d; a] := A[a b c] * B[c b d]
        buffer = TO.CUDABufferAllocator()
        C = TO.tensorcontract(
            A, ((1,), (2, 3)), false, B, ((2, 1), (3,)), false, ((2,), (1,)), true,
            TO.DefaultBackend(), buffer
        )
        @test isempty(buffer)
        @test C ≈ Cref
        @test buffer.max_offset > 0
    end
end
