using Test, TestExtras
using TensorKit
using TensorKit: type_repr
using LinearAlgebra: LinearAlgebra
using CUDA, AMDGPU

spacelist = default_spacelist(fast_tests)

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

rand_fs = Any[rand_f(Vector)]
CUDA.functional() && push!(rand_fs, rand_f(CuVector))
AMDGPU.functional() && push!(rand_fs, rand_f(ROCVector))

randn_fs = Any[randn_f(Vector)]
CUDA.functional() && push!(randn_fs, randn_f(CuVector))
AMDGPU.functional() && push!(randn_fs, randn_f(ROCVector))

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    symmetricbraiding = BraidingStyle(I) isa SymmetricBraiding
    println("---------------------------------------")
    println("Tensor linear algebra with symmetry: $Istr")
    println("---------------------------------------")

    @timedtestset "Tensor linear algebra with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)'
            for T in (Float32, ComplexF64), rand_f in rand_fs
                TestSuite.basic_linear_algebra(rand_f, T, W)
                TestSuite.tensor_norm(rand_f, T, W)
                TestSuite.tensor_dot(rand_f, T, W)
            end
            for T in (Float32, ComplexF64)
                ATs = [Vector{T}]
                if T ∈ TestSuite.BLASFloats
                    CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                    AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                end
                for AT in ATs
                    TestSuite.isomorphism_test(T, AT, V1, V2)
                    TestSuite.isometry_test(T, AT, V1)
                end
            end
        end
        if hasfusiontensor(I)
            @timedtestset "Basic linear algebra: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)'
                for T in (Float32, ComplexF64), rand_f in rand_fs
                    TestSuite.linalg_via_conversion(rand_f, T, W)
                end
            end
        end
        @timedtestset "Multiplication of isometries: test properties" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = (V4 ⊗ V5)'
            for T in (Float64, ComplexF64)
                ATs = [Vector{T}]
                if T ∈ TestSuite.BLASFloats
                    CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                    AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                end
                for AT in ATs
                    TestSuite.multiplying_isometries(AT, W1, W2)
                end
            end
        end
        @timedtestset "Multiplication and inverse: test compatibility" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = (V4 ⊗ V5)'
            for T in (Float64, ComplexF64), rand_f in rand_fs
                TestSuite.tensor_multiplication_and_inverse(rand_f, T, W1, W2)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Multiplication and inverse: test via conversion" begin
                W1 = V1 ⊗ V2 ⊗ V3
                W2 = (V4 ⊗ V5)'
                for T in (Float32, Float64, ComplexF32, ComplexF64), rand_f in rand_fs
                    TestSuite.tensor_multiplication_and_inverse_conversion(rand_f, T, W1, W2)
                end
            end
        end
        @timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)'
            for T in (ComplexF64,), randn_f in randn_fs
                TestSuite.diag_diagm(randn_f, T, W)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor functions" begin
                # TODO need better tensor function support for AMDGPU
                tf_randn_fs = Any[randn_f(Vector)]
                CUDA.functional() && push!(tf_randn_fs, randn_f(CuVector))
                for T in (Float64, ComplexF64), randn_f in tf_randn_fs
                    TestSuite.tensor_functions(randn_f, T, V1, V2)
                end
            end
        end
        @timedtestset "Sylvester equation" begin
            # TODO schur not defined for GPU arrays
            for T in (Float32, ComplexF64), rand_f in [rand] #rand_fs
                TestSuite.sylvester_test(rand_f, T, V)
            end
        end
    end
    TensorKit.empty_globalcaches!()
end
