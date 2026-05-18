using TensorKit
using TensorKit: type_repr, sectortype
using CUDA, CUDA.cuRAND, cuTENSOR, AMDGPU
using TestExtras

spacelist = default_spacelist(fast_tests)

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("Tensor constructions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensor constructions with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (fast_tests ? (Float64, ComplexF64) : (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat))
                ATs = [Vector{T}]
                if T ∈ TestSuite.BLASFloats
                    CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                    AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                end
                for AT in ATs
                    t = @constinferred zero_f(AT)(T, W)
                    TestSuite.basic_tensor_properties(t, W, T, AT)
                    t = @constinferred zeros(AT, W)
                    TestSuite.basic_tensor_properties(t, W, T, AT)
                    if !isempty(blocksectors(t)) # multifusion space ending on module gives empty data
                        TestSuite.basic_blocks_properties(t, W)
                    end
                end
            end
        end
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)'
            for T in (Int, Float32, ComplexF64)
                ATs = [Vector{T}]
                CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                for AT in ATs
                    t = @constinferred rand_f(AT)(T, W)
                    TestSuite.tensor_dict_conversion(t)
                end
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
                        ATs = [Vector{T}]
                        if T ∈ TestSuite.BLASFloats
                            CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                            AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                        end
                        for AT in ATs
                            if T == Int
                                t = TensorKit.TensorMapWithStorage{T, AT}(undef, W)
                                for (_, b) in blocks(t)
                                    rand!(b, -20:20)
                                end
                            else
                                t = @constinferred randn_f(AT)(T, W)
                            end
                            TestSuite.tensor_array_conversion(t, W)
                        end
                    end
                end
                for T in (Int, Float32, ComplexF64)
                    ATs = [Vector{T}]
                    CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                    AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                    for AT in ATs
                        t = randn_f(AT)(T, V1 ⊗ V2 ← zerospace(V1))
                        TestSuite.empty_tensor_array_conversion(t, AT)
                    end
                end
            end
        end
        if hasfusiontensor(I)
            @timedtestset "Real and imaginary parts" begin
                for T in (Float64, ComplexF64, ComplexF32)
                    ATs = [Vector{T}]
                    CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                    AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                    for AT in ATs
                        W = V1 ⊗ V2
                        t = @constinferred randn_f(AT)(T, W, W)
                        TestSuite.real_and_imaginary_parts(t)
                    end
                end
            end
        end
        @timedtestset "Tensor conversion" begin
            for T in (Float64, ComplexF64, ComplexF32)
                ATs = [Vector{T}]
                CUDA.functional() && push!(ATs, CuVector{T, CUDA.DeviceMemory})
                AMDGPU.functional() && push!(ATs, ROCVector{T, AMDGPU.Mem.HIPBuffer})
                for AT in ATs
                    W = V1 ⊗ V2
                    t = @constinferred randn_f(AT)(W ← W)
                    TestSuite.tensor_conversion(t)
                end
            end
        end
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "show tensors" begin
    for V in (ℂ^2, Z2Space(0 => 2, 1 => 2), SU2Space(0 => 2, 1 => 2))
        t1 = ones(Float32, V ⊗ V, V)
        t2 = randn(ComplexF64, V ⊗ V ⊗ V)
        t3 = randn(Float64, zero(V), zero(V))
        # test unlimited output
        for t in (t1, t2, t1', t2', t3)
            output = IOBuffer()
            summary(output, t)
            print(output, ":\n codomain: ")
            show(output, MIME("text/plain"), codomain(t))
            print(output, "\n domain: ")
            show(output, MIME("text/plain"), domain(t))
            print(output, "\n blocks: \n")
            first = true
            for (c, b) in blocks(t)
                first || print(output, "\n\n")
                print(output, " * ")
                show(output, MIME("text/plain"), c)
                print(output, " => ")
                show(output, MIME("text/plain"), b)
                first = false
            end
            outputstr = String(take!(output))
            @test outputstr == sprint(show, MIME("text/plain"), t)
        end

        # test limited output with a single block
        t = randn(Float64, V ⊗ V, V)' # we know there is a single space in the codomain, so that blocks have 2 rows
        output = IOBuffer()
        summary(output, t)
        print(output, ":\n codomain: ")
        show(output, MIME("text/plain"), codomain(t))
        print(output, "\n domain: ")
        show(output, MIME("text/plain"), domain(t))
        print(output, "\n blocks: \n")
        c = unit(sectortype(t))
        b = block(t, c)
        print(output, " * ")
        show(output, MIME("text/plain"), c)
        print(output, " => ")
        show(output, MIME("text/plain"), b)
        if length(blocks(t)) > 1
            print(output, "\n\n *   …   [output of 1 more block(s) truncated]")
        end
        outputstr = String(take!(output))
        @test outputstr == sprint(show, MIME("text/plain"), t; context = (:limit => true, :displaysize => (12, 100)))
    end
end
