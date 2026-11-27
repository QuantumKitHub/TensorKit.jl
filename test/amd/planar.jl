# Most planar functionality is tested in a vendor-independent way in `jlarrays/planar.jl`.
# The tests here only check vendor-specific storage types and run the device kernels from
# `TensorKitGPUArraysExt` on actual hardware.
using Test, TestExtras
using Adapt
using TensorOperations, AMDGPU
using Random
using TensorKit
using TensorKit: type_repr
using TensorKit: planartrace!

const AMDGPUExt = Base.get_extension(TensorKit, :TensorKitAMDGPUExt)
@assert !isnothing(AMDGPUExt)

Random.seed!(123456)

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    BraidingStyle(I) isa NoBraiding && continue
    @timedtestset "Braiding tensor + AMDGPU with symmetry: $Istr" verbose = true begin
        W = V[1] ⊗ V[2] ← V[2] ⊗ V[1]
        T = isreal(sectortype(W)) ? Float64 : ComplexF64
        t1 = @constinferred BraidingTensor{T, spacetype(V[2]), ROCVector{T, AMDGPU.Mem.HIPBuffer}}(W)
        @test scalartype(t1) == T
        @test storagetype(t1) == ROCVector{T, AMDGPU.Mem.HIPBuffer}
        t2 = @constinferred BraidingTensor{ComplexF64, spacetype(V[2]), ROCVector{ComplexF64, AMDGPU.Mem.HIPBuffer}}(W)
        @test storagetype(t2) == ROCVector{ComplexF64, AMDGPU.Mem.HIPBuffer}

        # materializing runs the braiding subblock fill kernel, braiding the identity runs the permute kernels
        t3 = @inferred TensorMap(t2)
        @test storagetype(t3) == ROCVector{ComplexF64, AMDGPU.Mem.HIPBuffer}
        t4 = braid(adapt(ROCArray, id(scalartype(t2), domain(t2))), ((2, 1), (3, 4)), (1, 2, 3, 4))
        @test t1 ≈ t4
        @test t3 ≈ t4
    end
end

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    @timedtestset "planartrace + AMDGPU with symmetry: $Istr" begin
        # (space, p, q) for a trace of neighbouring legs and a nested trace
        cases = (
            (V[1] ⊗ V[2] ⊗ V[3] ← V[4] ⊗ V[3], ((1, 2), (4,)), ((3,), (5,))),
            (V[1] ⊗ V[2] ⊗ V[3] ⊗ V[4] ← V[3] ⊗ V[4], ((1,), (2,)), ((3, 4), (5, 6))),
        )
        for (W, p, q) in cases, (α, β) in ((0.7, 0.3), (1.0, 0.0))
            A = randn(ComplexF64, W)
            C = randn(ComplexF64, TensorKit.select(W, p))
            Cgpu = planartrace!(adapt(ROCArray, C), adapt(ROCArray, A), p, q, α, β)
            @test adapt(Vector{ComplexF64}, Cgpu) ≈ planartrace!(C, A, p, q, α, β)
        end
    end
end

@testset "planartrace with large traces: $I" for (I, V, Vr) in (
        (FermionParity, Vect[FermionParity](0 => 40, 1 => 40), Vect[FermionParity](0 => 2, 1 => 3)),
        (FibonacciAnyon, Vect[FibonacciAnyon](:I => 30, :τ => 30), Vect[FibonacciAnyon](:I => 2, :τ => 1)),
    )
    # (space, p, q) for a full reduction and for a trace with few remaining entries, which are
    # both split into several chunks of traced entries
    cases = (
        (V ⊗ V ← V ⊗ V, ((), ()), ((1, 2), (3, 4))),
        (Vr ⊗ V ⊗ V ← V ⊗ V, ((1,), ()), ((2, 3), (4, 5))),
    )
    for (W, p, q) in cases, (α, β) in ((0.7, 0.3), (1.0, 0.0))
        A = randn(ComplexF64, W)
        C = randn(ComplexF64, TensorKit.select(W, p))
        Cgpu = planartrace!(adapt(ROCArray, C), adapt(ROCArray, A), p, q, α, β)
        @test adapt(Vector{ComplexF64}, Cgpu) ≈ planartrace!(C, A, p, q, α, β)
    end
end
