using Test, TestExtras
using Adapt
using TensorKit
using TensorKit: PlanarTrivial, ℙ
using TensorKit: planaradd!, planartrace!, planarcontract!
using TensorOperations, CUDA, cuTENSOR

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

@testset "Braiding tensor" begin
    for V in (Vtr, VU₁, VfU₁, VfSU₂, Vfib)
        W = V[1] ⊗ V[2] ← V[2] ⊗ V[1]
        t1 = @constinferred BraidingTensor(W, CuVector)
        @test space(t1) == W
        @test codomain(t1) == codomain(W)
        @test domain(t1) == domain(W)
        @test scalartype(t1) == (isreal(sectortype(W)) ? Float64 : ComplexF64)
        @test storagetype(t1) == CuVector{scalartype(t1), CUDA.DeviceMemory}
        t2 = @constinferred BraidingTensor{ComplexF64, typeof(W), CuVector{ComplexF64, CUDA.DeviceMemory}}(W)
        @test scalartype(t2) == ComplexF64
        @test storagetype(t2) == CuVector{ComplexF64, CUDA.DeviceMemory}
        t3 = @testinferred adapt(storagetype(t2), t1)
        @test storagetype(t3) == storagetype(t2)
        @test t3 == t2

        W2 = reverse(codomain(W)) ← domain(W)
        @test_throws SpaceMismatch BraidingTensor(W2)

        @test adjoint(t1) isa BraidingTensor
        @test complex(t1) isa BraidingTensor
        @test scalartype(complex(t1)) <: Complex

        t3 = @inferred TensorMap(t2)
        @test storagetype(t3) = CuVector{ComplexF64, CUDA.DeviceMemory}
        t4 = braid(id(storagetype(t2), domain(t2)), ((2, 1), (3, 4)), (1, 2, 3, 4))
        @test t1 ≈ t4
        for (c, b) in blocks(t1)
            @test block(t1, c) ≈ b ≈ block(t3, c)
        end
        for (f1, f2) in fusiontrees(t1)
            @test t1[f1, f2] ≈ t3[f1, f2]
        end

        t5 = @inferred TensorMap(t2')
        @test storagetype(t5) = CuVector{ComplexF64, CUDA.DeviceMemory}
        t6 = braid(id(storagetype(t2), domain(t2')), ((2, 1), (3, 4)), (4, 3, 2, 1))
        @test t5 ≈ t6
        for (c, b) in blocks(t1')
            @test block(t1', c) ≈ b ≈ block(t5, c)
        end
        for (f1, f2) in fusiontrees(t1')
            @test t1'[f1, f2] ≈ t5[f1, f2]
        end
    end
end

@testset "planar methods" verbose = true begin
    @testset "planaradd" begin
        A = CUDA.randn(ℂ^2 ⊗ ℂ^3 ← ℂ^6 ⊗ ℂ^5 ⊗ ℂ^4)
        C = CUDA.randn((ℂ^5)' ⊗ (ℂ^6)' ← ℂ^4 ⊗ (ℂ^3)' ⊗ (ℂ^2)')
        A′ = force_planar(A)
        C′ = force_planar(C)
        p = ((4, 3), (5, 2, 1))

        @test force_planar(tensoradd!(C, A, p, false, true, true)) ≈
            planaradd!(C′, A′, p, true, true)
    end

    @testset "planartrace" begin
        A = CUDA.randn(ℂ^2 ⊗ ℂ^3 ← ℂ^2 ⊗ ℂ^5 ⊗ ℂ^4)
        C = CUDA.randn((ℂ^5)' ⊗ ℂ^3 ← ℂ^4)
        A′ = force_planar(A)
        C′ = force_planar(C)
        p = ((4, 2), (5,))
        q = ((1,), (3,))

        @test force_planar(tensortrace!(C, A, p, q, false, true, true)) ≈
            planartrace!(C′, A′, p, q, true, true)
    end

    @testset "planarcontract" begin
        A = CUDA.randn(ℂ^2 ⊗ ℂ^3 ← ℂ^2 ⊗ ℂ^5 ⊗ ℂ^4)
        B = CUDA.randn(ℂ^2 ⊗ ℂ^4 ← ℂ^4 ⊗ ℂ^3)
        C = CUDA.randn((ℂ^5)' ⊗ (ℂ^2)' ⊗ ℂ^2 ← (ℂ^2)' ⊗ ℂ^4)

        A′ = force_planar(A)
        B′ = force_planar(B)
        C′ = force_planar(C)

        pA = ((1, 3, 4), (5, 2))
        pB = ((2, 4), (1, 3))
        pAB = ((3, 2, 1), (4, 5))

        @test force_planar(tensorcontract!(C, A, pA, false, B, pB, false, pAB, true, true)) ≈
            planarcontract!(C′, A′, pA, B′, pB, pAB, true, true)
    end
end

@testset "@planar" verbose = true begin
    T = ComplexF64

    @testset "contractcheck" begin
        V = ℂ^2
        A = CUDA.rand(T, V ⊗ V ← V)
        B = CUDA.rand(T, V ⊗ V ← V')
        @tensor C1[i j; k l] := A[i j; m] * B[k l; m]
        @tensor contractcheck = true C2[i j; k l] := A[i j; m] * B[k l; m]
        @test C1 ≈ C2
        B2 = CUDA.rand(T, V ⊗ V ← V) # wrong duality for third space
        @test_throws SpaceMismatch("incompatible spaces for m: $V ≠ $(V')") begin
            @tensor contractcheck = true C3[i j; k l] := A[i j; m] * B2[k l; m]
        end

        A = CUDA.rand(T, V ← V ⊗ V)
        B = CUDA.rand(T, V ⊗ V ← V)
        @planar C1[i; j] := A[i; k l] * τ[k l; m n] * B[m n; j]
        @planar contractcheck = true C2[i; j] := A[i; k l] * τ[k l; m n] * B[m n; j]
        @test C1 ≈ C2
        @test_throws SpaceMismatch("incompatible spaces for m: $V ≠ $(V')") begin
            @planar contractcheck = true C3[i; j] := A[i; k l] * τ[k l; m n] * B[n j; m]
        end
    end

    @testset "MPS networks" begin
        P = ℂ^2
        Vmps = ℂ^12
        Vmpo = ℂ^4

        # ∂AC
        # -------
        x = CUDA.randn(T, Vmps ⊗ P ← Vmps)
        O = CUDA.randn(T, Vmpo ⊗ P ← P ⊗ Vmpo)
        GL = CUDA.randn(T, Vmps ⊗ Vmpo' ← Vmps)
        GR = CUDA.randn(T, Vmps ⊗ Vmpo ← Vmps)

        x′ = force_planar(x)
        O′ = force_planar(O)
        GL′ = force_planar(GL)
        GR′ = force_planar(GR)

        for alloc in
            (TensorOperations.DefaultAllocator(), TensorOperations.CUDAAllocator())
            @tensor allocator = alloc y[-1 -2; -3] := GL[-1 2; 1] * x[1 3; 4] *
                O[2 -2; 3 5] * GR[4 5; -3]
            @planar allocator = alloc y′[-1 -2; -3] := GL′[-1 2; 1] * x′[1 3; 4] *
                O′[2 -2; 3 5] * GR′[4 5; -3]
            @test force_planar(y) ≈ y′
        end

        # ∂AC2
        # -------
        x2 = CUDA.randn(T, Vmps ⊗ P ← Vmps ⊗ P')
        x2′ = force_planar(x2)
        @tensor contractcheck = true y2[-1 -2; -3 -4] := GL[-1 7; 6] * x2[6 5; 1 3] *
            O[7 -2; 5 4] * O[4 -4; 3 2] *
            GR[1 2; -3]
        @planar y2′[-1 -2; -3 -4] := GL′[-1 7; 6] * x2′[6 5; 1 3] * O′[7 -2; 5 4] *
            O′[4 -4; 3 2] * GR′[1 2; -3]
        @test force_planar(y2) ≈ y2′

        # transfer matrix
        # ----------------
        v = CUDA.randn(T, Vmps ← Vmps)
        v′ = force_planar(v)
        @tensor ρ[-1; -2] := x[-1 2; 1] * conj(x[-2 2; 3]) * v[1; 3]
        @planar ρ′[-1; -2] := x′[-1 2; 1] * conj(x′[-2 2; 3]) * v′[1; 3]
        @test force_planar(ρ) ≈ ρ′

        @tensor ρ2[-1 -2; -3] := GL[1 -2; 3] * x[3 2; -3] * conj(x[1 2; -1])
        @plansor ρ3[-1 -2; -3] := GL[1 2; 4] * x[4 5; -3] * τ[2 3; 5 -2] * conj(x[1 3; -1])
        @planar ρ2′[-1 -2; -3] := GL′[1 2; 4] * x′[4 5; -3] * τ[2 3; 5 -2] *
            conj(x′[1 3; -1])
        @test force_planar(ρ2) ≈ ρ2′
        @test ρ2 ≈ ρ3

        # Periodic boundary conditions
        # ----------------------------
        f1 = isomorphism(storagetype(O), fuse(Vmpo^3), Vmpo ⊗ Vmpo' ⊗ Vmpo)
        f2 = isomorphism(storagetype(O), fuse(Vmpo^3), Vmpo ⊗ Vmpo' ⊗ Vmpo)
        f1′ = force_planar(f1)
        f2′ = force_planar(f2)
        @tensor O_periodic1[-1 -2; -3 -4] := O[1 -2; -3 2] * f1[-1; 1 3 4] *
            conj(f2[-4; 2 3 4])
        @plansor O_periodic2[-1 -2; -3 -4] := O[1 2; -3 6] * f1[-1; 1 3 5] *
            conj(f2[-4; 6 7 8]) * τ[2 3; 7 4] *
            τ[4 5; 8 -2]
        @planar O_periodic′[-1 -2; -3 -4] := O′[1 2; -3 6] * f1′[-1; 1 3 5] *
            conj(f2′[-4; 6 7 8]) * τ[2 3; 7 4] *
            τ[4 5; 8 -2]
        @test O_periodic1 ≈ O_periodic2
        @test force_planar(O_periodic1) ≈ O_periodic′
    end

    @testset "MERA networks" begin
        Vmera = ℂ^2

        u = CUDA.randn(T, Vmera ⊗ Vmera ← Vmera ⊗ Vmera)
        w = CUDA.randn(T, Vmera ⊗ Vmera ← Vmera)
        ρ = CUDA.randn(T, Vmera ⊗ Vmera ⊗ Vmera ← Vmera ⊗ Vmera ⊗ Vmera)
        h = CUDA.randn(T, Vmera ⊗ Vmera ⊗ Vmera ← Vmera ⊗ Vmera ⊗ Vmera)

        u′ = force_planar(u)
        w′ = force_planar(w)
        ρ′ = force_planar(ρ)
        h′ = force_planar(h)

        for alloc in
            (TensorOperations.DefaultAllocator(), TensorOperations.CUDAAllocator())
            @tensor allocator = alloc begin
                C = (
                    (
                        (
                            (
                                (
                                    ((h[9 3 4; 5 1 2] * u[1 2; 7 12]) * conj(u[3 4; 11 13])) *
                                        (u[8 5; 15 6] * w[6 7; 19])
                                ) *
                                    (conj(u[8 9; 17 10]) * conj(w[10 11; 22]))
                            ) *
                                ((w[12 14; 20] * conj(w[13 14; 23])) * ρ[18 19 20; 21 22 23])
                        ) *
                            w[16 15; 18]
                    ) * conj(w[16 17; 21])
                )
            end
            @planar allocator = alloc begin
                C′ = (
                    (
                        (
                            (
                                (
                                    ((h′[9 3 4; 5 1 2] * u′[1 2; 7 12]) * conj(u′[3 4; 11 13])) *
                                        (u′[8 5; 15 6] * w′[6 7; 19])
                                ) *
                                    (conj(u′[8 9; 17 10]) * conj(w′[10 11; 22]))
                            ) *
                                ((w′[12 14; 20] * conj(w′[13 14; 23])) * ρ′[18 19 20; 21 22 23])
                        ) *
                            w′[16 15; 18]
                    ) * conj(w′[16 17; 21])
                )
            end
            @test C ≈ C′
        end
    end

    @testset "Issue 93" begin
        T = Float64
        V1 = ℂ^2
        V2 = ℂ^3
        t1 = CUDA.rand(T, V1 ← V2)
        t2 = CUDA.rand(T, V2 ← V1)

        tr1 = @planar opt = true t1[a; b] * t2[b; a] / 2
        tr2 = @planar opt = true t1[d; a] * t2[b; c] * 1 / 2 * τ[c b; a d]
        tr3 = @planar opt = true t1[d; a] * t2[b; c] * τ[a c; d b] / 2
        tr4 = @planar opt = true t1[f; a] * 1 / 2 * t2[c; d] * τ[d b; c e] * τ[e b; a f]
        tr5 = @planar opt = true t1[f; a] * t2[c; d] / 2 * τ[d b; c e] * τ[a e; f b]
        tr6 = @planar opt = true t1[f; a] * t2[c; d] * τ[c d; e b] / 2 * τ[e b; a f]
        tr7 = @planar opt = true t1[f; a] * t2[c; d] * (τ[c d; e b] * τ[a e; f b] / 2)

        @test tr1 ≈ tr2 ≈ tr3 ≈ tr4 ≈ tr5 ≈ tr6 ≈ tr7

        tr1 = @plansor opt = true t1[a; b] * t2[b; a] / 2
        tr2 = @plansor opt = true t1[d; a] * t2[b; c] * 1 / 2 * τ[c b; a d]
        tr3 = @plansor opt = true t1[d; a] * t2[b; c] * τ[a c; d b] / 2
        tr4 = @plansor opt = true t1[f; a] * 1 / 2 * t2[c; d] * τ[d b; c e] * τ[e b; a f]
        tr5 = @plansor opt = true t1[f; a] * t2[c; d] / 2 * τ[d b; c e] * τ[a e; f b]
        tr6 = @plansor opt = true t1[f; a] * t2[c; d] * τ[c d; e b] / 2 * τ[e b; a f]
        tr7 = @plansor opt = true t1[f; a] * t2[c; d] * (τ[c d; e b] * τ[a e; f b] / 2)

        @test tr1 ≈ tr2 ≈ tr3 ≈ tr4 ≈ tr5 ≈ tr6 ≈ tr7
    end
    @testset "Issue 262" begin
        V = ℂ^2
        A = CUDA.rand(T, V ← V)
        B = CUDA.rand(T, V ← V')
        C = CUDA.rand(T, V' ← V)
        @planar D1[i; j] := A[i; j] + B[i; k] * C[k; j]
        @planar D2[i; j] := B[i; k] * C[k; j] + A[i; j]
        @test D1 ≈ D2
    end
end
