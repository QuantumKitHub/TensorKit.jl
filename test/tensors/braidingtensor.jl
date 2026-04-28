using Test, TestExtras
using TensorKit
using TensorKit: type_repr

spacelist = default_spacelist(fast_tests)

for V_tuple in spacelist
    I = sectortype(first(V_tuple))
    Istr = type_repr(I)
    BraidingStyle(I) isa NoBraiding && continue
    V = first(V_tuple)
    T = ComplexF64
    t = randn(T, V ⊗ V' ⊗ V' ⊗ V ← V ⊗ V')
    println("---------------------------------------")
    println("BraidingTensor planar contractions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "BraidingTensor planar contractions with symmetry: $Istr" verbose = true begin
        @timedtestset "τ as left factor, all legs contracted" begin
            # BraidingTensor(V, V') on leading codomain indices
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -2 -1] * t[1 2 -3 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-2 2; -1 1] * t[1 2 -3 -4; -5 -6]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V', V') on inner codomain indices
            ττ = TensorMap(BraidingTensor(V', V'))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V', V) on trailing codomain indices
            ττ = TensorMap(BraidingTensor(V', V))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -4 -3] * t[-1 -2 1 2; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-4 2; -3 1] * t[-1 -2 1 2; -5 -6]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4
        end

        @timedtestset "τ as left factor, mixed open legs" begin
            # BraidingTensor(V', V) with mixed index pattern
            ττ = TensorMap(BraidingTensor(V', V))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V, V') with mixed index pattern (inverse of previous)
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V, V) with mixed index pattern
            ττ = TensorMap(BraidingTensor(V, V))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4
        end

        @timedtestset "τ as right factor" begin
            # BraidingTensor(V', V) on all domain indices
            ττ = TensorMap(BraidingTensor(V', V))
            @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[1 2; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ[1 2; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[-6 -5; 2 1]
            @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[2 -6; 1 -5]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V, V') adjoint on all domain indices
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[1 2; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ'[1 2; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[-6 -5; 2 1]
            @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[2 -6; 1 -5]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V, V) with mixed domain legs
            ττ = TensorMap(BraidingTensor(V, V))
            @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ[-4 -6; 1 2]
            @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * ττ[-4 -6; 1 2]
            @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ[2 1; -6 -4]
            @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ'[-6 2; -4 1]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4
        end

        @timedtestset "τ with fully contracted output" begin
            # scalar output
            ττ = TensorMap(BraidingTensor(V', V))
            @planar t1[(); (-1, -2)] := τ[2 1; 3 4] * t[1 2 3 4; -1 -2]
            @planar t2[(); (-1, -2)] := ττ[2 1; 3 4] * t[1 2 3 4; -1 -2]
            @planar t3[(); (-1, -2)] := τ[4 3; 1 2] * t[1 2 3 4; -1 -2]
            @planar t4[(); (-1, -2)] := τ'[1 4; 2 3] * t[1 2 3 4; -1 -2]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # rank-1 output
            ττ = TensorMap(BraidingTensor(V, V))
            @planar t1[-1; -2] := τ[2 1; 3 4] * t[-1 1 2 3; -2 4]
            @planar t2[-1; -2] := ττ[2 1; 3 4] * t[-1 1 2 3; -2 4]
            @planar t3[-1; -2] := τ[4 3; 1 2] * t[-1 1 2 3; -2 4]
            @planar t4[-1; -2] := τ'[1 4; 2 3] * t[-1 1 2 3; -2 4]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # rank-2 output
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2] := τ[2 1; 3 4] * t[-1 -2 1 2; 4 3]
            @planar t2[-1 -2] := ττ[2 1; 3 4] * t[-1 -2 1 2; 4 3]
            @planar t3[-1 -2] := τ[4 3; 1 2] * t[-1 -2 1 2; 4 3]
            @planar t4[-1 -2] := τ'[1 4; 2 3] * t[-1 -2 1 2; 4 3]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4
        end

        @timedtestset "τ with one open codomain leg" begin
            # BraidingTensor(V, V') with one open codomain leg
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2; -3 -4] := τ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
            @planar t2[-1 -2; -3 -4] := ττ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
            @planar t3[-1 -2; -3 -4] := τ[2 1; 3 -1] * t[1 2 3 -2; -3 -4]
            @planar t4[-1 -2; -3 -4] := τ'[3 2; -1 1] * t[1 2 3 -2; -3 -4]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V', V') adjoint with one open codomain leg
            ττ = TensorMap(BraidingTensor(V', V'))
            @planar t1[-1 -2; -3 -4] := τ'[-2 3; 1 2] * t[-1 1 2 3; -3 -4]
            @planar t2[-1 -2; -3 -4] := ττ'[-2 3; 1 2] * t[-1 1 2 3; -3 -4]
            @planar t3[-1 -2; -3 -4] := τ'[2 1; 3 -2] * t[-1 1 2 3; -3 -4]
            @planar t4[-1 -2; -3 -4] := τ[3 2; -2 1] * t[-1 1 2 3; -3 -4]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V', V) with one open codomain leg
            ττ = TensorMap(BraidingTensor(V', V))
            @planar t1[-1 -2 -3; -4] := τ[-3 3; 1 2] * t[-1 -2 1 2; -4 3]
            @planar t2[-1 -2 -3; -4] := ττ[-3 3; 1 2] * t[-1 -2 1 2; -4 3]
            @planar t3[-1 -2 -3; -4] := τ[2 1; 3 -3] * t[-1 -2 1 2; -4 3]
            @planar t4[-1 -2 -3; -4] := τ'[3 2; -3 1] * t[-1 -2 1 2; -4 3]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4
        end

        @timedtestset "τ as right factor with open domain leg" begin
            # BraidingTensor(V', V) as right factor with one open domain leg
            ττ = TensorMap(BraidingTensor(V', V))
            @planar t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[1 2; -4 3]
            @planar t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ[1 2; -4 3]
            @planar t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[3 -4; 2 1]
            @planar t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[2 3; 1 -4]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V, V') adjoint as right factor with one open domain leg
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[1 2; -4 3]
            @planar t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ'[1 2; -4 3]
            @planar t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[3 -4; 2 1]
            @planar t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[2 3; 1 -4]
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4
        end
    end
    TensorKit.empty_globalcaches!()
end
