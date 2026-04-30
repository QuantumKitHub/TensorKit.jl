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
        @timedtestset "planaradd! with BraidingTensor" begin
            b = BraidingTensor(V, V')
            bb = TensorMap(b)
            # Cyclic rotations of the planar leg cycle (cod1, cod2, dom2, dom1).
            # Use transpose (F-symbols only) as reference, since permute requires SymmetricBraiding.
            # rotation 0 (identity)
            @planar t1[-1 -2; -3 -4] := b[-1 -2; -3 -4]
            @test t1 ≈ bb
            # rotation 1: single-tree cycle (4,1,2,3) → (p1=(2,4), p2=(1,3))
            @planar t2[-1 -2; -3 -4] := b[-3 -1; -4 -2]
            @test t2 ≈ transpose(bb, ((2, 4), (1, 3)))
            # rotation 2: single-tree cycle (3,4,1,2) → (p1=(4,3), p2=(2,1))
            @planar t3[-1 -2; -3 -4] := b[-4 -3; -2 -1]
            @test t3 ≈ transpose(bb, ((4, 3), (2, 1)))
            # rotation 3: single-tree cycle (2,3,4,1) → (p1=(3,1), p2=(4,2))
            @planar t4[-1 -2; -3 -4] := b[-2 -4; -1 -3]
            @test t4 ≈ transpose(bb, ((3, 1), (4, 2)))
            # adjoint BraidingTensor (rotation 0)
            ba = b'
            @planar t5[-1 -2; -3 -4] := ba[-1 -2; -3 -4]
            @test t5 ≈ TensorMap(ba)
        end

        @timedtestset "τ as left factor, all legs contracted" begin
            # BraidingTensor(V, V') on leading codomain indices
            ττ = TensorMap(BraidingTensor(V, V'))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -2 -1] * t[1 2 -3 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-2 2; -1 1] * t[1 2 -3 -4; -5 -6]
            @test t1 ≈ braid(t, ((2, 1, 3, 4), (5, 6)), (1, 2, 3, 4, 5, 6))
            @test t1 ≈ t2
            @test t1 ≈ t3
            @test t1 ≈ t4

            # BraidingTensor(V', V') on inner codomain indices
            ττ = TensorMap(BraidingTensor(V', V'))
            @planar t1[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
            @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
            @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
            @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
            @test t1 ≈ braid(t, ((1, 3, 2, 4), (5, 6)), (1, 2, 3, 4, 5, 6))
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

        @timedtestset "BraidingTensor × BraidingTensor" begin
            # b1 domain == b2 codomain == V⊗V', straight-through (planar) contraction
            b1 = BraidingTensor(V, V')   # space: V'⊗V ← V⊗V'
            b2 = BraidingTensor(V', V)   # space: V⊗V' ← V'⊗V
            bb1 = TensorMap(b1)
            bb2 = TensorMap(b2)
            @planar t1[-1 -2; -3 -4] := b1[-1 -2; 1 2] * b2[1 2; -3 -4]
            @planar t2[-1 -2; -3 -4] := bb1[-1 -2; 1 2] * bb2[1 2; -3 -4]
            @test t1 ≈ t2
        end
    end
    TensorKit.empty_globalcaches!()
end
