using Test, TestExtras
using TensorKit
using TensorKit: type_repr


spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    symmetricbraiding = BraidingStyle(I) isa SymmetricBraiding
    println("---------------------------------------")
    println("Tensor contractions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensor contractions with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Full trace: test self-consistency" begin
            if symmetricbraiding
                t = rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
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
            t = rand(ComplexF64, V1 ⊗ V2 ← V1 ⊗ V2) # avoid permutes
            ss = @constinferred tr(t)
            @test conj(ss) ≈ tr(t')
            @planar s2 = t[a b; a b]
            @planar t3[a; b] := t[a c; b c]
            @planar s3 = t3[a; a]

            @test ss ≈ s2
            @test ss ≈ s3
        end
        @timedtestset "Partial trace: test self-consistency" begin
            if symmetricbraiding
                t = rand(ComplexF64, V1 ⊗ V2 ⊗ V3 ← V1 ⊗ V2 ⊗ V3)
                @tensor t2[a; b] := t[c d b; c d a]
                @tensor t4[a b; c d] := t[e d c; e b a]
                @tensor t5[a; b] := t4[a c; b c]
                @test t2 ≈ t5
            end
            t = rand(ComplexF64, V3 ⊗ V4 ⊗ V5 ← V3 ⊗ V4 ⊗ V5) # compatible with module fusion
            @planar t2[a; b] := t[c a d; c b d]
            @planar t4[a b; c d] := t[e a b; e c d]
            @planar t5[a; b] := t4[a c; b c]
            @test t2 ≈ t5
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Trace: test via conversion" begin
                t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
                @tensor t2[a, b] := t[c, d, b, d, c, a]
                @tensor t3[a, b] := convert(Array, t)[c, d, b, d, c, a]
                @test t3 ≈ convert(Array, t2)
            end
        end
        #TODO: find version that works for all multifusion cases
        symmetricbraiding && @timedtestset "Trace and contraction" begin
            t1 = rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
            t2 = rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
            t3 = t1 ⊗ t2
            @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
            @tensor tb[a, b] := t3[x, y, a, y, b, x]
            @test ta ≈ tb
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor contraction: test via conversion" begin
                A1 = randn(ComplexF64, V1' * V2', V3')
                A2 = randn(ComplexF64, V3 * V4, V5)
                rhoL = randn(ComplexF64, V1, V1)
                rhoR = randn(ComplexF64, V5, V5)' # test adjoint tensor
                H = randn(ComplexF64, V2 * V4, V2 * V4)
                @tensor HrA12[a, s1, s2, c] := rhoL[a, a'] * conj(A1[a', t1, b]) *
                    A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]

                @tensor HrA12array[a, s1, s2, c] := convert(Array, rhoL)[a, a'] *
                    conj(convert(Array, A1)[a', t1, b]) * convert(Array, A2)[b, t2, c'] *
                    convert(Array, rhoR)[c', c] * convert(Array, H)[s1, s2, t1, t2]

                @test HrA12array ≈ convert(Array, HrA12)
            end
        end
        @timedtestset "Tensor product: test via norm preservation" begin
            for T in (Float32, ComplexF64)
                t1 = rand(T, V1, V5')
                t2 = rand(T, V2 ⊗ V3, V4')
                t = @constinferred (t1 ⊗ t2)
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor product: test via conversion" begin
                for T in (Float32, ComplexF64)
                    t1 = rand(T, V1, V5')
                    t2 = rand(T, V2 ⊗ V3, V4')
                    t = @constinferred (t1 ⊗ t2)
                    d1 = dim(codomain(t1))
                    d2 = dim(codomain(t2))
                    d3 = dim(domain(t1))
                    d4 = dim(domain(t2))
                    At = convert(Array, t)
                    @test reshape(At, (d1, d2, d3, d4)) ≈
                        reshape(convert(Array, t1), (d1, 1, d3, 1)) .*
                        reshape(convert(Array, t2), (1, d2, 1, d4))
                end
            end
        end
        symmetricbraiding && @timedtestset "Tensor product: test via tensor contraction" begin
            for T in (Float32, ComplexF64)
                t1 = rand(T, V1, V5')
                t2 = rand(T, V2 ⊗ V3, V4')
                t = @constinferred (t1 ⊗ t2)
                @tensor t′[1 2 3; 4 5] := t1[1; 4] * t2[2 3; 5]
                @test t ≈ t′
            end
        end
        @timedtestset "Tensor absorption" begin
            # absorbing small into large
            t1 = zeros((V1 ⊕ V1) ⊗ (V2 ⊕ V2), (V3 ⊗ (V4 ⊕ V4) ⊗ V5)')
            t2 = rand(V1 ⊗ V2, (V3 ⊗ V4 ⊗ V5)')
            t3 = @constinferred absorb(t1, t2)
            @test norm(t3) ≈ norm(t2)
            @test norm(t1) == 0
            t4 = @constinferred absorb!(t1, t2)
            @test t1 === t4
            @test t3 ≈ t4

            # absorbing large into small
            t1 = rand((V1 ⊕ V1) ⊗ (V2 ⊕ V2), (V3 ⊗ (V4 ⊕ V4) ⊗ V5)')
            t2 = zeros(V1 ⊗ V2, (V3 ⊗ V4 ⊗ V5)')
            t3 = @constinferred absorb(t2, t1)
            @test norm(t3) < norm(t1)
            @test norm(t2) == 0
            t4 = @constinferred absorb!(t2, t1)
            @test t2 === t4
            @test t3 ≈ t4
        end
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "Deligne tensor product: test via conversion" begin
    using .TestSetup: Vtr, VRepℤ₂, VRepSU₂, VRepA4
    @testset for Vlist1 in (Vtr, VRepSU₂), Vlist2 in (VRepℤ₂, VRepA4)
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = rand(T, V2 ⊗ V3, (V4 ⊗ V5)')
            t2 = rand(T, W2, (W3 ⊗ W4)')
            t = @constinferred (t1 ⊠ t2)
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            At = convert(Array, t)
            @test reshape(At, (d1, d2, d3, d4)) ≈
                reshape(convert(Array, t1), (d1, 1, d3, 1)) .*
                reshape(convert(Array, t2), (1, d2, 1, d4))
        end
    end
end
