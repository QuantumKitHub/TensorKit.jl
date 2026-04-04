using Test, TestExtras
using TensorKit
using TensorKit: type_repr
using Combinatorics: permutations


spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    hasbraiding = BraidingStyle(I) isa HasBraiding
    symmetricbraiding = BraidingStyle(I) isa SymmetricBraiding
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Trivial space insertion and removal" begin
            W = V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)'
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W)
                t2 = @constinferred insertleftunit(t)
                @test t2 == @constinferred insertrightunit(t)
                @test space(t2) == insertleftunit(space(t))
                @test @constinferred(removeunit(t2, $(numind(t2)))) == t
                t3 = @constinferred insertleftunit(t; copy = true)
                @test t3 == @constinferred insertrightunit(t; copy = true)
                @test @constinferred(removeunit(t3, $(numind(t3)))) == t

                @test numind(t2) == numind(t) + 1
                @test scalartype(t2) === T
                @test t.data === t2.data

                @test t.data !== t3.data
                for (c, b) in blocks(t)
                    @test b == block(t3, c)
                end

                t4 = @constinferred insertrightunit(t, 3; dual = true)
                @test numin(t4) == numin(t) + 1 && numout(t4) == numout(t)
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
        symmetricbraiding && @timedtestset "Permutations: test via inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = rand(ComplexF64, W)
            t′ = randn!(similar(t))
            for k in 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n -> p[n], k)
                    p2 = ntuple(n -> p[k + n], 5 - k)
                    t2 = @constinferred permute(t, (p1, p2))
                    @test norm(t2) ≈ norm(t)
                    t2′ = permute(t′, (p1, p2))
                    @test dot(t2′, t2) ≈ dot(t′, t) ≈ dot(transpose(t2′), transpose(t2))
                end

                t3 = @constinferred repartition(t, $k)
                @test norm(t3) ≈ norm(t)
                t3′ = @constinferred repartition!(similar(t3), t′)
                @test norm(t3′) ≈ norm(t′)
                @test dot(t′, t) ≈ dot(t3′, t3)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Permutations: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
                t = rand(ComplexF64, W)
                a = convert(Array, t)
                for k in 0:5
                    for p in permutations(1:5)
                        p1 = ntuple(n -> p[n], k)
                        p2 = ntuple(n -> p[k + n], 5 - k)
                        t2 = permute(t, (p1, p2))
                        a2 = convert(Array, t2)
                        @test a2 ≈ permutedims(a, (p1..., p2...))
                        @test convert(Array, transpose(t2)) ≈
                            permutedims(a2, (5, 4, 3, 2, 1))
                    end

                    t3 = repartition(t, k)
                    a3 = convert(Array, t3)
                    @test a3 ≈ permutedims(
                        a, (ntuple(identity, k)..., reverse(ntuple(i -> i + k, 5 - k))...)
                    )
                end
            end
        end
        hasbraiding && @timedtestset "Index flipping: test flipping inverse" begin
            t = rand(ComplexF64, V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)')
            for i in 1:5
                @test t ≈ flip(flip(t, i), i; inv = true)
                @test t ≈ flip(flip(t, i; inv = true), i)
            end
        end
        symmetricbraiding && @timedtestset "Index flipping: test via explicit flip" begin
            t = rand(ComplexF64, V1 ⊗ V1' ← V1' ⊗ V1)
            F1 = unitary(flip(V1), V1)

            @tensor tf[a, b; c, d] := F1[a, a'] * t[a', b; c, d]
            @test flip(t, 1) ≈ tf
            @tensor tf[a, b; c, d] := conj(F1[b, b']) * t[a, b'; c, d]
            @test twist!(flip(t, 2), 2) ≈ tf
            @tensor tf[a, b; c, d] := F1[c, c'] * t[a, b; c', d]
            @test flip(t, 3) ≈ tf
            @tensor tf[a, b; c, d] := conj(F1[d, d']) * t[a, b; c, d']
            @test twist!(flip(t, 4), 4) ≈ tf
        end
        symmetricbraiding && @timedtestset "Index flipping: test via contraction" begin
            t1 = rand(ComplexF64, V1 ⊗ V2 ⊗ V3 ← V4)
            t2 = rand(ComplexF64, V2' ⊗ V5 ← V4' ⊗ V1)
            @tensor ta[a, b] := t1[x, y, a, z] * t2[y, b, z, x]
            @tensor tb[a, b] := flip(t1, 1)[x, y, a, z] * flip(t2, 4)[y, b, z, x]
            @test ta ≈ tb
            @tensor tb[a, b] := flip(t1, (2, 4))[x, y, a, z] * flip(t2, (1, 3))[y, b, z, x]
            @test ta ≈ tb
            @tensor tb[a, b] := flip(t1, (1, 2, 4))[x, y, a, z] * flip(t2, (1, 3, 4))[y, b, z, x]
            @tensor tb[a, b] := flip(t1, (1, 3))[x, y, a, z] * flip(t2, (2, 4))[y, b, z, x]
            @test flip(ta, (1, 2)) ≈ tb
        end
    end
    TensorKit.empty_globalcaches!()
end
