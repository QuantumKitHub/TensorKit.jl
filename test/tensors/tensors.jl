using Test, TestExtras
using TensorKit
using TensorKit: type_repr
using Combinatorics: permutations
using LinearAlgebra: LinearAlgebra

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

spacelist = try
    if get(ENV, "CI", "false") == "true"
        println("Detected running on CI")
        if Sys.iswindows()
            (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VIB_diag, VIB_M)
        elseif Sys.isapple()
            (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VfU₁, VfSU₂, VSU₂U₁, VIB_diag, VIB_M) #, VSU₃)
        else
            (Vtr, Vℤ₂, Vfℤ₂, VU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M) #, VSU₃)
        end
    else
        (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M) #, VSU₃)
    end
catch
    (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M) #, VSU₃)
end

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    symmetricbraiding = isa(BraidingStyle(I), SymmetricBraiding)
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
                t = @constinferred zeros(T, W)
                @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, Vector{T}}
                # blocks
                bs = @constinferred blocks(t)
                if !isempty(blocksectors(t)) # multifusion space ending on module gives empty data
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
        end
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                t = @constinferred rand(T, W)
                d = convert(Dict, t)
                @test t == convert(TensorMap, d)
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
                        if T == Int
                            t = TensorMap{T}(undef, W)
                            for (_, b) in blocks(t)
                                rand!(b, -20:20)
                            end
                        else
                            t = @constinferred randn(T, W)
                        end
                        a = @constinferred convert(Array, t)
                        b = reshape(a, dim(codomain(W)), dim(domain(W)))
                        @test t ≈ @constinferred TensorMap(a, W)
                        @test t ≈ @constinferred TensorMap(b, W)
                        @test t === @constinferred TensorMap(t.data, W)
                    end
                end
                for T in (Int, Float32, ComplexF64)
                    t = randn(T, V1 ⊗ V2 ← zerospace(V1))
                    a = convert(Array, t)
                    @test norm(a) == 0
                end
            end
        end
        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W)
                @test scalartype(t) == T
                @test space(t) == W
                @test space(t') == W'
                @test dim(t) == dim(space(t))
                @test codomain(t) == codomain(W)
                @test domain(t) == domain(W)
                # blocks for adjoint
                bs = @constinferred blocks(t')
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W'))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t', first(blocksectors(t')))
                @test b1 == b2
                @test eltype(bs) === Pair{typeof(c), typeof(b1)}
                @test typeof(b1) === TensorKit.blocktype(t')
                @test typeof(c) === sectortype(t)
                # linear algebra
                @test isa(@constinferred(norm(t)), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
                p = 3 * rand(Float64)
                @test norm(t + t, p) ≈ 2 * norm(t, p)
                @test norm(t) ≈ norm(t')

                t2 = @constinferred rand!(similar(t))
                β = rand(T)
                @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')

                if isa(UnitStyle(I), SimpleUnit) || !isempty(blocksectors(V2 ⊗ V1))
                    i1 = @constinferred(isomorphism(T, V1 ⊗ V2, V2 ⊗ V1)) # can't reverse fusion here when modules are involved
                    i2 = @constinferred(isomorphism(Vector{T}, V2 ⊗ V1, V1 ⊗ V2))
                    @test i1 * i2 == @constinferred(id(T, V1 ⊗ V2))
                    @test i2 * i1 == @constinferred(id(Vector{T}, V2 ⊗ V1))
                end

                w = @constinferred isometry(T, V1 ⊗ (rightunitspace(V1) ⊕ rightunitspace(V1)), V1)
                @test dim(w) == 2 * dim(V1 ← V1)
                @test w' * w == id(Vector{T}, V1)
                @test w * w' == (w * w')^2
            end
        end
        @timedtestset "Trivial space insertion and removal" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
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
        if hasfusiontensor(I)
            @timedtestset "Basic linear algebra: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for T in (Float32, ComplexF64)
                    t = rand(T, W)
                    t2 = @constinferred rand!(similar(t))
                    @test norm(t, 2) ≈ norm(convert(Array, t), 2)
                    @test dot(t2, t) ≈ dot(convert(Array, t2), convert(Array, t))
                    α = rand(T)
                    @test convert(Array, α * t) ≈ α * convert(Array, t)
                    @test convert(Array, t + t) ≈ 2 * convert(Array, t)
                end
            end
            @timedtestset "Real and imaginary parts" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64, ComplexF32)
                    t = @constinferred randn(T, W, W)

                    tr = @constinferred real(t)
                    @test scalartype(tr) <: Real
                    @test real(convert(Array, t)) == convert(Array, tr)

                    ti = @constinferred imag(t)
                    @test scalartype(ti) <: Real
                    @test imag(convert(Array, t)) == convert(Array, ti)

                    tc = @inferred complex(t)
                    @test scalartype(tc) <: Complex
                    @test complex(convert(Array, t)) == convert(Array, tc)

                    tc2 = @inferred complex(tr, ti)
                    @test tc2 ≈ tc
                end
            end
        end
        @timedtestset "Tensor conversion" begin
            W = V1 ⊗ V2
            t = @constinferred randn(W ← W)
            @test typeof(convert(TensorMap, t')) == typeof(t)
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            @test typeof(convert(typeof(tc), t')) == typeof(tc)
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
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
        symmetricbraiding && @timedtestset "Full trace: test self-consistency" begin
            # TODO: extend to multifusion but keep these @tensor tests
            t = rand(ComplexF64, V1 ⊗ V2' ← V1 ⊗ V2')
            s = @constinferred tr(t)
            @test conj(s) ≈ tr(t')
            if !isdual(V1)
                t2 = twist!(t, 1)
            end
            if isdual(V2)
                t2 = twist!(t, 2)
            end
            ss = tr(t2)
            @tensor s2 = t[a b; a b]
            @tensor t3[a; b] := t[a c; b c]
            @tensor s3 = t3[a; a]
            @test ss ≈ s2
            @test ss ≈ s3
        end
        @timedtestset "Partial trace: test self-consistency" begin
            # TODO: extend to multifusion but keep these @tensor tests
            t = rand(ComplexF64, V1 ⊗ V2 ⊗ V3 ← V1 ⊗ V2 ⊗ V3)
            @tensor t2[a; b] := t[c d b; c d a]
            @tensor t4[a b; c d] := t[e d c; e b a]
            @tensor t5[a; b] := t4[a c; b c]
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
        (UnitStyle(I) == SimpleUnit()) && @timedtestset "Trace and contraction" begin
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
        (BraidingStyle(I) != NoBraiding()) && @timedtestset "Index flipping: test flipping inverse" begin
            t = rand(ComplexF64, V1 ⊗ V1' ← V1' ⊗ V1)
            for i in 1:4
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
        @timedtestset "Multiplication of isometries: test properties" begin
            W2 = V4 ⊗ V5
            W1 = W2 ⊗ (unitspace(V1) ⊕ unitspace(V1))
            for T in (Float64, ComplexF64)
                t1 = randisometry(T, W1, W2)
                t2 = randisometry(T, W2 ← W2)
                @test isisometry(t1)
                @test isunitary(t2)
                P = t1 * t1'
                @test P * P ≈ P
            end
        end
        @timedtestset "Multiplication and inverse: test compatibility" begin
            W1 = V1 ⊗ V2 ⊗ V3
            W2 = V4 ⊗ V5
            for T in (Float64, ComplexF64)
                t1 = rand(T, W1, W1)
                t2 = rand(T, W2 ← W2)
                t = rand(T, W1, W2)
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
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Multiplication and inverse: test via conversion" begin
                W1 = V1 ⊗ V2 ⊗ V3
                W2 = V4 ⊗ V5
                for T in (Float32, Float64, ComplexF32, ComplexF64)
                    t1 = rand(T, W1 ← W1)
                    t2 = rand(T, W2, W2)
                    t = rand(T, W1 ← W2)
                    d1 = dim(W1)
                    d2 = dim(W2)
                    At1 = reshape(convert(Array, t1), d1, d1)
                    At2 = reshape(convert(Array, t2), d2, d2)
                    At = reshape(convert(Array, t), d1, d2)
                    @test reshape(convert(Array, t1 * t), d1, d2) ≈ At1 * At
                    @test reshape(convert(Array, t1' * t), d1, d2) ≈ At1' * At
                    @test reshape(convert(Array, t2 * t'), d2, d1) ≈ At2 * At'
                    @test reshape(convert(Array, t2' * t'), d2, d1) ≈ At2' * At'

                    @test reshape(convert(Array, inv(t1)), d1, d1) ≈ inv(At1)
                    @test reshape(convert(Array, pinv(t)), d2, d1) ≈ pinv(At)

                    if T == Float32 || T == ComplexF32
                        continue
                    end

                    @test reshape(convert(Array, t1 \ t), d1, d2) ≈ At1 \ At
                    @test reshape(convert(Array, t1' \ t), d1, d2) ≈ At1' \ At
                    @test reshape(convert(Array, t2 \ t'), d2, d1) ≈ At2 \ At'
                    @test reshape(convert(Array, t2' \ t'), d2, d1) ≈ At2' \ At'

                    @test reshape(convert(Array, t2 / t), d2, d1) ≈ At2 / At
                    @test reshape(convert(Array, t2' / t), d2, d1) ≈ At2' / At
                    @test reshape(convert(Array, t1 / t'), d1, d2) ≈ At1 / At'
                    @test reshape(convert(Array, t1' / t'), d1, d2) ≈ At1' / At'
                end
            end
        end
        @timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            t = randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor functions" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64)
                    t = randn(T, W, W)
                    s = dim(W)
                    expt = @constinferred exp(t)
                    @test reshape(convert(Array, expt), (s, s)) ≈
                        exp(reshape(convert(Array, t), (s, s)))

                    @test (@constinferred sqrt(t))^2 ≈ t
                    @test reshape(convert(Array, sqrt(t^2)), (s, s)) ≈
                        sqrt(reshape(convert(Array, t^2), (s, s)))

                    @test exp(@constinferred log(expt)) ≈ expt
                    @test reshape(convert(Array, log(expt)), (s, s)) ≈
                        log(reshape(convert(Array, expt), (s, s)))

                    @test (@constinferred cos(t))^2 + (@constinferred sin(t))^2 ≈ id(W)
                    @test (@constinferred tan(t)) ≈ sin(t) / cos(t)
                    @test (@constinferred cot(t)) ≈ cos(t) / sin(t)
                    @test (@constinferred cosh(t))^2 - (@constinferred sinh(t))^2 ≈ id(W)
                    @test (@constinferred tanh(t)) ≈ sinh(t) / cosh(t)
                    @test (@constinferred coth(t)) ≈ cosh(t) / sinh(t)

                    t1 = sin(t)
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
                    @test coth(@constinferred acoth(t8)) ≈ t8
                    t = randn(T, W, V1) # not square
                    for f in
                        (
                            cos, sin, tan, cot, cosh, sinh, tanh, coth, atan, acot, asinh,
                            sqrt, log, asin, acos, acosh, atanh, acoth,
                        )
                        @test_throws SpaceMismatch f(t)
                    end
                end
            end
        end
        @timedtestset "Sylvester equation" begin
            for T in (Float32, ComplexF64)
                tA = rand(T, V1 ⊗ V3, V1 ⊗ V3)
                tB = rand(T, V2 ⊗ V4, V2 ⊗ V4)
                tA = 3 // 2 * left_polar(tA)[1]
                tB = 1 // 5 * left_polar(tB)[1]
                tC = rand(T, V1 ⊗ V3, V2 ⊗ V4)
                t = @constinferred sylvester(tA, tB, tC)
                @test codomain(t) == V1 ⊗ V3
                @test domain(t) == V2 ⊗ V4
                @test norm(tA * t + t * tB + tC) <
                    (norm(tA) + norm(tB) + norm(tC)) * eps(real(T))^(2 / 3)
                if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
                    matrix(x) = reshape(convert(Array, x), dim(codomain(x)), dim(domain(x)))
                    @test matrix(t) ≈ sylvester(matrix(tA), matrix(tB), matrix(tC))
                end
            end
        end
        @timedtestset "Tensor product: test via norm preservation" begin
            for T in (Float32, ComplexF64)
                if isa(UnitStyle(I), SimpleUnit) || !isempty(blocksectors(V2 ⊗ V1))
                    t1 = rand(T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
                    t2 = rand(T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
                else
                    t1 = rand(T, V3 ⊗ V4 ⊗ V5, V1 ⊗ V2)
                    t2 = rand(T, V5' ⊗ V4' ⊗ V3', V2' ⊗ V1')
                end
                t = @constinferred (t1 ⊗ t2)
                @test norm(t) ≈ norm(t1) * norm(t2)
            end
        end
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            @timedtestset "Tensor product: test via conversion" begin
                for T in (Float32, ComplexF64)
                    t1 = rand(T, V2 ⊗ V3 ⊗ V1, V1)
                    t2 = rand(T, V2 ⊗ V1 ⊗ V3, V2)
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
                t1 = rand(T, V2 ⊗ V3 ⊗ V1)
                t2 = rand(T, V2 ⊗ V1 ⊗ V3)
                t = @constinferred (t1 ⊗ t2)
                @tensor t′[1, 2, 3, 4, 5, 6] := t1[1, 2, 3] * t2[4, 5, 6]
                @test t ≈ t′
            end
        end
        @timedtestset "Tensor absorption" begin
            # absorbing small into large
            if isa(UnitStyle(I), SimpleUnit) || !isempty(blocksectors(V2 ⊗ V3))
                t1 = zeros(V1 ⊕ V1, V2 ⊗ V3)
                t2 = rand(V1, V2 ⊗ V3)
            else
                t1 = zeros(V1 ⊕ V2, V3 ⊗ V4 ⊗ V5)
                t2 = rand(V1, V3 ⊗ V4 ⊗ V5)
            end
            t3 = @constinferred absorb(t1, t2)
            @test norm(t3) ≈ norm(t2)
            @test norm(t1) == 0
            t4 = @constinferred absorb!(t1, t2)
            @test t1 === t4
            @test t3 ≈ t4

            # absorbing large into small
            if isa(UnitStyle(I), SimpleUnit) || !isempty(blocksectors(V2 ⊗ V3))
                t1 = rand(V1 ⊕ V1, V2 ⊗ V3)
                t2 = zeros(V1, V2 ⊗ V3)
            else
                t1 = rand(V1 ⊕ V2, V3 ⊗ V4 ⊗ V5)
                t2 = zeros(V1, V3 ⊗ V4 ⊗ V5)
            end
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
    @testset for Vlist1 in (Vtr, VSU₂), Vlist2 in (Vtr, Vℤ₂)
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = rand(T, V1 ⊗ V2, V3' ⊗ V4)
            t2 = rand(T, W2, W1 ⊗ W1')
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
