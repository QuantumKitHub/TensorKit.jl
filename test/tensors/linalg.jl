using Test, TestExtras
using TensorKit
using TensorKit: type_repr
using LinearAlgebra: LinearAlgebra


spacelist = if fast_tests
    (Vtr, Vℤ₃, VSU₂)
elseif get(ENV, "CI", "false") == "true"
    println("Detected running on CI")
    if Sys.iswindows()
        (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VIB_diag)
    elseif Sys.isapple()
        (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VfU₁, VfSU₂, VSU₂U₁, VIB_M) #, VSU₃)
    else
        (Vtr, Vℤ₂, Vfℤ₂, VU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M) #, VSU₃)
    end
else
    (Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, VIB_diag, VIB_M) #, VSU₃)
end

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    symmetricbraiding = BraidingStyle(I) isa SymmetricBraiding
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
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

                if UnitStyle(I) isa SimpleUnit || !isempty(blocksectors(V2 ⊗ V1))
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
        end
        @timedtestset "Multiplication of isometries: test properties" begin
            W2 = V4 ⊗ V5
            W1 = W2 ⊗ (unitspace(V1) ⊕ unitspace(V1))
            for T in (Float64, ComplexF64)
                t1 = randisometry(T, W1, W2)
                t2 = randisometry(T, W2 ← W2)
                @test isisometric(t1)
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
    end
    TensorKit.empty_globalcaches!()
end
