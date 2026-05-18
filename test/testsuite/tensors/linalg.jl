function basic_linear_algebra(rand_f, T, W)
    t = @testinferred rand_f(T, W)
    @test scalartype(t) == T
    @test space(t) == W
    @test space(t') == W'
    @test dim(t) == dim(space(t))
    @test codomain(t) == codomain(W)
    @test domain(t) == domain(W)

    # blocks for adjoint
    bs = @testinferred blocks(t')
    (c, b1), state = @testinferred Nothing iterate(bs)
    @test c == first(blocksectors(W'))
    next = @testinferred Nothing iterate(bs, state)
    b2 = @testinferred block(t', first(blocksectors(t')))
    @test b1 == b2
    @test eltype(bs) === Pair{typeof(c), typeof(b1)}
    @test typeof(b1) === TensorKit.blocktype(t')
    return @test typeof(c) === sectortype(t)
end

function tensor_norm(rand_f, T, W)
    return @testset "Tensor norm" begin
        t = @testinferred rand_f(T, W)
        @test scalartype(t) == T
        @test space(t) == W
        @test space(t') == W'
        @test dim(t) == dim(space(t))
        @test codomain(t) == codomain(W)
        @test domain(t) == domain(W)

        @test isa(@testinferred(norm(t)), real(T))
        @test norm(t)^2 ≈ dot(t, t)
        α = rand(T)
        @test norm(α * t) ≈ abs(α) * norm(t)
        @test norm(t + t, 2) ≈ 2 * norm(t, 2)
        @test norm(t + t, 1) ≈ 2 * norm(t, 1)
        @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
        p = 3 * rand(Float64)
        @test norm(t + t, p) ≈ 2 * norm(t, p)
        @test norm(t) ≈ norm(t')
    end
end

function tensor_dot(rand_f, T, W)
    return @testset "Tensor dot" begin
        t = @testinferred rand_f(T, W)
        @test scalartype(t) == T
        @test space(t) == W
        @test space(t') == W'
        @test dim(t) == dim(space(t))
        @test codomain(t) == codomain(W)
        @test domain(t) == domain(W)

        t2 = @testinferred rand_f(T, W)
        α = rand(T)
        β = rand(T)
        @test @testinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
        @test dot(t2, t) ≈ conj(dot(t, t2))
        @test dot(t2, t) ≈ conj(dot(t2', t'))
        @test dot(t2, t) ≈ dot(t', t2')
    end
end

function isomorphism_test(T, AT, V1, V2)
    return @testset "Isomorphism" begin
        I = TensorKit.sectortype(V1)
        if UnitStyle(I) isa SimpleUnit || !isempty(blocksectors(V2 ⊗ V1))
            i1 = @testinferred(isomorphism(AT, V1 ⊗ V2, V2 ⊗ V1)) # can't reverse fusion here when modules are involved
            i2 = @testinferred(isomorphism(AT, V2 ⊗ V1, V1 ⊗ V2))
            @test i1 * i2 == @testinferred(id(AT, V1 ⊗ V2))
            @test i2 * i1 == @testinferred(id(AT, V2 ⊗ V1))
        end
    end
end

function isometry_test(T, AT, V1)
    return @testset "Isometry" begin
        w = @testinferred isometry(AT, V1 ⊗ (rightunitspace(V1) ⊕ rightunitspace(V1)), V1)
        @test dim(w) == 2 * dim(V1 ← V1)
        @test w' * w == id(AT, V1)
        @test w * w' == (w * w')^2
    end
end

function linalg_via_conversion(rand_f, T, W)
    return @testset "Linalg via conversion" begin
        t = rand_f(T, W)
        t2 = rand_f(T, W)
        @test norm(t, 2) ≈ norm(convert(Array, t), 2)
        @test dot(t2, t) ≈ dot(convert(Array, t2), convert(Array, t))
        α = rand(T)
        @test convert(Array, α * t) ≈ α * convert(Array, t)
        @test convert(Array, t + t) ≈ 2 * convert(Array, t)
    end
end

function multiplying_isometries(AT, W1, W2)
    return @testset "Multiplication of isometries" begin
        t1 = randisometry(AT, W1, W2)
        t2 = randisometry(AT, W2 ← W2)
        @test isisometric(t1)
        @test isunitary(t2)
        P = t1 * t1'
        @test P * P ≈ P
    end
end

function tensor_multiplication_and_inverse(rand_f, T, W1, W2)
    return @testset "Multiplication and inverse -- compatibility" begin
        t1 = rand_f(T, W1, W1)
        t2 = rand_f(T, W2 ← W2)
        t = rand_f(T, W1, W2)
        @test t1 * (t1 \ t) ≈ t
        @test (t / t2) * t2 ≈ t
        @test t1 \ one(t1) ≈ inv(t1)
        # TODO pinv doesn't work for GPUArrays yet
        if TensorKit.storagetype(t1) <: Vector
            @test one(t1) / t1 ≈ pinv(t1)
        end
        @test_throws SpaceMismatch inv(t)
        @test_throws SpaceMismatch t2 \ t
        @test_throws SpaceMismatch t / t1
        # TODO pinv doesn't work for GPUArrays yet
        if TensorKit.storagetype(t) <: Vector
            tp = pinv(t) * t
            @test tp ≈ tp * tp
        end
    end
end

function tensor_multiplication_and_inverse_conversion(rand_f, T, W1, W2)
    return @testset "Multiplication and inverse -- conversion" begin
        t1 = rand_f(T, W1 ← W1)
        t2 = rand_f(T, W2, W2)
        t = rand_f(T, W1 ← W2)
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
        # TODO pinv doesn't work for GPUArrays yet
        if TensorKit.storagetype(t1) <: Vector
            @test reshape(convert(Array, pinv(t)), d2, d1) ≈ pinv(At)
        end

        if !(T == Float32 || T == ComplexF32)
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

function diag_diagm(randn_f, T, W)
    return @testset "diag/diagm" begin
        t = randn_f(T, W)
        d = LinearAlgebra.diag(t)
        D = LinearAlgebra.diagm(codomain(t), domain(t), d)
        @test LinearAlgebra.isdiag(D)
        @test LinearAlgebra.diag(D) == d
    end
end

function tensor_functions(randn_f, T, V1, V2)
    return @testset "Tensor functions" begin
        W = V1 ⊗ V2
        t = randn_f(T, W, W)
        s = dim(W)
        expt = @testinferred exp(t)
        @test reshape(convert(Array, expt), (s, s)) ≈
            exp(reshape(convert(Array, t), (s, s)))

        @test (@testinferred sqrt(t))^2 ≈ t
        @test reshape(convert(Array, sqrt(t^2)), (s, s)) ≈
            sqrt(reshape(convert(Array, t^2), (s, s)))

        @test exp(@testinferred log(expt)) ≈ expt
        @test reshape(convert(Array, log(expt)), (s, s)) ≈
            log(reshape(convert(Array, expt), (s, s)))

        @test (@testinferred cos(t))^2 + (@testinferred sin(t))^2 ≈ id(W)
        @test (@testinferred tan(t)) ≈ sin(t) / cos(t)
        @test (@testinferred cot(t)) ≈ cos(t) / sin(t)
        @test (@testinferred cosh(t))^2 - (@testinferred sinh(t))^2 ≈ id(W)
        @test (@testinferred tanh(t)) ≈ sinh(t) / cosh(t)
        @test (@testinferred coth(t)) ≈ cosh(t) / sinh(t)

        t1 = sin(t)
        @test sin(@testinferred asin(t1)) ≈ t1
        t2 = cos(t)
        @test cos(@testinferred acos(t2)) ≈ t2
        t3 = sinh(t)
        @test sinh(@testinferred asinh(t3)) ≈ t3
        t4 = cosh(t)
        @test cosh(@testinferred acosh(t4)) ≈ t4
        t5 = tan(t)
        @test tan(@testinferred atan(t5)) ≈ t5
        t6 = cot(t)
        @test cot(@testinferred acot(t6)) ≈ t6
        t7 = tanh(t)
        @test tanh(@testinferred atanh(t7)) ≈ t7
        t8 = coth(t)
        @test coth(@testinferred acoth(t8)) ≈ t8
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

function sylvester_test(rand_f, T, V)
    return @testset "Sylvester" begin
        V1, V2, V3, V4, V5 = V
        tA = rand_f(T, V1 ⊗ V2, V1 ⊗ V2)
        tB = rand_f(T, (V3 ⊗ V4 ⊗ V5)', (V3 ⊗ V4 ⊗ V5)')
        tA = 3 // 2 * left_polar(tA)[1]
        tB = 1 // 5 * left_polar(tB)[1]
        tC = rand_f(T, V1 ⊗ V2, (V3 ⊗ V4 ⊗ V5)')
        t = @testinferred sylvester(tA, tB, tC)
        @test codomain(t) == V1 ⊗ V2
        @test domain(t) == (V3 ⊗ V4 ⊗ V5)'
        lnorm = norm(tA * t + t * tB + tC)
        rnorm = (norm(tA) + norm(tB) + norm(tC))
        @test lnorm < rnorm * eps(real(T))^(2 / 3)
        I = TensorKit.sectortype(first(V))
        if BraidingStyle(I) isa Bosonic && hasfusiontensor(I)
            matrix(x) = reshape(convert(Array, x), dim(codomain(x)), dim(domain(x)))
            @test matrix(t) ≈ sylvester(matrix(tA), matrix(tB), matrix(tC))
        end
    end
end
