diagspacelist = ((ℂ^4)', ℂ[Z2Irrep](0 => 2, 1 => 3),
                 ℂ[FermionNumber](0 => 2, 1 => 2, -1 => 1),
                 ℂ[SU2Irrep](0 => 2, 1 => 1)', ℂ[FibonacciAnyon](:I => 2, :τ => 2))

@testset "DiagonalTensor with domain $V" for V in diagspacelist
    @timedtestset "Basic properties and algebra" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            # constructors
            t = @constinferred DiagonalTensorMap{T}(undef, V)
            t = @constinferred DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = @constinferred DiagonalTensorMap{T}(undef, space(t))
            @test space(t2) == space(t)
            @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2 ← V)
            t2 = @constinferred DiagonalTensorMap{T}(undef, domain(t))
            @test space(t2) == space(t)
            @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2)
            # properties
            @test @constinferred(hash(t)) == hash(deepcopy(t))
            @test scalartype(t) == T
            @test codomain(t) == ProductSpace(V)
            @test domain(t) == ProductSpace(V)
            @test space(t) == (V ← V)
            @test space(t') == (V ← V)
            @test dim(t) == dim(space(t))
            # blocks
            bs = @constinferred blocks(t)
            (c, b1), state = @constinferred Nothing iterate(bs)
            @test c == first(blocksectors(V ← V))
            next = @constinferred Nothing iterate(bs, state)
            b2 = @constinferred block(t, first(blocksectors(t)))
            @test b1 == b2
            @test eltype(bs) === Pair{typeof(c),typeof(b1)}
            @test typeof(b1) === TK.blocktype(t)
            # basic linear algebra
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

            @test t == @constinferred(TensorMap(t))
            @test norm(t + TensorMap(t)) ≈ 2 * norm(t)

            @test norm(zerovector!(t)) == 0
            @test norm(one!(t)) ≈ sqrt(dim(V))
            @test one!(t) == id(V)
            @test norm(one!(t) - id(V)) == 0

            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t3 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            α = rand(T)
            β = rand(T)
            @test @constinferred(dot(t1, t2)) ≈ conj(dot(t2, t1))
            @test dot(t2, t1) ≈ conj(dot(t2', t1'))
            @test dot(t3, α * t1 + β * t2) ≈ α * dot(t3, t1) + β * dot(t3, t2)
        end
    end
    I = sectortype(V)
    @timedtestset "Basic linear algebra: test via conversion" begin
        for T in (Float32, ComplexF64)
            t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @test norm(t1, 2) ≈ norm(convert(TensorMap, t1), 2)
            @test dot(t2, t1) ≈ dot(convert(TensorMap, t2), convert(TensorMap, t1))
            α = rand(T)
            @test convert(TensorMap, α * t1) ≈ α * convert(TensorMap, t1)
            @test convert(TensorMap, t1') ≈ convert(TensorMap, t1)'
            @test convert(TensorMap, t1 + t2) ≈
                  convert(TensorMap, t1) + convert(TensorMap, t2)
        end
    end
    @timedtestset "Real and imaginary parts" begin
        for T in (Float64, ComplexF64, ComplexF32)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)

            tr = @constinferred real(t)
            @test scalartype(tr) <: Real
            @test real(convert(TensorMap, t)) == convert(TensorMap, tr)

            ti = @constinferred imag(t)
            @test scalartype(ti) <: Real
            @test imag(convert(TensorMap, t)) == convert(TensorMap, ti)

            tc = @inferred complex(t)
            @test scalartype(tc) <: Complex
            @test complex(convert(TensorMap, t)) == convert(TensorMap, tc)

            tc2 = @inferred complex(tr, ti)
            @test tc2 ≈ tc
        end
    end
    @timedtestset "Tensor conversion" begin
        t = @constinferred DiagonalTensorMap(undef, V)
        rand!(t.data)
        # element type conversion
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
        # to and from generic TensorMap
        td = DiagonalTensorMap(TensorMap(t))
        @test t == td
        @test typeof(td) == typeof(t)
    end
    I = sectortype(V)
    if BraidingStyle(I) isa SymmetricBraiding
        @timedtestset "Permutations" begin
            t = DiagonalTensorMap(randn(ComplexF64, reduceddim(V)), V)
            t1 = @constinferred permute(t, $(((2,), (1,))))
            if BraidingStyle(sectortype(V)) isa Bosonic
                @test t1 ≈ transpose(t)
            end
            @test convert(TensorMap, t1) == permute(convert(TensorMap, t), (((2,), (1,))))
            t2 = @constinferred permute(t, $(((1, 2), ())))
            @test convert(TensorMap, t2) == permute(convert(TensorMap, t), (((1, 2), ())))
            t3 = @constinferred permute(t, $(((2, 1), ())))
            @test convert(TensorMap, t3) == permute(convert(TensorMap, t), (((2, 1), ())))
            t4 = @constinferred permute(t, $(((), (1, 2))))
            @test convert(TensorMap, t4) == permute(convert(TensorMap, t), (((), (1, 2))))
            t5 = @constinferred permute(t, $(((), (2, 1))))
            @test convert(TensorMap, t5) == permute(convert(TensorMap, t), (((), (2, 1))))
        end
    end
    @timedtestset "Trace, Multiplication and inverse" begin
        t1 = DiagonalTensorMap(rand(Float64, reduceddim(V)), V)
        t2 = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
        @test tr(TensorMap(t1)) == @constinferred tr(t1)
        @test tr(TensorMap(t2)) == @constinferred tr(t2)
        @test TensorMap(@constinferred t1 * t2) ≈ TensorMap(t1) * TensorMap(t2)
        @test TensorMap(@constinferred t1 \ t2) ≈ TensorMap(t1) \ TensorMap(t2)
        @test TensorMap(@constinferred t1 / t2) ≈ TensorMap(t1) / TensorMap(t2)
        @test TensorMap(@constinferred inv(t1)) ≈ inv(TensorMap(t1))
        @test TensorMap(@constinferred pinv(t1)) ≈ pinv(TensorMap(t1))
        @test all(Base.Fix2(isa, DiagonalTensorMap),
                  (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1)))

        u = randn(Float64, V * V' * V, V)
        @test u * t1 ≈ u * TensorMap(t1)
        @test u / t1 ≈ u / TensorMap(t1)
        @test t1 * u' ≈ TensorMap(t1) * u'
        @test t1 \ u' ≈ TensorMap(t1) \ u'

        t3 = rand(Float64, V ← V^2)
        t4 = rand(ComplexF64, V ← V^2)
        @test t1 * t3 ≈ lmul!(t1, copy(t3))
        @test t2 * t4 ≈ lmul!(t2, copy(t4))

        t3 = rand(Float64, V^2 ← V)
        t4 = rand(ComplexF64, V^2 ← V)
        @test t3 * t1 ≈ rmul!(copy(t3), t1)
        @test t4 * t2 ≈ rmul!(copy(t4), t2)
    end
    @timedtestset "Tensor contraction" begin
        d = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
        t = TensorMap(d)
        A = randn(ComplexF64, V ⊗ V' ⊗ V, V)
        B = randn(ComplexF64, V ⊗ V' ⊗ V, V ⊗ V')
        if BraidingStyle(I) isa SymmetricBraiding
            @tensor C[a b c; d] := A[a b c; e] * d[e, d]
            @test C ≈ A * d
            @tensor D[a; b] := d[a, c] * d[c, b]
            @test D ≈ d * d
            @test D isa DiagonalTensorMap
        end
        @planar E1[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * d[1; -4]
        @planar E2[-1 -2 -3; -4 -5] := B[-1 -2 -3; 1 -5] * t[1; -4]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * d'[-5; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 -3; -4 1] * t'[-5; 1]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * d[-1; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[1 -2 -3; -4 -5] * t[-1; 1]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * d[1; -2]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 1 -3; -4 -5] * t[1; -2]
        @test E1 ≈ E2
        @planar E1[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * d'[-3; 1]
        @planar E2[-1 -2 -3; -4 -5] = B[-1 -2 1; -4 -5] * t'[-3; 1]
        @test E1 ≈ E2
    end
    @timedtestset "Factorization" begin
        for T in (Float32, ComplexF64)
            t = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            @testset "eig" begin
                D, W = @constinferred eig(t)
                @test t * W ≈ W * D
                t2 = t + t'
                D2, V2 = @constinferred eigh(t2)
                VdV2 = V2' * V2
                @test VdV2 ≈ one(VdV2)
                @test t2 * V2 ≈ V2 * D2

                @test rank(D) ≈ rank(t)
                @test cond(D) ≈ cond(t)
                @test all(((s, t),) -> isapprox(s, t),
                          zip(values(LinearAlgebra.eigvals(D)),
                              values(LinearAlgebra.eigvals(t))))
            end
            @testset "leftorth with $alg" for alg in (TK.QR(), TK.QL())
                Q, R = @constinferred leftorth(t; alg=alg)
                QdQ = Q' * Q
                @test QdQ ≈ one(QdQ)
                @test Q * R ≈ t
                if alg isa Polar
                    @test isposdef(R)
                end
            end
            @testset "rightorth with $alg" for alg in (TK.RQ(), TK.LQ())
                L, Q = @constinferred rightorth(t; alg=alg)
                QQd = Q * Q'
                @test QQd ≈ one(QQd)
                @test L * Q ≈ t
                if alg isa Polar
                    @test isposdef(L)
                end
            end
            @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                U, S, Vᴴ = @constinferred tsvd(t; alg=alg)
                UdU = U' * U
                @test UdU ≈ one(UdU)
                VdV = Vᴴ * Vᴴ'
                @test VdV ≈ one(VdV)
                @test U * S * Vᴴ ≈ t

                @test rank(S) ≈ rank(t)
                @test cond(S) ≈ cond(t)
                @test all(((s, t),) -> isapprox(s, t),
                          zip(values(LinearAlgebra.svdvals(S)),
                              values(LinearAlgebra.svdvals(t))))
            end
        end
    end
    @timedtestset "Tensor functions" begin
        for T in (Float64, ComplexF64)
            d = DiagonalTensorMap(rand(T, reduceddim(V)), V)
            # rand is important for positive numbers in the real case, for log and sqrt
            t = TensorMap(d)
            @test @constinferred exp(d) ≈ exp(t)
            @test @constinferred log(d) ≈ log(t)
            @test @constinferred sqrt(d) ≈ sqrt(t)
            @test @constinferred sin(d) ≈ sin(t)
            @test @constinferred cos(d) ≈ cos(t)
            @test @constinferred tan(d) ≈ tan(t)
            @test @constinferred cot(d) ≈ cot(t)
            @test @constinferred sinh(d) ≈ sinh(t)
            @test @constinferred cosh(d) ≈ cosh(t)
            @test @constinferred tanh(d) ≈ tanh(t)
            @test @constinferred coth(d) ≈ coth(t)
            @test @constinferred asin(d) ≈ asin(t)
            @test @constinferred acos(d) ≈ acos(t)
            @test @constinferred atan(d) ≈ atan(t)
            @test @constinferred acot(d) ≈ acot(t)
            @test @constinferred asinh(d) ≈ asinh(t)
            @test @constinferred acosh(one(d) + d) ≈ acosh(one(t) + t)
            @test @constinferred atanh(d) ≈ atanh(t)
            @test @constinferred acoth(one(t) + d) ≈ acoth(one(d) + t)
        end
    end
end
