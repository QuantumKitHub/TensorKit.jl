# DiagonalTensor operations
# =====================

@testsuite :diagonal_tensors "basic properties and algebra" V -> begin
    for T in (fast_tests[] ? (Float64, ComplexF64) : (Float32, Float64, ComplexF32, ComplexF64, BigFloat))
        # constructors
        t = @testinferred DiagonalTensorMap{T}(undef, V)
        t = @testinferred DiagonalTensorMap(rand(T, reduceddim(V)), V)
        t2 = @testinferred DiagonalTensorMap{T}(undef, space(t))
        @test space(t2) == space(t)
        @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2 ← V)
        t2 = @testinferred DiagonalTensorMap{T}(undef, domain(t))
        @test space(t2) == space(t)
        @test_throws ArgumentError DiagonalTensorMap{T}(undef, V^2)
        # properties
        @test @testinferred(hash(t)) == hash(deepcopy(t))
        @test scalartype(t) == T
        @test codomain(t) == ProductSpace(V)
        @test domain(t) == ProductSpace(V)
        @test space(t) == (V ← V)
        @test space(t') == (V ← V)
        @test dim(t) == dim(space(t))
        # blocks
        bs = @testinferred blocks(t)
        (c, b1), state = @testinferred Nothing iterate(bs)
        @test c == first(blocksectors(V ← V))
        next = @testinferred Nothing iterate(bs, state)
        b2 = @testinferred block(t, first(blocksectors(t)))
        @test b1 == b2
        @test eltype(bs) === Pair{typeof(c), typeof(b1)}
        @test typeof(b1) === TensorKit.blocktype(t)
        # basic linear algebra
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

        @test t == @testinferred(TensorMap(t))
        @test norm(t + TensorMap(t)) ≈ norm(TensorMap(t) + t) ≈ 2 * norm(t)

        @test norm(zerovector!(t)) == 0
        @test norm(one!(t)) ≈ sqrt(dim(V))
        @test one!(t) == id(V)
        if T != BigFloat # seems broken for now
            @test norm(one!(t) - id(V)) == 0
        end

        t2 = randn!(TensorMap(t))
        @test t2 + t ≈ t + t2 ≈ t2 + TensorMap(t)

        t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
        t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
        t3 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
        α = rand(T)
        β = rand(T)
        @test @testinferred(dot(t1, t2)) ≈ conj(dot(t2, t1))
        @test dot(t2, t1) ≈ conj(dot(t2', t1'))
        @test dot(t3, α * t1 + β * t2) ≈ α * dot(t3, t1) + β * dot(t3, t2)
    end
end

@testsuite :diagonal_tensors "linear algebra conversion" V -> begin
    for T in (Float32, ComplexF64)
        t1 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
        t2 = DiagonalTensorMap(rand(T, reduceddim(V)), V)
        @test norm(t1, 2) ≈ norm(convert(TensorMap, t1), 2)
        @test dot(t2, t1) ≈ dot(convert(TensorMap, t2), convert(TensorMap, t1))
        α = rand(T)
        @test convert(TensorMap, α * t1) ≈ α * convert(TensorMap, t1)
        @test convert(TensorMap, t1') ≈ convert(TensorMap, t1)'
        @test convert(TensorMap, t1 + t2) ≈ convert(TensorMap, t1) + convert(TensorMap, t2)
    end
end

@testsuite :diagonal_tensors "real and imaginary parts" V -> begin
    for T in (Float64, ComplexF64, ComplexF32)
        t = DiagonalTensorMap(rand(T, reduceddim(V)), V)

        tr = @testinferred real(t)
        @test scalartype(tr) <: Real
        @test real(convert(TensorMap, t)) == convert(TensorMap, tr)

        ti = @testinferred imag(t)
        @test scalartype(ti) <: Real
        @test imag(convert(TensorMap, t)) == convert(TensorMap, ti)

        tc = @inferred complex(t)
        @test scalartype(tc) <: Complex
        @test complex(convert(TensorMap, t)) == convert(TensorMap, tc)

        tc2 = @inferred complex(tr, ti)
        @test tc2 ≈ tc
    end
end

@testsuite :diagonal_tensors "tensor conversion" V -> begin
    t = @testinferred DiagonalTensorMap(undef, V)
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

@testsuite :diagonal_tensors "permutations" V -> begin
    I = sectortype(V)
    BraidingStyle(I) isa SymmetricBraiding || return nothing
    t = DiagonalTensorMap(randn(ComplexF64, reduceddim(V)), V)
    t_tm = convert(TensorMap, t)

    # preserving diagonal
    t1 = @testinferred permute(t, ((2,), (1,)))
    @test t1 isa DiagonalTensorMap
    @test convert(TensorMap, t1) == permute(t_tm, (((2,), (1,))))
    t1′ = @testinferred transpose(t)
    @test t1′ isa DiagonalTensorMap
    @test convert(TensorMap, t1′) == transpose(t_tm)
    BraidingStyle(I) isa Bosonic && @test t1 ≈ t1′

    # not preserving diagonal
    t2 = @testinferred permute(t, ((1, 2), ()))
    @test convert(TensorMap, t2) == permute(t_tm, (((1, 2), ())))
    t3 = @testinferred permute(t, ((2, 1), ()))
    @test convert(TensorMap, t3) == permute(t_tm, (((2, 1), ())))
    t4 = @testinferred permute(t, ((), (1, 2)))
    @test convert(TensorMap, t4) == permute(t_tm, (((), (1, 2))))
    t5 = @testinferred permute(t, ((), (2, 1)))
    @test convert(TensorMap, t5) == permute(t_tm, (((), (2, 1))))
end

@testsuite :diagonal_tensors "trace, multiplication and inverse" V -> begin
    t1 = DiagonalTensorMap(rand(Float64, reduceddim(V)), V)
    t2 = DiagonalTensorMap(rand(ComplexF64, reduceddim(V)), V)
    @test tr(TensorMap(t1)) == @testinferred tr(t1)
    @test tr(TensorMap(t2)) == @testinferred tr(t2)
    @test TensorMap(@testinferred t1 * t2) ≈ TensorMap(t1) * TensorMap(t2)
    @test TensorMap(@testinferred t1 \ t2) ≈ TensorMap(t1) \ TensorMap(t2)
    @test TensorMap(@testinferred t1 / t2) ≈ TensorMap(t1) / TensorMap(t2)
    @test TensorMap(@testinferred inv(t1)) ≈ inv(TensorMap(t1))
    @test TensorMap(@testinferred pinv(t1)) ≈ pinv(TensorMap(t1))
    @test all(
        Base.Fix2(isa, DiagonalTensorMap), (t1 * t2, t1 \ t2, t1 / t2, inv(t1), pinv(t1))
    )

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

@testsuite :diagonal_tensors "contraction" V -> begin
    I = sectortype(V)
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

@testsuite :diagonal_tensors "tensor functions" V -> begin
    for T in (Float64, ComplexF64)
        d = DiagonalTensorMap(rand(T, reduceddim(V)), V)
        # rand is important for positive numbers in the real case, for log and sqrt
        t = TensorMap(d)
        @test @testinferred exp(d) ≈ exp(t)
        @test @testinferred log(d) ≈ log(t)
        @test @testinferred sqrt(d) ≈ sqrt(t)
        @test @testinferred sin(d) ≈ sin(t)
        @test @testinferred cos(d) ≈ cos(t)
        @test @testinferred tan(d) ≈ tan(t)
        @test @testinferred cot(d) ≈ cot(t)
        @test @testinferred sinh(d) ≈ sinh(t)
        @test @testinferred cosh(d) ≈ cosh(t)
        @test @testinferred tanh(d) ≈ tanh(t)
        @test @testinferred coth(d) ≈ coth(t)
        @test @testinferred asin(d) ≈ asin(t)
        @test @testinferred acos(d) ≈ acos(t)
        @test @testinferred atan(d) ≈ atan(t)
        @test @testinferred acot(d) ≈ acot(t)
        @test @testinferred asinh(d) ≈ asinh(t)
        @test @testinferred acosh(one(d) + d) ≈ acosh(one(t) + t)
        @test @testinferred atanh(d) ≈ atanh(t)
        @test @testinferred acoth(one(t) + d) ≈ acoth(one(d) + t)
    end
end
