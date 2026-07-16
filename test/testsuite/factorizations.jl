# Factorization tests
# ===================

eltypes = (Float32, ComplexF64)

# eigenvalue decompositions
#--------------------------
@testsuite :factorizations "eigenvalue decomposition" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)

    for T in eltypes,
            t in (
                rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        d, v = @constinferred eig_full(t)
        @test t * v ≈ v * d

        d, v = @constinferred eig_full(t, DefaultAlgorithm())
        @test t * v ≈ v * d

        d′ = @constinferred eig_vals(t)
        @test d′ ≈ diagview(d)
        @test d′ isa TensorKit.SectorVector

        d′ = @constinferred eig_vals(t, DefaultAlgorithm())
        @test d′ ≈ diagview(d)
        @test d′ isa TensorKit.SectorVector

        d2 = @constinferred DiagonalTensorMap(d′)
        @test d2 ≈ d

        vdv = project_hermitian!(v' * v)
        @test @constinferred isposdef(vdv)
        t isa DiagonalTensorMap || @test !isposdef(t) # unlikely for non-hermitian map

        nvals = round(Int, dim(domain(t)) / 2)
        d, v = @constinferred eig_trunc(t; trunc = truncrank(nvals))
        @test t * v ≈ v * d
        @test abs(dim(domain(d)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

        d, v = @constinferred eig_trunc(t, DefaultAlgorithm(; trunc = truncrank(nvals)))
        @test t * v ≈ v * d
        @test abs(dim(domain(d)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

        t2 = @constinferred project_hermitian(t)
        D, V = eigen(t2)
        @test isisometric(V)
        D̃, Ṽ = @constinferred eigh_full(t2)
        @test D ≈ D̃
        @test V ≈ Ṽ

        D̃, Ṽ = @constinferred eigh_full(t2, DefaultAlgorithm())
        @test D ≈ D̃
        @test V ≈ Ṽ

        λ = minimum(real, diagview(D))
        @test cond(Ṽ) ≈ one(real(T))
        @test isposdef(t2) == isposdef(λ)
        @test isposdef(t2 - λ * one(t2) + 0.1 * one(t2))
        @test !isposdef(t2 - λ * one(t2) - 0.1 * one(t2))

        d, v = @constinferred eigh_full(t2)
        @test t2 * v ≈ v * d
        @test isunitary(v)

        d′ = @constinferred eigh_vals(t2)
        @test d′ ≈ diagview(d)
        @test d′ isa TensorKit.SectorVector

        d′ = @constinferred eigh_vals(t2, DefaultAlgorithm())
        @test d′ ≈ diagview(d)
        @test d′ isa TensorKit.SectorVector

        λ = minimum(real, diagview(d))
        @test cond(v) ≈ one(real(T))
        @test isposdef(t2) == isposdef(λ)
        @test isposdef(t2 - λ * one(t) + 0.1 * one(t2))
        @test !isposdef(t2 - λ * one(t) - 0.1 * one(t2))

        d, v = @constinferred eigh_trunc(t2; trunc = truncrank(nvals))
        @test t2 * v ≈ v * d
        @test abs(dim(domain(d)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

        d, v = @constinferred eigh_trunc(t2, DefaultAlgorithm(; trunc = truncrank(nvals)))
        @test t2 * v ≈ v * d
        @test abs(dim(domain(d)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)
    end
end

# QR and LQ decompositions
#--------------------------
@testsuite :factorizations "QR decomposition" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, W, W), rand(T, W, W)',
                rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)')',
                rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5)), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        Q, R = @constinferred qr_full(t)
        @test Q * R ≈ t
        @test isunitary(Q)

        Q, R = @constinferred qr_full(t, DefaultAlgorithm())
        @test Q * R ≈ t
        @test isunitary(Q)

        Q, R = @constinferred qr_compact(t)
        @test Q * R ≈ t
        @test isisometric(Q)

        Q, R = @constinferred qr_compact(t, DefaultAlgorithm())
        @test Q * R ≈ t
        @test isisometric(Q)

        Q, R = @constinferred left_orth(t)
        @test Q * R ≈ t
        @test isisometric(Q)

        Q, R = @constinferred left_orth(t, DefaultAlgorithm())
        @test Q * R ≈ t
        @test isisometric(Q)

        N = @constinferred qr_null(t)
        @test isisometric(N)
        @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

        N = @constinferred qr_null(t, DefaultAlgorithm())
        @test isisometric(N)
        @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

        N = @constinferred left_null(t)
        @test isisometric(N)
        @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

        N = @constinferred left_null(t, DefaultAlgorithm())
        @test isisometric(N)
        @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
    end

    # empty tensor
    for T in eltypes
        t = rand(T, V1 ⊗ V2, zerospace(V1))

        Q, R = @constinferred qr_full(t)
        @test Q * R ≈ t
        @test isunitary(Q)
        @test dim(R) == dim(t) == 0

        Q, R = @constinferred qr_full(t, DefaultAlgorithm())
        @test Q * R ≈ t
        @test isunitary(Q)
        @test dim(R) == dim(t) == 0

        Q, R = @constinferred qr_compact(t)
        @test Q * R ≈ t
        @test isisometric(Q)
        @test dim(Q) == dim(R) == dim(t)

        Q, R = @constinferred qr_compact(t, DefaultAlgorithm())
        @test Q * R ≈ t
        @test isisometric(Q)
        @test dim(Q) == dim(R) == dim(t)

        Q, R = @constinferred left_orth(t)
        @test Q * R ≈ t
        @test isisometric(Q)
        @test dim(Q) == dim(R) == dim(t)

        Q, R = @constinferred left_orth(t, DefaultAlgorithm())
        @test Q * R ≈ t
        @test isisometric(Q)
        @test dim(Q) == dim(R) == dim(t)

        N = @constinferred qr_null(t)
        @test isunitary(N)
        @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))

        N = @constinferred qr_null(t, DefaultAlgorithm())
        @test isunitary(N)
        @test norm(N' * t) ≈ 0 atol = 100 * eps(norm(t))
    end
end

@testsuite :factorizations "LQ decomposition" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, W, W), rand(T, W, W)',
                rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)')',
                rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5)), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        L, Q = @constinferred lq_full(t)
        @test L * Q ≈ t
        @test isunitary(Q)

        L, Q = @constinferred lq_full(t, DefaultAlgorithm())
        @test L * Q ≈ t
        @test isunitary(Q)

        L, Q = @constinferred lq_compact(t)
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)

        L, Q = @constinferred lq_compact(t, DefaultAlgorithm())
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)

        L, Q = @constinferred right_orth(t)
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)

        L, Q = @constinferred right_orth(t, DefaultAlgorithm())
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)

        Nᴴ = @constinferred lq_null(t)
        @test isisometric(Nᴴ; side = :right)
        @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))

        Nᴴ = @constinferred lq_null(t, DefaultAlgorithm())
        @test isisometric(Nᴴ; side = :right)
        @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
    end

    for T in eltypes
        # empty tensor
        t = rand(T, zerospace(V1), V1 ⊗ V2)

        L, Q = @constinferred lq_full(t)
        @test L * Q ≈ t
        @test isunitary(Q)
        @test dim(L) == dim(t) == 0

        L, Q = @constinferred lq_full(t, DefaultAlgorithm())
        @test L * Q ≈ t
        @test isunitary(Q)
        @test dim(L) == dim(t) == 0

        L, Q = @constinferred lq_compact(t)
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)
        @test dim(Q) == dim(L) == dim(t)

        L, Q = @constinferred lq_compact(t, DefaultAlgorithm())
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)
        @test dim(Q) == dim(L) == dim(t)

        L, Q = @constinferred right_orth(t)
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)
        @test dim(Q) == dim(L) == dim(t)

        L, Q = @constinferred right_orth(t, DefaultAlgorithm())
        @test L * Q ≈ t
        @test isisometric(Q; side = :right)
        @test dim(Q) == dim(L) == dim(t)

        Nᴴ = @constinferred lq_null(t)
        @test isunitary(Nᴴ)
        @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))

        Nᴴ = @constinferred lq_null(t, DefaultAlgorithm())
        @test isunitary(Nᴴ)
        @test norm(t * Nᴴ') ≈ 0 atol = 100 * eps(norm(t))
    end
end

# projections
#------------
@testsuite :factorizations "hermitian projections" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )
        normalize!(t)
        noisefactor = eps(real(T))^(3 / 4)

        th = (t + t') / 2
        ta = (t - t') / 2
        tc = copy(t)

        th′ = @constinferred project_hermitian(t)
        @test ishermitian(th′)
        @test th′ ≈ th

        th′ = @constinferred project_hermitian(t, DefaultAlgorithm())
        @test ishermitian(th′)
        @test th′ ≈ th

        @test t == tc
        th_approx = th + noisefactor * ta
        @test !ishermitian(th_approx) || (T <: Real && t isa DiagonalTensorMap)
        @test ishermitian(th_approx; atol = 10 * noisefactor)

        ta′ = project_antihermitian(t)
        @test isantihermitian(ta′)
        @test ta′ ≈ ta

        ta′ = @constinferred project_antihermitian(t, DefaultAlgorithm())
        @test isantihermitian(ta′)
        @test ta′ ≈ ta

        @test t == tc
        ta_approx = ta + noisefactor * th
        @test !isantihermitian(ta_approx)
        @test isantihermitian(ta_approx; atol = 10 * noisefactor) || (T <: Real && t isa DiagonalTensorMap)
    end

    @test_throws SpaceMismatch project_hermitian(rand(V1, V1^2))
    @test_throws SpaceMismatch project_antihermitian(rand(V1, V1^2))
    if V1 != V1'
        @test_throws SpaceMismatch project_hermitian(rand(V1, V1'))
        @test_throws SpaceMismatch project_antihermitian(rand(V1, V1'))
    end
end

@testsuite :factorizations "isometric projections" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, W, W), rand(T, W, W)', rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
            )
        t2 = project_isometric(t)
        @test isisometric(t2)
        t2′ = @constinferred project_isometric(t, DefaultAlgorithm())
        @test isisometric(t2′)
        @test t2′ * ((t2′)' * t) ≈ t

        t3 = project_isometric(t2)
        @test t3 ≈ t2 # stability of the projection
        @test t2 * (t2' * t) ≈ t

        tc = similar(t)
        t3 = @constinferred project_isometric!(copy!(tc, t), t2)
        @test t3 === t2
        @test isisometric(t2)

        # test that t2 is closer to A then any other isometry
        for k in 1:10
            δt = randn!(similar(t))
            t3 = project_isometric(t + δt / 100)
            @test norm(t - t3) > norm(t - t2)
        end
    end
end

# singular value decompositions
#------------------------------
@testsuite :factorizations "condition number and rank" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                randn(T, W, W), randn(T, W, W)',
                randn(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), randn(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                randn(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), randn(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        d1, d2 = dim(codomain(t)), dim(domain(t))
        r = rank(t)
        @test r ≈ min(d1, d2)
        @test typeof(r) == typeof(d1)
        M = left_null(t)
        @test @constinferred(rank(M)) + r ≈ d1
        Mᴴ = right_null(t)
        @test rank(Mᴴ) + r ≈ d2
    end
    for T in eltypes
        u = unitary(T, V1 ⊗ V2, V1 ⊗ V2)
        @test @constinferred(cond(u)) ≈ one(real(T))
        @test @constinferred(rank(u)) ≈ dim(V1 ⊗ V2)

        t = rand(T, zerospace(V1), W)
        @test rank(t) == 0
        t2 = rand(T, zerospace(V1) * zerospace(V2), zerospace(V1) * zerospace(V2))
        @test rank(t2) == 0
        @test cond(t2) == 0.0
    end
    for T in eltypes, t in (randn(T, W, W), randn(T, W, W)')
        project_hermitian!(t)
        vals = @constinferred eigh_vals(t)
        λmax = maximum(abs, vals)
        λmin = minimum(abs, vals)
        @test cond(t) ≈ λmax / λmin
    end
end

@testsuite :factorizations "polar decomposition" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, W, W),
                rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'),
                rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        @assert domain(t) ≾ codomain(t)
        w, p = @constinferred left_polar(t)
        @test w * p ≈ t
        @test isisometric(w)
        @test isposdef(p)

        w′, p′ = @constinferred left_polar(t, DefaultAlgorithm())
        @test w ≈ w′
        @test p ≈ p′

        w, p = @constinferred left_orth(t; alg = :polar)
        @test w * p ≈ t
        @test isisometric(w)
    end

    for T in eltypes,
            t in (
                rand(T, W, W),
                rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'),
                rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        @assert codomain(t) ≾ domain(t)
        p, wᴴ = @constinferred right_polar(t)
        @test p * wᴴ ≈ t
        @test isisometric(wᴴ; side = :right)
        @test isposdef(p)

        p′, wᴴ′ = @constinferred right_polar(t, DefaultAlgorithm())
        @test p′ ≈ p
        @test wᴴ′ ≈ wᴴ

        p, wᴴ = @constinferred right_orth(t; alg = :polar)
        @test p * wᴴ ≈ t
        @test isisometric(wᴴ; side = :right)
    end
end

@testsuite :factorizations "SVD" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, W, W), rand(T, W, W)',
                rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        u, s, vᴴ = @constinferred svd_full(t)
        @test u * s * vᴴ ≈ t
        @test isunitary(u)
        @test isunitary(vᴴ)

        u′, s′, vᴴ′ = @constinferred svd_full(t, DefaultAlgorithm())
        @test u ≈ u′
        @test s ≈ s′
        @test vᴴ ≈ vᴴ′

        u, s, vᴴ = @constinferred svd_compact(t)
        @test u * s * vᴴ ≈ t
        @test isisometric(u)
        @test isposdef(s)
        @test isisometric(vᴴ; side = :right)

        u′, s′, vᴴ′ = @constinferred svd_compact(t, DefaultAlgorithm())
        @test u ≈ u′
        @test s ≈ s′
        @test vᴴ ≈ vᴴ′

        s′ = @constinferred svd_vals(t)
        @test s′ ≈ diagview(s)
        @test s′ isa TensorKit.SectorVector

        s′ = @constinferred svd_vals(t, DefaultAlgorithm())
        @test s′ ≈ diagview(s)
        @test s′ isa TensorKit.SectorVector

        s2 = @constinferred DiagonalTensorMap(s′)
        @test s2 ≈ s

        v, c = @constinferred left_orth(t; alg = :svd)
        @test v * c ≈ t
        @test isisometric(v)

        c, vᴴ = @constinferred right_orth(t; alg = :svd)
        @test c * vᴴ ≈ t
        @test isisometric(vᴴ; side = :right)

        atol = norm(t) * defaulttol(T) # tol used by `:svd` left_null/right_null

        N = @constinferred left_null(t; alg = :svd)
        @test isisometric(N)
        @test norm(N' * t) ≈ 0 atol = atol

        N = @constinferred left_null(t; trunc = (; atol = 6 * atol))
        @test isisometric(N)
        @test norm(N' * t) ≈ 0 atol = 10 * atol

        Nᴴ = @constinferred right_null(t; alg = :svd)
        @test isisometric(Nᴴ; side = :right)
        @test norm(t * Nᴴ') ≈ 0 atol = atol

        Nᴴ = @constinferred right_null(t; trunc = (; atol = 6 * atol))
        @test isisometric(Nᴴ; side = :right)
        @test norm(t * Nᴴ') ≈ 0 atol = 10 * atol
    end

    # empty tensor
    for T in eltypes, t in (rand(T, W, zerospace(V1)), rand(T, zerospace(V1), W))
        U, S, Vᴴ = @constinferred svd_full(t)
        @test U * S * Vᴴ ≈ t
        @test isunitary(U)
        @test isunitary(Vᴴ)

        U, S, Vᴴ = @constinferred svd_full(t, DefaultAlgorithm())
        @test U * S * Vᴴ ≈ t
        @test isunitary(U)
        @test isunitary(Vᴴ)

        U, S, Vᴴ = @constinferred svd_compact(t)
        @test U * S * Vᴴ ≈ t
        @test dim(U) == dim(S) == dim(Vᴴ) == dim(t) == 0

        U, S, Vᴴ = @constinferred svd_compact(t, DefaultAlgorithm())
        @test U * S * Vᴴ ≈ t
        @test dim(U) == dim(S) == dim(Vᴴ) == dim(t) == 0
    end
end

@testsuite :factorizations "truncated SVD" V -> begin
    V1, V2, V3, V4, V5 = V
    W = V1 ⊗ V2 ⊗ V3
    Vd = fuse(V1 ⊗ V2)
    for T in eltypes,
            t in (
                rand(T, W, W), rand(T, W, W)',
                rand(T, (V1 ⊗ V2 ⊗ V3), (V4 ⊗ V5)'), rand(T, (V1 ⊗ V2)', (V3 ⊗ V4 ⊗ V5))',
                rand(T, (V1 ⊗ V2), (V3 ⊗ V4 ⊗ V5)'), rand(T, (V1 ⊗ V2 ⊗ V3)', (V4 ⊗ V5))',
                DiagonalTensorMap(rand(T, reduceddim(Vd)), Vd),
            )

        @constinferred normalize!(t)

        U, S, Vᴴ, ϵ = @constinferred svd_trunc(t; trunc = notrunc())
        @test U * S * Vᴴ ≈ t
        @test ϵ ≈ 0
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)

        U, S, Vᴴ, ϵ = @constinferred svd_trunc(t, DefaultAlgorithm(; trunc = notrunc()))
        @test U * S * Vᴴ ≈ t
        @test ϵ ≈ 0
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)

        # when rank of t is already smaller than truncrank
        t_rank = ceil(Int, min(dim(codomain(t)), dim(domain(t))))
        U, S, Vᴴ, ϵ = @constinferred svd_trunc(t; trunc = truncrank(t_rank + 1))
        @test U * S * Vᴴ ≈ t
        @test ϵ ≈ 0
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)

        U, S, Vᴴ, ϵ = @constinferred svd_trunc(t, DefaultAlgorithm(; trunc = truncrank(t_rank + 1)))
        @test U * S * Vᴴ ≈ t
        @test ϵ ≈ 0
        @test isisometric(U)
        @test isisometric(Vᴴ; side = :right)

        # dimension of S is a float for IsingBimodule
        nvals = round(Int, dim(domain(S)) / 2)
        trunc = truncrank(nvals)
        U1, S1, Vᴴ1, ϵ1 = @constinferred svd_trunc(t; trunc)
        @test t * Vᴴ1' ≈ U1 * S1
        @test isisometric(U1)
        @test isisometric(Vᴴ1; side = :right)
        @test norm(t - U1 * S1 * Vᴴ1) ≈ ϵ1 atol = eps(real(T))^(4 / 5)
        @test abs(dim(domain(S1)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

        λ = minimum(diagview(S1))
        trunc = trunctol(; atol = λ - 10eps(λ))
        U2, S2, Vᴴ2, ϵ2 = @constinferred svd_trunc(t; trunc)
        @test t * Vᴴ2' ≈ U2 * S2
        @test isisometric(U2)
        @test isisometric(Vᴴ2; side = :right)
        @test norm(t - U2 * S2 * Vᴴ2) ≈ ϵ2 atol = eps(real(T))^(4 / 5)
        @test minimum(diagview(S1)) >= λ
        @test U2 ≈ U1
        @test S2 ≈ S1
        @test Vᴴ2 ≈ Vᴴ1
        @test ϵ1 ≈ ϵ2

        trunc = truncspace(space(S2, 1))
        @test spacetype(typeof(trunc)) == spacetype(W)
        @test sectortype(trunc) == sectortype(W)
        U3, S3, Vᴴ3, ϵ3 = @constinferred svd_trunc(t; trunc)
        @test t * Vᴴ3' ≈ U3 * S3
        @test isisometric(U3)
        @test isisometric(Vᴴ3; side = :right)
        @test norm(t - U3 * S3 * Vᴴ3) ≈ ϵ3 atol = eps(real(T))^(4 / 5)
        @test space(S3, 1) ≾ space(S2, 1)

        for trunc in (truncerror(; atol = ϵ2), truncerror(; rtol = ϵ2 / norm(t)))
            U4, S4, Vᴴ4, ϵ4 = @constinferred svd_trunc(t; trunc)
            @test t * Vᴴ4' ≈ U4 * S4
            @test isisometric(U4)
            @test isisometric(Vᴴ4; side = :right)
            @test norm(t - U4 * S4 * Vᴴ4) ≈ ϵ4 atol = eps(real(T))^(4 / 5)
            @test ϵ4 ≤ ϵ2
        end

        trunc = truncrank(nvals) & trunctol(; atol = λ - 10eps(λ))
        U5, S5, Vᴴ5, ϵ5 = @constinferred svd_trunc(t; trunc)
        @test t * Vᴴ5' ≈ U5 * S5
        @test isisometric(U5)
        @test isisometric(Vᴴ5; side = :right)
        @test norm(t - U5 * S5 * Vᴴ5) ≈ ϵ5 atol = eps(real(T))^(4 / 5)
        @test minimum(diagview(S5)) >= λ
        @test abs(dim(domain(S5)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)

        trunc = truncrank(nvals) | trunctol(; atol = λ - 10eps(λ))
        U5, S5, Vᴴ5, ϵ5 = @constinferred svd_trunc(t; trunc)
        @test t * Vᴴ5' ≈ U5 * S5
        @test isisometric(U5)
        @test isisometric(Vᴴ5; side = :right)
        @test norm(t - U5 * S5 * Vᴴ5) ≈ ϵ5 atol = eps(real(T))^(4 / 5)
        @test minimum(diagview(S5)) >= λ
        @test abs(dim(domain(S5)) - nvals) ≤ maximum(c -> blockdim(domain(t), c), blocksectors(t); init = 1)
    end
end
