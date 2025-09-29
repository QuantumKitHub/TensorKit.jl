using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences: FiniteDifferences
using Random
using LinearAlgebra
using Zygote
using MatrixAlgebraKit
using MatrixAlgebraKit: LAPACK_HouseholderQR, LAPACK_HouseholderLQ, diagview

const _repartition = @static if isdefined(Base, :get_extension)
    Base.get_extension(TensorKit, :TensorKitChainRulesCoreExt)._repartition
else
    TensorKit.TensorKitChainRulesCoreExt._repartition
end

# Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return randn!(similar(x))
end
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::DiagonalTensorMap)
    V = x.domain
    return DiagonalTensorMap(randn(eltype(x), reduceddim(V)), V)
end
ChainRulesTestUtils.rand_tangent(::AbstractRNG, ::VectorSpace) = NoTangent()
function ChainRulesTestUtils.test_approx(actual::AbstractTensorMap,
                                         expected::AbstractTensorMap, msg=""; kwargs...)
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
    return nothing
end

# make sure that norms are computed correctly:
function FiniteDifferences.to_vec(t::TensorKit.SectorDict)
    T = scalartype(valtype(t))
    vec = mapreduce(vcat, t; init=T[]) do (c, b)
        return reshape(b, :) .* sqrt(dim(c))
    end
    vec_real = T <: Real ? vec : collect(reinterpret(real(T), vec))

    function from_vec(x_real)
        x = T <: Real ? x_real : reinterpret(T, x_real)
        ctr = 0
        return TensorKit.SectorDict(c => (n = length(b);
                                          b′ = reshape(view(x, ctr .+ (1:n)), size(b)) ./
                                               sqrt(dim(c));
                                          ctr += n;
                                          b′)
                                    for (c, b) in t)
    end
    return vec_real, from_vec
end

# Float32 and finite differences don't mix well
precision(::Type{<:Union{Float32,Complex{Float32}}}) = 1.0e-2
precision(::Type{<:Union{Float64,Complex{Float64}}}) = 1.0e-6

function randindextuple(N::Int, k::Int=rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end

function test_ad_rrule(f, args...; check_inferred=false, kwargs...)
    test_rrule(Zygote.ZygoteRuleConfig(), f, args...;
               rrule_f=rrule_via_ad, check_inferred, kwargs...)
    return nothing
end

# Gauge fixing tangents
# ---------------------
function remove_qrgauge_dependence!(ΔQ, t, Q)
    for (c, b) in blocks(ΔQ)
        m, n = size(block(t, c))
        minmn = min(m, n)
        Qc = block(Q, c)
        Q1 = view(Qc, 1:m, 1:minmn)
        ΔQ2 = view(b, :, (minmn + 1):m)
        mul!(ΔQ2, Q1, Q1' * ΔQ2)
    end
    return ΔQ
end

function remove_lqgauge_dependence!(ΔQ, t, Q)
    for (c, b) in blocks(ΔQ)
        m, n = size(block(t, c))
        minmn = min(m, n)
        Qc = block(Q, c)
        Q1 = view(Qc, 1:minmn, 1:n)
        ΔQ2 = view(b, (minmn + 1):n, :)
        mul!(ΔQ2, ΔQ2 * Q1', Q1)
    end
    return ΔQ
end
function remove_eiggauge_dependence!(ΔV, D, V;
                                     degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(D))
    gaugepart = V' * ΔV
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_eighgauge_dependence!(ΔV, D, V;
                                      degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(D))
    gaugepart = V' * ΔV
    gaugepart = (gaugepart - gaugepart') / 2
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V / (V' * V), gaugepart, -1, 1)
    return ΔV
end
function remove_svdgauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ;
                                     degeneracy_atol=MatrixAlgebraKit.default_pullback_gaugetol(S))
    gaugepart = U' * ΔU + Vᴴ * ΔVᴴ'
    gaugepart = (gaugepart - gaugepart') / 2
    for (c, b) in blocks(gaugepart)
        Sd = diagview(block(S, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Sc)) .- diagview(Sc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Sd[i] - Sd[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔU, U, gaugepart, -1, 1)
    return ΔU, ΔVᴴ
end

project_hermitian(A) = (A + A') / 2

# Tests
# -----

ChainRulesTestUtils.test_method_tables()

spacelist = try
    if ENV["CI"] == "true"
        println("Detected running on CI")
        if Sys.iswindows()
            (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂)
        elseif Sys.isapple()
            (Vtr, Vℤ₃, VfU₁, VfSU₂)
        else
            (Vtr, VU₁, VCU₁, VfSU₂, Vfib)
        end
    else
        (Vtr, Vℤ₃, VU₁, VfU₁, VSU₂, VfSU₂, Vfib)
    end
catch
    (Vtr, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, Vfib)
end

for V in spacelist
    I = sectortype(eltype(V))
    Istr = TensorKit.type_repr(I)
    eltypes = isreal(sectortype(eltype(V))) ? (Float64, ComplexF64) : (ComplexF64,)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
    println("---------------------------------------")
    println("Auto-diff with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "AD with symmetry $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2
        @timedtestset "Basic utility" begin
            T1 = randn(Float64, V[1] ⊗ V[2] ← V[3] ⊗ V[4])
            T2 = randn(ComplexF64, V[1] ⊗ V[2] ← V[3] ⊗ V[4])

            P1 = ProjectTo(T1)
            @test P1(T1) == T1
            @test P1(T2) == real(T2)

            test_rrule(copy, T1)
            test_rrule(copy, T2)
            test_rrule(TensorKit.copy_oftype, T1, ComplexF64)
            if symmetricbraiding
                test_rrule(TensorKit.permutedcopy_oftype, T1, ComplexF64,
                           ((3, 1), (2, 4)))

                test_rrule(convert, Array, T1)
                test_rrule(TensorMap, convert(Array, T1), codomain(T1), domain(T1);
                           fkwargs=(; tol=Inf))
            end

            test_rrule(Base.getproperty, T1, :data)
            test_rrule(TensorMap{scalartype(T1)}, T1.data, T1.space)
            test_rrule(Base.getproperty, T2, :data)
            test_rrule(TensorMap{scalartype(T2)}, T2.data, T2.space)
        end

        @timedtestset "Basic utility (DiagonalTensor)" begin
            for v in V
                rdim = reduceddim(v)
                D1 = DiagonalTensorMap(randn(rdim), v)
                D2 = DiagonalTensorMap(randn(rdim), v)
                D = D1 + im * D2
                T1 = TensorMap(D1)
                T2 = TensorMap(D2)
                T = T1 + im * T2

                # real -> real
                P1 = ProjectTo(D1)
                @test P1(D1) == D1
                @test P1(T1) == D1

                # complex -> complex
                P2 = ProjectTo(D)
                @test P2(D) == D
                @test P2(T) == D

                # real -> complex
                @test P2(D1) == D1 + 0 * im * D1
                @test P2(T1) == D1 + 0 * im * D1

                # complex -> real
                @test P1(D) == D1
                @test P1(T) == D1

                test_rrule(DiagonalTensorMap, D1.data, D1.domain)
                test_rrule(DiagonalTensorMap, D.data, D.domain)
                test_rrule(Base.getproperty, D, :data)
                test_rrule(Base.getproperty, D1, :data)

                test_rrule(DiagonalTensorMap, rand!(T1))
                test_rrule(DiagonalTensorMap, randn!(T))
            end
        end

        @timedtestset "Basic Linear Algebra with scalartype $T" for T in eltypes
            A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            B = randn(T, space(A))

            test_rrule(real, A)
            test_rrule(imag, A)

            test_rrule(+, A, B)
            test_rrule(-, A)
            test_rrule(-, A, B)

            α = randn(T)
            test_rrule(*, α, A)
            test_rrule(*, A, α)

            C = randn(T, domain(A), codomain(A))
            test_rrule(*, A, C)

            symmetricbraiding && test_rrule(permute, A, ((1, 3, 2), (5, 4)))
            test_rrule(twist, A, 1)
            test_rrule(twist, A, [1, 3])

            test_rrule(flip, A, 1)
            test_rrule(flip, A, [1, 3, 4])

            D = randn(T, V[1] ⊗ V[2] ← V[3])
            E = randn(T, V[4] ← V[5])
            symmetricbraiding && test_rrule(⊗, D, E)
        end

        @timedtestset "Linear Algebra part II with scalartype $T" for T in eltypes
            for i in 1:3
                E = randn(T, ⊗(V[1:i]...) ← ⊗(V[1:i]...))
                test_rrule(LinearAlgebra.tr, E)
                test_rrule(exp, E; check_inferred=false)
                test_rrule(inv, E)
            end

            A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            test_rrule(LinearAlgebra.adjoint, A)
            test_rrule(LinearAlgebra.norm, A, 2)

            B = randn(T, space(A))
            test_rrule(LinearAlgebra.dot, A, B)
        end

        @timedtestset "Matrix functions ($T)" for T in eltypes
            for f in (sqrt, exp)
                check_inferred = false # !(T <: Real) # not type-stable for real functions
                t1 = randn(T, V[1] ← V[1])
                t2 = randn(T, V[2] ← V[2])
                d = DiagonalTensorMap{T}(undef, V[1])
                (T <: Real && f === sqrt) ? randexp!(d.data) : randn!(d.data)
                d2 = DiagonalTensorMap{T}(undef, V[1])
                (T <: Real && f === sqrt) ? randexp!(d2.data) : randn!(d2.data)
                test_rrule(f, t1; rrule_f=Zygote.rrule_via_ad, check_inferred)
                test_rrule(f, t2; rrule_f=Zygote.rrule_via_ad, check_inferred)
                test_rrule(f, d; check_inferred, output_tangent=d2)
            end
        end

        symmetricbraiding &&
            @timedtestset "TensorOperations with scalartype $T" for T in eltypes
                atol = precision(T)
                rtol = precision(T)

                @timedtestset "tensortrace!" begin
                    for _ in 1:5
                        k1 = rand(0:3)
                        k2 = k1 == 3 ? 1 : rand(1:2)
                        V1 = map(v -> rand(Bool) ? v' : v, rand(V, k1))
                        V2 = map(v -> rand(Bool) ? v' : v, rand(V, k2))

                        (_p, _q) = randindextuple(k1 + 2 * k2, k1)
                        p = _repartition(_p, rand(0:k1))
                        q = _repartition(_q, k2)
                        ip = _repartition(invperm(linearize((_p, _q))),
                                          rand(0:(k1 + 2 * k2)))
                        A = randn(T, permute(prod(V1) ⊗ prod(V2) ← prod(V2), ip))

                        α = randn(T)
                        β = randn(T)
                        for conjA in (false, true)
                            C = randn!(TensorOperations.tensoralloc_add(T, A, p, conjA,
                                                                        Val(false)))
                            test_rrule(tensortrace!, C, A, p, q, conjA, α, β; atol, rtol)
                        end
                    end
                end

                @timedtestset "tensoradd!" begin
                    A = randn(T, V[1] ⊗ V[2] ⊗ V[3] ← V[4] ⊗ V[5])
                    α = randn(T)
                    β = randn(T)

                    # repeat a couple times to get some distribution of arrows
                    for _ in 1:5
                        p = randindextuple(length(V))

                        C1 = randn!(TensorOperations.tensoralloc_add(T, A, p, false,
                                                                     Val(false)))
                        test_rrule(tensoradd!, C1, A, p, false, α, β; atol, rtol)

                        C2 = randn!(TensorOperations.tensoralloc_add(T, A, p, true,
                                                                     Val(false)))
                        test_rrule(tensoradd!, C2, A, p, true, α, β; atol, rtol)

                        A = rand(Bool) ? C1 : C2
                    end
                end

                @timedtestset "tensorcontract!" begin
                    for _ in 1:5
                        d = 0
                        local V1, V2, V3
                        # retry a couple times to make sure there are at least some nonzero elements
                        for _ in 1:10
                            k1 = rand(0:3)
                            k2 = rand(0:2)
                            k3 = rand(0:2)
                            V1 = prod(v -> rand(Bool) ? v' : v, rand(V, k1); init=one(V[1]))
                            V2 = prod(v -> rand(Bool) ? v' : v, rand(V, k2); init=one(V[1]))
                            V3 = prod(v -> rand(Bool) ? v' : v, rand(V, k3); init=one(V[1]))
                            d = min(dim(V1 ← V2), dim(V1' ← V2), dim(V2 ← V3),
                                    dim(V2' ← V3))
                            d > 0 && break
                        end
                        ipA = randindextuple(length(V1) + length(V2))
                        pA = _repartition(invperm(linearize(ipA)), length(V1))
                        ipB = randindextuple(length(V2) + length(V3))
                        pB = _repartition(invperm(linearize(ipB)), length(V2))
                        pAB = randindextuple(length(V1) + length(V3))

                        α = randn(T)
                        β = randn(T)
                        V2_conj = prod(conj, V2; init=one(V[1]))

                        for conjA in (false, true), conjB in (false, true)
                            A = randn(T, permute(V1 ← (conjA ? V2_conj : V2), ipA))
                            B = randn(T, permute((conjB ? V2_conj : V2) ← V3, ipB))
                            C = randn!(TensorOperations.tensoralloc_contract(T, A, pA,
                                                                             conjA,
                                                                             B, pB, conjB,
                                                                             pAB,
                                                                             Val(false)))
                            test_rrule(tensorcontract!, C,
                                       A, pA, conjA, B, pB, conjB, pAB,
                                       α, β; atol, rtol)
                        end
                    end
                end

                @timedtestset "tensorscalar" begin
                    A = randn(T, ProductSpace{typeof(V[1]),0}())
                    test_rrule(tensorscalar, A)
                end
            end

        @timedtestset "Factorizations" begin
            @testset "QR" begin
                for T in eltypes,
                    t in (randn(T, W, W), randn(T, W, W)',
                          randn(T, W, V1), randn(T, V1, W),
                          randn(T, W, V1)', randn(T, V1, W)',
                          DiagonalTensorMap(randn(T, reduceddim(V1)), V1))

                    atol = rtol = precision(T) * dim(space(t))
                    fkwargs = (; positive=true) # make FiniteDifferences happy

                    test_ad_rrule(qr_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(first ∘ qr_compact, t; fkwargs, atol, rtol,)
                    test_ad_rrule(last ∘ qr_compact, t; fkwargs, atol, rtol,)

                    # qr_full/qr_null requires being careful with gauges
                    Q, R = qr_full(t)
                    ΔQ = rand_tangent(Q)
                    ΔR = rand_tangent(R)

                    if fuse(domain(t)) ≺ fuse(codomain(t))
                        _, full_pb = Zygote.pullback(qr_full, t)
                        @test_logs (:warn, r"^`qr") match_mode = :any full_pb((ΔQ, ΔR))
                    end

                    remove_qrgauge_dependence!(ΔQ, t, Q)

                    test_ad_rrule(qr_full, t; fkwargs, atol, rtol, output_tangent=(ΔQ, ΔR))
                    test_ad_rrule(first ∘ qr_full, t; fkwargs, atol, rtol,
                                  output_tangent=ΔQ)
                    test_ad_rrule(last ∘ qr_full, t; fkwargs, atol, rtol, output_tangent=ΔR)

                    # TODO: figure out the following:
                    # N = qr_null(t)
                    # ΔN = Q * rand(T, domain(Q) ← domain(N))
                    # test_ad_rrule(qr_null, t; fkwargs, atol, rtol, output_tangent=ΔN)

                    # if fuse(domain(t)) ≺ fuse(codomain(t))
                    #     _, null_pb = Zygote.pullback(qr_null, t)
                    #     @test_logs (:warn, r"^`qr") match_mode = :any null_pb(rand_tangent(N))
                    # end
                end
            end

            @testset "LQ" begin
                for T in eltypes,
                    t in (randn(T, W, W), randn(T, W, W)',
                          randn(T, W, V1), randn(T, V1, W),
                          randn(T, W, V1)', randn(T, V1, W)',
                          DiagonalTensorMap(randn(T, reduceddim(V1)), V1))

                    atol = rtol = precision(T) * dim(space(t))
                    fkwargs = (; positive=true) # make FiniteDifferences happy

                    test_ad_rrule(lq_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(first ∘ lq_compact, t; fkwargs, atol, rtol)
                    test_ad_rrule(last ∘ lq_compact, t; fkwargs, atol, rtol)

                    # lq_full/lq_null requires being careful with gauges
                    L, Q = lq_full(t)
                    ΔQ = rand_tangent(Q)
                    ΔL = rand_tangent(L)

                    if fuse(codomain(t)) ≺ fuse(domain(t))
                        _, full_pb = Zygote.pullback(lq_full, t)
                        # broken due to typo in MAK
                        # @test_logs (:warn, r"^`lq") match_mode = :any full_pb((ΔL, ΔQ))
                    end

                    remove_lqgauge_dependence!(ΔQ, t, Q)

                    test_ad_rrule(lq_full, t; fkwargs, atol, rtol, output_tangent=(ΔL, ΔQ))
                    test_ad_rrule(first ∘ lq_full, t; fkwargs, atol, rtol,
                                  output_tangent=ΔL)
                    test_ad_rrule(last ∘ lq_full, t; fkwargs, atol, rtol, output_tangent=ΔQ)

                    # TODO: figure out the following
                    # Nᴴ = lq_null(t)
                    # ΔN = rand(T, codomain(Nᴴ) ← codomain(Q)) * Q
                    # test_ad_rrule(lq_null, t; fkwargs, atol, rtol, output_tangent=Nᴴ)

                    # if fuse(codomain(t)) ≺ fuse(domain(t))
                    #     _, null_pb = Zygote.pullback(lq_null, t)
                    #     # broken due to typo in MAK
                    #     # @test_logs (:warn, r"^`lq") match_mode = :any null_pb(rand_tangent(Nᴴ))
                    # end
                end
            end

            @testset "Eigenvalue decomposition" begin
                for T in eltypes,
                    t in (rand(T, V1, V1), rand(T, W, W), rand(T, W, W)',
                          DiagonalTensorMap(rand(T, reduceddim(V1)), V1))

                    atol = rtol = precision(T) * dim(space(t))

                    d, v = eig_full(t)
                    Δv = rand_tangent(v)
                    Δd = rand_tangent(d)
                    Δd2 = randn!(similar(d, space(d)))
                    remove_eiggauge_dependence!(Δv, d, v)

                    test_ad_rrule(eig_full, t; output_tangent=(Δd, Δv), atol, rtol)
                    test_ad_rrule(first ∘ eig_full, t; output_tangent=Δd, atol, rtol)
                    test_ad_rrule(last ∘ eig_full, t; output_tangent=Δv, atol, rtol)
                    test_ad_rrule(eig_full, t; output_tangent=(Δd2, Δv), atol, rtol)

                    add!(t, t')
                    d, v = eigh_full(t)
                    Δv = rand_tangent(v)
                    Δd = rand_tangent(d)
                    Δd2 = randn!(similar(d, space(d)))
                    remove_eighgauge_dependence!(Δv, d, v)

                    # necessary for FiniteDifferences to not complain
                    eigh_full′ = eigh_full ∘ project_hermitian

                    test_ad_rrule(eigh_full′, t; output_tangent=(Δd, Δv), atol, rtol)
                    test_ad_rrule(first ∘ eigh_full′, t; output_tangent=Δd, atol, rtol)
                    test_ad_rrule(last ∘ eigh_full′, t; output_tangent=Δv, atol, rtol)
                    test_ad_rrule(eigh_full′, t; output_tangent=(Δd2, Δv), atol, rtol)
                end
            end

            @testset "Singular value decomposition" begin
                for T in eltypes,
                    t in (randn(T, V1, V1), randn(T, W, W), randn(T, W, W))
                    # TODO: fix diagonaltensormap case
                    #   DiagonalTensorMap(rand(T, reduceddim(V1)), V1))

                    atol = rtol = degeneracy_atol = precision(T) * dim(space(t))
                    USVᴴ = svd_compact(t)
                    ΔU, ΔS, ΔVᴴ = rand_tangent.(USVᴴ)
                    ΔS2 = randn!(similar(ΔS, space(ΔS)))
                    ΔU, ΔVᴴ = remove_svdgauge_dependence!(ΔU, ΔVᴴ, USVᴴ...; degeneracy_atol)

                    test_ad_rrule(svd_full, t; output_tangent=(ΔU, ΔS, ΔVᴴ), atol, rtol)
                    test_ad_rrule(svd_full, t; output_tangent=(ΔU, ΔS2, ΔVᴴ), atol, rtol)
                    test_ad_rrule(svd_compact, t; output_tangent=(ΔU, ΔS, ΔVᴴ), atol, rtol)
                    test_ad_rrule(svd_compact, t; output_tangent=(ΔU, ΔS2, ΔVᴴ), atol, rtol)

                    # TODO: I'm not sure how to properly test with spaces that might change
                    # with the finite-difference methods, as then the jacobian is ill-defined.

                    trunc = truncrank(round(Int, min(dim(domain(t)), dim(codomain(t))) ÷ 2))
                    USVᴴ_trunc = svd_trunc(t; trunc)
                    ΔUSVᴴ_trunc = rand_tangent.(USVᴴ_trunc)
                    remove_svdgauge_dependence!(ΔUSVᴴ_trunc[1], ΔUSVᴴ_trunc[3],
                                                USVᴴ_trunc...;
                                                degeneracy_atol)
                    # test_ad_rrule(svd_trunc, t;
                    #               fkwargs=(; trunc), output_tangent=ΔUSVᴴ_trunc, atol, rtol)

                    trunc = truncspace(space(USVᴴ_trunc[2], 1))
                    USVᴴ_trunc = svd_trunc(t; trunc)
                    ΔUSVᴴ_trunc = rand_tangent.(USVᴴ_trunc)
                    remove_svdgauge_dependence!(ΔUSVᴴ_trunc[1], ΔUSVᴴ_trunc[3],
                                                USVᴴ_trunc...;
                                                degeneracy_atol)
                    test_ad_rrule(svd_trunc, t;
                                  fkwargs=(; trunc), output_tangent=ΔUSVᴴ_trunc, atol, rtol)

                    # ϵ = norm(*(USVᴴ_trunc...) - t)
                    # trunc = truncerror(; atol=ϵ)
                    # USVᴴ_trunc = svd_trunc(t; trunc)
                    # ΔUSVᴴ_trunc = rand_tangent.(USVᴴ_trunc)
                    # remove_svdgauge_dependence!(ΔUSVᴴ_trunc[1], ΔUSVᴴ_trunc[3], USVᴴ_trunc...;
                    #                            degeneracy_atol)
                    # test_ad_rrule(svd_trunc, t;
                    #               fkwargs=(; trunc), output_tangent=ΔUSVᴴ_trunc, atol, rtol)

                    tol = minimum(((c, b),) -> minimum(diagview(b)), blocks(USVᴴ_trunc[2]))
                    trunc = trunctol(; atol=10 * tol)
                    USVᴴ_trunc = svd_trunc(t; trunc)
                    ΔUSVᴴ_trunc = rand_tangent.(USVᴴ_trunc)
                    remove_svdgauge_dependence!(ΔUSVᴴ_trunc[1], ΔUSVᴴ_trunc[3],
                                                USVᴴ_trunc...;
                                                degeneracy_atol)
                    test_ad_rrule(svd_trunc, t;
                                  fkwargs=(; trunc), output_tangent=ΔUSVᴴ_trunc, atol, rtol)
                end
            end

            # let D = LinearAlgebra.eigvals(C)
            #     ΔD = diag(randn(complex(scalartype(C)), space(C)))
            #     test_rrule(LinearAlgebra.eigvals, C; atol, output_tangent=ΔD,
            #                fkwargs=(; sortby=nothing))
            # end

            # let S = LinearAlgebra.svdvals(C)
            #     ΔS = diag(randn(real(scalartype(C)), space(C)))
            #     test_rrule(LinearAlgebra.svdvals, C; atol, output_tangent=ΔS)
            # end
        end
    end
end
