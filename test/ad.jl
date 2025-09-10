using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences: FiniteDifferences
using Random
using LinearAlgebra
using Zygote

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
precision(::Type{<:Union{Float32,Complex{Float32}}}) = 1e-2
precision(::Type{<:Union{Float64,Complex{Float64}}}) = 1e-6

function randindextuple(N::Int, k::Int=rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end

# rrules for functions that destroy inputs
# ----------------------------------------
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd), args...; kwargs...)
    return ChainRulesCore.rrule(tsvd!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(LinearAlgebra.svdvals), args...; kwargs...)
    return ChainRulesCore.rrule(svdvals!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.eig), args...; kwargs...)
    return ChainRulesCore.rrule(eig!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.eigh), args...; kwargs...)
    return ChainRulesCore.rrule(eigh!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(LinearAlgebra.eigvals), args...; kwargs...)
    return ChainRulesCore.rrule(eigvals!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.leftorth), args...; kwargs...)
    return ChainRulesCore.rrule(leftorth!, args...; kwargs...)
end
function ChainRulesCore.rrule(::typeof(TensorKit.rightorth), args...; kwargs...)
    return ChainRulesCore.rrule(rightorth!, args...; kwargs...)
end

# eigh′: make argument of eigh explicitly Hermitian
#---------------------------------------------------
eigh′(t::AbstractTensorMap) = eigh(scale!(t + t', 1 / 2))

function ChainRulesCore.rrule(::typeof(eigh′), args...; kwargs...)
    return ChainRulesCore.rrule(eigh!, args...; kwargs...)
end

# complex-valued svd?
# -------------------
function remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
    # simple implementation, assumes no degeneracies or zeros in singular values
    gaugepart = U' * ΔU + V * ΔV'
    for (c, b) in blocks(gaugepart)
        mul!(block(ΔU, c), block(U, c), Diagonal(imag(diag(b))), -im, 1)
    end
    return ΔU, ΔV
end

# Tests
# -----

ChainRulesTestUtils.test_method_tables()

Vlist = ((ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
         (ℂ[Z2Irrep](0 => 1, 1 => 1),
          ℂ[Z2Irrep](0 => 1, 1 => 2)',
          ℂ[Z2Irrep](0 => 3, 1 => 2)',
          ℂ[Z2Irrep](0 => 2, 1 => 3),
          ℂ[Z2Irrep](0 => 2, 1 => 2)),
         (ℂ[FermionParity](0 => 1, 1 => 1),
          ℂ[FermionParity](0 => 1, 1 => 2)',
          ℂ[FermionParity](0 => 2, 1 => 2)',
          ℂ[FermionParity](0 => 2, 1 => 3),
          ℂ[FermionParity](0 => 2, 1 => 2)),
         (ℂ[U1Irrep](0 => 2, 1 => 1, -1 => 1),
          ℂ[U1Irrep](0 => 3, 1 => 1, -1 => 1),
          ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
          ℂ[U1Irrep](0 => 1, 1 => 1, -1 => 2),
          ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 1)'),
         (ℂ[SU2Irrep](0 => 2, 1 // 2 => 1),
          ℂ[SU2Irrep](0 => 1, 1 => 1),
          ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
          ℂ[SU2Irrep](1 // 2 => 2),
          ℂ[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)'),
         (ℂ[FibonacciAnyon](:I => 1, :τ => 1),
          ℂ[FibonacciAnyon](:I => 1, :τ => 2)',
          ℂ[FibonacciAnyon](:I => 3, :τ => 2)',
          ℂ[FibonacciAnyon](:I => 2, :τ => 3),
          ℂ[FibonacciAnyon](:I => 2, :τ => 2)))

@timedtestset "Automatic Differentiation with spacetype $(TensorKit.type_repr(eltype(V)))" verbose = true for V in
                                                                                                              Vlist
    eltypes = isreal(sectortype(eltype(V))) ? (Float64, ComplexF64) : (ComplexF64,)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

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
            test_rrule(TensorKit.permutedcopy_oftype, T1, ComplexF64, ((3, 1), (2, 4)))

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
                    ip = _repartition(invperm(linearize((_p, _q))), rand(0:(k1 + 2 * k2)))
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

                    C2 = randn!(TensorOperations.tensoralloc_add(T, A, p, true, Val(false)))
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
                        d = min(dim(V1 ← V2), dim(V1' ← V2), dim(V2 ← V3), dim(V2' ← V3))
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
                                                                         B, pB, conjB, pAB,
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

    @timedtestset "Factorizations with scalartype $T" for T in eltypes
        A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        B = randn(T, space(A)')
        C = randn(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2])
        H = randn(T, V[3] ⊗ V[4] ← V[3] ⊗ V[4])
        H = (H + H') / 2
        atol = precision(T)

        for alg in (TensorKit.QR(), TensorKit.QRpos())
            test_rrule(leftorth, A; fkwargs=(; alg=alg), atol)
            test_rrule(leftorth, B; fkwargs=(; alg=alg), atol)
            test_rrule(leftorth, C; fkwargs=(; alg=alg), atol)
        end

        for alg in (TensorKit.LQ(), TensorKit.LQpos())
            test_rrule(rightorth, A; fkwargs=(; alg=alg), atol)
            test_rrule(rightorth, B; fkwargs=(; alg=alg), atol)
            test_rrule(rightorth, C; fkwargs=(; alg=alg), atol)
        end

        let (D, V) = eig(C)
            ΔD = randn(scalartype(D), space(D))
            ΔV = randn(scalartype(V), space(V))
            gaugepart = V' * ΔV
            for (c, b) in blocks(gaugepart)
                mul!(block(ΔV, c), inv(block(V, c))', Diagonal(diag(b)), -1, 1)
            end
            test_rrule(eig, C; atol, output_tangent=(ΔD, ΔV))
        end

        let (D, U) = eigh′(H)
            ΔD = randn(scalartype(D), space(D))
            ΔU = randn(scalartype(U), space(U))
            if T <: Complex
                gaugepart = U' * ΔU
                for (c, b) in blocks(gaugepart)
                    mul!(block(ΔU, c), block(U, c), Diagonal(imag(diag(b))), -im, 1)
                end
            end
            test_rrule(eigh′, H; atol, output_tangent=(ΔD, ΔU))
        end

        let (U, S, V) = tsvd(A)
            ΔU = randn(scalartype(U), space(U))
            ΔS = randn(scalartype(S), space(S))
            ΔV = randn(scalartype(V), space(V))
            if T <: Complex # remove gauge dependent components
                gaugepart = U' * ΔU + V * ΔV'
                for (c, b) in blocks(gaugepart)
                    mul!(block(ΔU, c), block(U, c), Diagonal(imag(diag(b))), -im, 1)
                end
            end
            test_rrule(tsvd, A; atol, output_tangent=(ΔU, ΔS, ΔV))

            allS = mapreduce(x -> diag(x[2]), vcat, blocks(S))
            truncval = (maximum(allS) + minimum(allS)) / 2
            U, S, V = tsvd(A; trunc=truncerr(truncval))
            ΔU = randn(scalartype(U), space(U))
            ΔS = randn(scalartype(S), space(S))
            ΔV = randn(scalartype(V), space(V))
            T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
            test_rrule(tsvd, A; atol, output_tangent=(ΔU, ΔS, ΔV),
                       fkwargs=(; trunc=truncerr(truncval)))
        end

        let (U, S, V) = tsvd(B)
            ΔU = randn(scalartype(U), space(U))
            ΔS = randn(scalartype(S), space(S))
            ΔV = randn(scalartype(V), space(V))
            T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
            test_rrule(tsvd, B; atol, output_tangent=(ΔU, ΔS, ΔV))

            Vtrunc = spacetype(S)(TensorKit.SectorDict(c => ceil(Int, size(b, 1) / 2)
                                                       for (c, b) in blocks(S)))

            U, S, V = tsvd(B; trunc=truncspace(Vtrunc))
            ΔU = randn(scalartype(U), space(U))
            ΔS = randn(scalartype(S), space(S))
            ΔV = randn(scalartype(V), space(V))
            T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
            test_rrule(tsvd, B; atol, output_tangent=(ΔU, ΔS, ΔV),
                       fkwargs=(; trunc=truncspace(Vtrunc)))
        end

        let (U, S, V) = tsvd(C)
            ΔU = randn(scalartype(U), space(U))
            ΔS = randn(scalartype(S), space(S))
            ΔV = randn(scalartype(V), space(V))
            T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
            test_rrule(tsvd, C; atol, output_tangent=(ΔU, ΔS, ΔV))

            c, = argmax(x -> sqrt(dim(x[1])) * maximum(diag(x[2])), blocks(S))
            trunc = truncdim(round(Int, 2 * dim(c)))
            U, S, V = tsvd(C; trunc)
            ΔU = randn(scalartype(U), space(U))
            ΔS = randn(scalartype(S), space(S))
            ΔV = randn(scalartype(V), space(V))
            T <: Complex && remove_svdgauge_depence!(ΔU, ΔV, U, S, V)
            test_rrule(tsvd, C; atol, output_tangent=(ΔU, ΔS, ΔV), fkwargs=(; trunc))
        end

        let D = LinearAlgebra.eigvals(C)
            ΔD = diag(randn(complex(scalartype(C)), space(C)))
            test_rrule(LinearAlgebra.eigvals, C; atol, output_tangent=ΔD,
                       fkwargs=(; sortby=nothing))
        end

        let S = LinearAlgebra.svdvals(C)
            ΔS = diag(randn(real(scalartype(C)), space(C)))
            test_rrule(LinearAlgebra.svdvals, C; atol, output_tangent=ΔS)
        end
    end
end
