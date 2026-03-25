using Test, TestExtras
using TensorKit
using TensorKit: type_repr, SectorDict
using TensorOperations
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences: FiniteDifferences, central_fdm, forward_fdm
using Random
using LinearAlgebra
using Zygote
using MatrixAlgebraKit

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
function ChainRulesTestUtils.test_approx(
        actual::AbstractTensorMap, expected::AbstractTensorMap, msg = ""; kwargs...
    )
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
    return nothing
end

# Float32 and finite differences don't mix well
precision(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
precision(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-5

function test_ad_rrule(f, args...; check_inferred = false, kwargs...)
    test_rrule(
        Zygote.ZygoteRuleConfig(), f, args...;
        rrule_f = rrule_via_ad, check_inferred, kwargs...
    )
    return nothing
end

# project_hermitian is non-differentiable for now
_project_hermitian(x) = (x + x') / 2

# Tests
# -----

ChainRulesTestUtils.test_method_tables()

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
    (
        Vect[Z2Irrep](0 => 1, 1 => 1),
        Vect[Z2Irrep](0 => 1, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 3),
        Vect[Z2Irrep](0 => 2, 1 => 2),
    ),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2)',
        Vect[FermionParity](0 => 2, 1 => 1)',
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
    (
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
        Vect[U1Irrep](0 => 1, 1 => 1, -1 => 2),
        Vect[U1Irrep](0 => 1, 1 => 2, -1 => 1)',
    ),
    (
        Vect[SU2Irrep](0 => 2, 1 // 2 => 1),
        Vect[SU2Irrep](0 => 1, 1 => 1),
        Vect[SU2Irrep](1 // 2 => 1, 1 => 1)',
        Vect[SU2Irrep](1 // 2 => 2),
        Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
    ),
    (
        Vect[FibonacciAnyon](:I => 2, :τ => 1),
        Vect[FibonacciAnyon](:I => 1, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 3),
        Vect[FibonacciAnyon](:I => 2, :τ => 2),
    ),
)

for V in spacelist
    I = sectortype(eltype(V))
    Istr = type_repr(I)
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
                test_rrule(convert, Array, T1)
                test_rrule(
                    TensorMap, convert(Array, T1), codomain(T1), domain(T1);
                    fkwargs = (; tol = Inf)
                )
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

            test_rrule(transpose, A, ((2, 5, 4), (1, 3)))
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
            atol = precision(T)
            rtol = precision(T)
            for i in 1:3
                E = randn(T, ⊗(V[1:i]...) ← ⊗(V[1:i]...))
                test_rrule(LinearAlgebra.tr, E; atol, rtol)
                test_rrule(exp, E; check_inferred = false, atol, rtol)
                test_rrule(inv, E; atol, rtol)
            end

            A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
            test_rrule(LinearAlgebra.adjoint, A; atol, rtol)
            test_rrule(LinearAlgebra.norm, A, 2; atol, rtol)

            B = randn(T, space(A))
            test_rrule(LinearAlgebra.dot, A, B; atol, rtol)
        end

        @timedtestset "Matrix functions ($T)" for T in eltypes
            atol = precision(T)
            rtol = precision(T)
            for f in (sqrt, exp)
                check_inferred = false # !(T <: Real) # not type-stable for real functions
                t1 = randn(T, V[1] ← V[1])
                t2 = randn(T, V[2] ← V[2])
                d = DiagonalTensorMap{T}(undef, V[1])
                d2 = DiagonalTensorMap{T}(undef, V[1])
                d3 = DiagonalTensorMap{T}(undef, V[1])
                if (T <: Real && f === sqrt)
                    # ensuring no square root of negative numbers
                    randexp!(d.data)
                    d.data .+= 5
                    randexp!(d2.data)
                    d2.data .+= 5
                    randexp!(d3.data)
                    d3.data .+= 5
                else
                    randn!(d.data)
                    randn!(d2.data)
                    randn!(d3.data)
                end

                test_rrule(f, t1; rrule_f = Zygote.rrule_via_ad, check_inferred, atol, rtol)
                test_rrule(f, t2; rrule_f = Zygote.rrule_via_ad, check_inferred, atol, rtol)
                test_rrule(f, d ⊢ d2; check_inferred, output_tangent = d3, atol, rtol)
            end
        end
    end
end
