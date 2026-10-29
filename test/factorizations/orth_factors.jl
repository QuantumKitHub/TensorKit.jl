using TensorKit
using MatrixAlgebraKit: DefaultAlgorithm
using Test

@testset "Orthogonalization without a basis" begin
    for T in (Float32, ComplexF64), (V, W) in (
            (ℂ^7, ℂ^4),
            (U1Space(0 => 3, 1 => 2), U1Space(0 => 2, -1 => 1)),
            (SU2Space(0 => 3, 1/2 => 1), SU2Space(0 => 1, 1/2 => 2)),
        )
        t = randn(T, V ← W)
        V0 = zerospace(spacetype(t))
        for A in (t, t'), positive in (false, true), alg in ((; positive), DefaultAlgorithm(; positive))
            Q0 = similar(A, V0 ← V0)
            Q, R = left_orth(A; positive)
            R0 = similar(R)
            Q2, R2 = left_orth!(copy(A), (Q0, R0); alg)
            @test Q2 === Q0
            @test R2 === R0
            @test dim(Q2) == 0
            @test space(R2) == space(R)
            @test R2 ≈ R
            @test Q * R2 ≈ A

            L, Q = right_orth(A; positive)
            L0 = similar(L)
            L2, Q2 = right_orth!(copy(A), (L0, Q0); alg)
            @test Q2 === Q0
            @test L2 === L0
            @test dim(Q2) == 0
            @test space(L2) == space(L)
            @test L2 ≈ L
            @test L2 * Q ≈ A
        end
        for positive in (false, true)
            d = DiagonalTensorMap(randn(T, reduceddim(V)), V)
            Q0 = DiagonalTensorMap(T[], V0)
            Q, R = left_orth(d; positive)
            Q2, R2 = left_orth!(copy(d), (Q0, similar(R)); positive)
            @test Q2 === Q0
            @test R2 ≈ R
            @test Q * R2 ≈ d
            L, Q = right_orth(d; positive)
            L2, Q2 = right_orth!(copy(d), (similar(L), Q0); positive)
            @test Q2 === Q0
            @test L2 ≈ L
            @test L2 * Q ≈ d
        end
    end
end
