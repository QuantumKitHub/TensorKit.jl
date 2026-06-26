using Test, TestExtras
using TensorKit
using Random

spaces = [ℂ^4, Vect[U1Irrep](0 => 1, 1 => 2), Vect[SU2Irrep](0 => 1, 1 // 2 => 1)]
scalartypes = [Float64, ComplexF64]

@timedtestset "exp!(τ,A) for $space, scalartype(A) = $st1, scalartype(τ) = $st2" for space in spaces, st1 in scalartypes, st2 in scalartypes
    A = randn(st1, space, space)
    τ = rand(st2)

    @test exp!(copy(A)) == exp!((1.0, copy(A)))

    A2 = exp!((τ, A))
    if st1 <: Real && st2 <: Complex
        @test objectid(A2) != objectid(A)
    else
        @test objectid(A2) == objectid(A)
    end

    expτA = exp!((τ, copy(A)))
    expminτA = exp!((-τ, copy(A)))
    @test expτA * expminτA ≈ id(scalartype(expτA), space)
    @test expτA ≈ inv(expminτA)
end
