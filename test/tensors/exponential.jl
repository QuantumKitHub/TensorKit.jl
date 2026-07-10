using Test, TestExtras
using TensorKit, MatrixAlgebraKit
using Random

spaces = [ℂ^4, Vect[U1Irrep](0 => 1, 1 => 2), Vect[SU2Irrep](0 => 1, 1 // 2 => 1)]
scalartypes = [Float64, ComplexF32, ComplexF64]
algorithms = [MatrixFunctionViaLA(), MatrixFunctionViaEig(DefaultAlgorithm()), MatrixFunctionViaEigh(DefaultAlgorithm())]

@timedtestset "exponential for Hermitian matrices with $space, scalartype(A) = $st1, scalartype(τ) = $st2" for space in spaces, st1 in scalartypes, st2 in scalartypes
    A = randn(st1, space, space)
    A = project_hermitian!(A)
    τ = rand(st2)

    expA = @constinferred exponential(A)
    expτA = @constinferred exponential((τ, A))

    for alg in algorithms
        expA2 = @constinferred exponential(A, alg)
        expτA2 = @constinferred exponential((τ, A), alg)

        @test expA ≈ expA2
        @test expτA ≈ expτA2
    end
end

@timedtestset "exponential! for general matrices for $space, scalartype(A) = $st1, scalartype(τ) = $st2" for space in spaces, st1 in scalartypes, st2 in scalartypes
    A = randn(st1, space, space)
    τ = rand(st2)

    @test exponential!(copy(A)) == exponential!((1.0, copy(A)))

    A2 = exponential!((τ, A))
    if st1 <: Real && st2 <: Complex
        @test objectid(A2) != objectid(A)
    else
        @test objectid(A2) == objectid(A)
    end

    expτA = exponential!((τ, copy(A)))
    expminτA = exponential!((-τ, copy(A)))
    @test expτA * expminτA ≈ id(scalartype(expτA), space)
    @test expτA ≈ inv(expminτA)
end

@timedtestset "exponential! for diagonal matrices for $space, scalartype(A) = $st1, scalartype(τ) = $st2" for space in spaces, st1 in scalartypes, st2 in scalartypes
    A = DiagonalTensorMap(rand(st1, reduceddim(space)), space)
    τ = rand(st2)

    exponential!(copy(A))
    @test exponential!(copy(A)) ≈ exponential!((1.0, copy(A))) #, DiagonalAlgorithm())

    A2 = @constinferred exponential!((τ, A))
    @test A2 isa DiagonalTensorMap
    if st1 <: Real && st2 <: Complex
        @test objectid(A2) != objectid(A)
    else
        @test objectid(A2) == objectid(A)
    end

    expτA = exponential!((τ, copy(A)))
    expminτA = exponential!((-τ, copy(A)))
    @test expτA * expminτA ≈ id(scalartype(expτA), space)
    @test expτA ≈ inv(expminτA)
end
