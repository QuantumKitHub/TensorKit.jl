using Test, TestExtras
using TensorKit
using MatrixAlgebraKit: DefaultAlgorithm, MatrixFunctionViaLA, MatrixFunctionViaEig,
    MatrixFunctionViaEigh, MatrixFunctionViaTaylor
using Random

spacelist = default_spacelist(fast_tests)
scalartypes = (Float32, Float64, ComplexF64)

# algorithms that agree on Hermitian input
hermitian_algs = (
    MatrixFunctionViaLA(), MatrixFunctionViaEig(DefaultAlgorithm()),
    MatrixFunctionViaEigh(DefaultAlgorithm()), MatrixFunctionViaTaylor(),
)
# algorithms valid for general (non-Hermitian) input
general_algs = (
    MatrixFunctionViaLA(), MatrixFunctionViaEig(DefaultAlgorithm()),
    MatrixFunctionViaTaylor(),
)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("Matrix exponentials with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Matrix exponentials with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        W = V1 ⊗ V2 ⊗ V3
        Vd = fuse(V1 ⊗ V2)

        # explicit (non-diagonal) algorithms only apply to dense endomorphisms;
        # `DiagonalTensorMap` inputs are exercised through the default algorithm below
        @testset "exponential for Hermitian matrices" begin
            for T1 in scalartypes, T2 in scalartypes,
                    A in (randn(T1, V1, V1), randn(T1, W, W))

                A = project_hermitian!(A)
                τ = rand(T2)

                expA = @constinferred exponential(A)
                expτA = @constinferred exponential((τ, A))

                for alg in hermitian_algs
                    @test expA ≈ @constinferred exponential(A, alg)
                    @test expτA ≈ @constinferred exponential((τ, A), alg)
                end
            end
        end

        @testset "exponential! for general matrices" begin
            for T1 in scalartypes, T2 in scalartypes,
                    A in (
                        randn(T1, V1, V1), randn(T1, W, W),
                        DiagonalTensorMap(randn(T1, reduceddim(Vd)), Vd),
                    )

                τ = rand(T2)

                expA = @constinferred exponential(A)
                if !(A isa DiagonalTensorMap) # diagonal only supports the default algorithm == DiagonalAlgorithm
                    for alg in general_algs
                        @test expA ≈ exponential(A, alg)
                    end
                end

                @test exponential!(copy(A)) ≈ exponential!((1.0, copy(A)))

                # exp(A)² == exp(2A)
                @test expA * expA ≈ exponential((2, A))

                # in-place semantics: aliases the input, unless a complex scalar forces
                # a real tensor to widen to complex
                Ain = copy(A)
                A2 = @constinferred exponential!((τ, Ain))
                A isa DiagonalTensorMap && @test A2 isa DiagonalTensorMap
                if T1 <: Real && T2 <: Complex
                    @test A2 !== Ain
                else
                    @test A2 === Ain
                end

                # exp(τA) and exp(-τA) are inverse
                # the inverse roundtrip is ill-conditioned; only assert at full (Float64) precision
                expτA = exponential!((τ, copy(A)))
                expmτA = exponential!((-τ, copy(A)))
                real(scalartype(expτA)) == Float64 && @test expτA * expmτA ≈ id(scalartype(expτA), domain(A))
            end
        end
    end
end
