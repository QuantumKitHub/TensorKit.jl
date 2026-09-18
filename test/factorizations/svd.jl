using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: DefaultAlgorithm, defaulttol, diagview

spacelist = factorization_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------------------------------")
    println("Singular value and polar decompositions with symmetry: $Istr")
    println("---------------------------------------------------------------")
    @timedtestset "Singular value and polar decompositions with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.run_testsuite(:factorizations, "condition number and rank", V)
        TensorKitTestSuite.run_testsuite(:factorizations, "polar decomposition", V)
        TensorKitTestSuite.run_testsuite(:factorizations, "SVD", V)
        TensorKitTestSuite.run_testsuite(:factorizations, "truncated SVD", V)
    end
end
