using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: DefaultAlgorithm, diagview

spacelist = factorization_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("Projections with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Projections with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.run_testsuite(:factorizations, "hermitian projections", V)
        TensorKitTestSuite.run_testsuite(:factorizations, "isometric projections", V)
    end
end
