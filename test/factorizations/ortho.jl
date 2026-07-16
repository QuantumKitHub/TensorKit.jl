using Test, TestExtras
using TensorKit
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit: DefaultAlgorithm, diagview

spacelist = factorization_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("QR and LQ decompositions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "QR and LQ decompositions with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.run_testsuite(:factorizations, "QR decomposition", V)
        TensorKitTestSuite.run_testsuite(:factorizations, "LQ decomposition", V)
    end
end
