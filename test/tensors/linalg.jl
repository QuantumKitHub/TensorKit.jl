using Test, TestExtras
using TensorKit
using TensorKit: type_repr
using LinearAlgebra: LinearAlgebra

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("Tensor linear algebra with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensor linear algebra with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.run_testsuite(:tensors, "basic linear algebra", V)
        TensorKitTestSuite.run_testsuite(:tensors, "linear algebra conversion", V)
        TensorKitTestSuite.run_testsuite(:tensors, "multiplication of isometries", V)
        TensorKitTestSuite.run_testsuite(:tensors, "multiplication and inverse compatibility", V)
        TensorKitTestSuite.run_testsuite(:tensors, "multiplication and inverse conversion", V)
        TensorKitTestSuite.run_testsuite(:tensors, "diag and diagm", V)
        TensorKitTestSuite.run_testsuite(:tensors, "tensor functions", V)
        TensorKitTestSuite.run_testsuite(:tensors, "sylvester equation", V)
    end
    TensorKit.empty_globalcaches!()
end
