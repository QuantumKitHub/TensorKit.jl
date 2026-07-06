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
        TensorKitTestSuite.test_tensors_linear_algebra(V)
    end
    TensorKit.empty_globalcaches!()
end
