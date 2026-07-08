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
        TensorKitTestSuite.test_tensors_basic_linear_algebra(V)
        TensorKitTestSuite.test_tensors_linear_algebra_conversion(V)
        TensorKitTestSuite.test_tensors_multiplication_of_isometries(V)
        TensorKitTestSuite.test_tensors_multiplication_and_inverse_compatibility(V)
        TensorKitTestSuite.test_tensors_multiplication_and_inverse_conversion(V)
        TensorKitTestSuite.test_tensors_diag_and_diagm(V)
        TensorKitTestSuite.test_tensors_tensor_functions(V)
        TensorKitTestSuite.test_tensors_sylvester_equation(V)
    end
    TensorKit.empty_globalcaches!()
end
