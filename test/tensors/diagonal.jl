using Test, TestExtras
using TensorKit

diagspacelist = (
    (ℂ^4)',
    Vect[Z2Irrep](0 => 2, 1 => 3),
    Vect[Z3Element{1}](0 => 2, 1 => 3, 2 => 1),
    Vect[A4Irrep](0 => 1, 1 => 2, 2 => 2, 3 => 2),
    Vect[FermionNumber](0 => 2, 1 => 2, -1 => 1),
    Vect[SU2Irrep](0 => 2, 1 => 1)',
    Vect[FibonacciAnyon](:I => 2, :τ => 2),
    Vect[Z3Element{1}](0 => 2, 1 => 2, 2 => 1),
    Vect[IsingBimodule]((1, 1, 0) => 2, (1, 1, 1) => 3),
)

for V in diagspacelist
    I = sectortype(V)
    Istr = type_repr(I)
    println("---------------------------------------")
    println("DiagonalTensor with domain $V")
    println("---------------------------------------")
    @timedtestset "DiagonalTensor with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.test_diagonal_tensors_basic_properties_and_algebra(V)
        TensorKitTestSuite.test_diagonal_tensors_linear_algebra_conversion(V)
        TensorKitTestSuite.test_diagonal_tensors_real_and_imaginary_parts(V)
        TensorKitTestSuite.test_diagonal_tensors_tensor_conversion(V)
        TensorKitTestSuite.test_diagonal_tensors_permutations(V)
        TensorKitTestSuite.test_diagonal_tensors_trace_multiplication_and_inverse(V)
        TensorKitTestSuite.test_diagonal_tensors_contraction(V)
        TensorKitTestSuite.test_diagonal_tensors_tensor_functions(V)
    end
    TensorKit.empty_globalcaches!()
end
