using Test, TestExtras
using TensorKit
using TensorKit: type_repr

spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    BraidingStyle(I) isa NoBraiding && continue
    println("---------------------------------------")
    println("BraidingTensor planar contractions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "BraidingTensor planar contractions with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.test_tensors_braiding_tensor_planaradd(V)
        TensorKitTestSuite.test_tensors_braiding_tensor_left_full_contraction(V)
        TensorKitTestSuite.test_tensors_braiding_tensor_left_partial_contraction(V)
        TensorKitTestSuite.test_tensors_braiding_tensor_right_full_contraction(V)
        TensorKitTestSuite.test_tensors_braiding_tensor_full_contraction_output(V)
        TensorKitTestSuite.test_tensors_braiding_tensor_open_codomain_leg(V)
        TensorKitTestSuite.test_tensors_braiding_tensor_open_domain_leg(V)
        TensorKitTestSuite.test_tensors_contraction_between_braiding_tensors(V)
    end
    TensorKit.empty_globalcaches!()
end
