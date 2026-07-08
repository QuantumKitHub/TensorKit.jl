using Test, TestExtras
using TensorKit
using TensorKit: type_repr


spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("Tensor index manipulations with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensor index manipulations with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.test_tensors_trivial_space_insertion_and_removal(V)
        TensorKitTestSuite.test_tensors_permutations_via_inner_product_invariance(V)
        TensorKitTestSuite.test_tensors_permutations_via_conversion(V)
        TensorKitTestSuite.test_tensors_index_flipping_inverse(V)
        TensorKitTestSuite.test_tensors_index_flipping_explicit(V)
        TensorKitTestSuite.test_tensors_index_flipping_via_contraction(V)
        TensorKitTestSuite.test_tensors_braid_adjoint_identity(V)
    end
    TensorKit.empty_globalcaches!()
end
