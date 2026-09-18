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
        TensorKitTestSuite.run_testsuite(:tensors, "trivial space insertion and removal", V)
        TensorKitTestSuite.run_testsuite(:tensors, "permutations via inner product invariance", V)
        TensorKitTestSuite.run_testsuite(:tensors, "permutations via conversion", V)
        TensorKitTestSuite.run_testsuite(:tensors, "index flipping inverse", V)
        TensorKitTestSuite.run_testsuite(:tensors, "index flipping explicit", V)
        TensorKitTestSuite.run_testsuite(:tensors, "index flipping via contraction", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braid adjoint identity", V)
    end
    TensorKit.empty_globalcaches!()
end
