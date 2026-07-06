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
        TensorKitTestSuite.test_tensors_index_manipulations(V)
    end
    TensorKit.empty_globalcaches!()
end
