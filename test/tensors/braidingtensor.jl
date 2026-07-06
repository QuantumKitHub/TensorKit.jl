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
        TensorKitTestSuite.test_tensors_braiding_tensor(V)
    end
    TensorKit.empty_globalcaches!()
end
