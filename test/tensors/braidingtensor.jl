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
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor planaradd!", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor left full contraction", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor left partial contraction", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor right full contraction", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor full contraction output", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor open codomain leg", V)
        TensorKitTestSuite.run_testsuite(:tensors, "braiding tensor open domain leg", V)
        TensorKitTestSuite.run_testsuite(:tensors, "contraction between braiding tensors", V)
    end
    TensorKit.empty_globalcaches!()
end
