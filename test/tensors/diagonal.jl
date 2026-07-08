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
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "basic properties and algebra", V)
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "linear algebra conversion", V)
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "real and imaginary parts", V)

        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "tensor conversion", V)
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "permutations", V)
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "trace, multiplication and inverse", V)
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "contraction", V)
        TensorKitTestSuite.run_testsuite(:diagonal_tensors, "tensor functions", V)
    end
    TensorKit.empty_globalcaches!()
end
