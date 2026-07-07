using Test, TestExtras
using TensorKit
import TensorKit as TK

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

@timedtestset "Double fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in (fast_tests ? fast_sectorlist : sectorlist)
    TensorKitTestSuite.test_double_fusiontrees_bending(I)
    TensorKitTestSuite.test_double_fusiontrees_folding(I)
    TensorKitTestSuite.test_double_fusiontrees_repartitioning(I)
    TensorKitTestSuite.test_double_fusiontrees_transposition(I)
    TensorKitTestSuite.test_double_fusiontrees_permutation_and_braiding(I)
    TensorKitTestSuite.test_double_fusiontrees_planar_trace(I)
    TK.empty_globalcaches!()
end
