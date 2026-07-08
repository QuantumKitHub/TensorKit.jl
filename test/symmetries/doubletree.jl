using Test, TestExtras
using TensorKit
import TensorKit as TK

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

@timedtestset "Double fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in (fast_tests ? fast_sectorlist : sectorlist)
    TensorKitTestSuite.run_testsuite(:double_fusiontrees, "bending", I)
    TensorKitTestSuite.run_testsuite(:double_fusiontrees, "folding", I)
    TensorKitTestSuite.run_testsuite(:double_fusiontrees, "repartitioning", I)
    TensorKitTestSuite.run_testsuite(:double_fusiontrees, "transposition", I)
    TensorKitTestSuite.run_testsuite(:double_fusiontrees, "permutation and braiding", I)
    TensorKitTestSuite.run_testsuite(:double_fusiontrees, "planar trace", I)
    TK.empty_globalcaches!()
end
