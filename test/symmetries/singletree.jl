using Test, TestExtras
using TensorKit
import TensorKit as TK

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

@timedtestset "Single fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in (fast_tests ? fast_sectorlist : sectorlist)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "iterate and printing", I)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "constructor properties", I)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "split and join", I)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "multi fmove", I)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "insertat", I)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "merging", I)
    TensorKitTestSuite.run_testsuite(:single_fusiontrees, "elementary planar trace", I)
    TK.empty_globalcaches!()
end
