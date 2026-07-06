using Test, TestExtras
using TensorKit
import TensorKit as TK

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

@timedtestset "Single fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in (fast_tests ? fast_sectorlist : sectorlist)
    TensorKitTestSuite.test_fusiontrees_iterate_and_printing(I)
    TensorKitTestSuite.test_fusiontrees_constructor_properties(I)
    TensorKitTestSuite.test_fusiontrees_split_and_join(I)
    TensorKitTestSuite.test_fusiontrees_multi_fmove(I)
    TensorKitTestSuite.test_fusiontrees_insertat(I)
    TensorKitTestSuite.test_fusiontrees_merging(I)
    TensorKitTestSuite.test_fusiontrees_elementary_planar_trace(I)
    TK.empty_globalcaches!()
end
