"""
    module TensorKitTestSuite

Test suite and utilities that ensure a reusable way of verifying that a custom `Sector`
type is correctly supported by `GradedSpace`.

Downstream packages may include this test suite as follows:

```julia
import TensorKit
testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKit))), # TensorKit root
    "test", "testsuite", "TensorKitTestSuite.jl"
)
include(testsuite_path)

TensorKitTestSuite.test_graded_space(MySector)
```

Nothing is exported, so test functions and utilities are called with the module prefix.
This requires `Test`, `TestExtras`, `TensorKit` and `TensorKitSectors` to be available in the
active environment.

Sector-level helpers are reused from `TensorKitSectors.SectorTestSuite` internally,
but deliberately *not* re-exported here.
"""
module TensorKitTestSuite

using Test
using TestExtras

using TensorKit
using TensorKit: type_repr, hassector
using TensorKitSectors

# Reuse TensorKitSectors's own sector-level test helpers
sectortestsuite_path = joinpath(
    dirname(dirname(pathof(TensorKitSectors))), "test", "testsuite.jl"
)
include(sectortestsuite_path)
using .SectorTestSuite: randsector, hasfusiontensor

"""
    eval_show(x)

Use `show` to generate a string representation of `x`, then parse and evaluate the resulting expression.
"""
function eval_show(x)
    str = sprint(show, x; context = (:module => @__MODULE__))
    ex = Meta.parse(str)
    return eval(ex)
end

include("spaces.jl")

end # module TensorKitTestSuite
