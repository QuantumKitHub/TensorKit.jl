"""
    module TensorKitTestSuite

Test suite and utilities that ensure a reusable way of verifying that a custom `Sector`
type is correctly supported by fusion trees, `GradedSpace`, and `TensorMap`.

Downstream packages may include this test suite as follows:

```julia
import TensorKit
testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKit))), # TensorKit root
    "test", "testsuite", "TensorKitTestSuite.jl"
)
include(testsuite_path)
using .TensorKitTestSuite

TensorKitTestSuite.run_testsuite(:single_fusiontrees, "test", MySector)
TensorKitTestSuite.run_testsuite(:double_fusiontrees, "test", MySector)
TensorKitTestSuite.run_testsuite(:spaces, "test", MySector)
TensorKitTestSuite.run_testsuite(:tensors, "test", (V1, V2, V3, V4, V5)) # 5 mutually compatible spaces
TensorKitTestSuite.run_testsuite(:factorizations, "test", (V1, V2, V3, V4, V5)) # 5 mutually compatible spaces
TensorKitTestSuite.run_testsuite(:diagonal_tensors, "test", V) # 1 space for diagonal tensors
```

The entry points are denoted by the `Symbol`s above, with "test" being the name of the test suite.
These are independent and may be run selectively. See [`@testsuite`](@ref) for more information.
This module additionally exports:
* [`force_planar`](@ref)
* [`eval_show`](@ref)

Sector-level helpers are reused from `TensorKitSectors.SectorTestSuite` internally,
but deliberately *not* re-exported here.
"""
module TensorKitTestSuite

export run_testsuite
export force_planar, eval_show

using Test
using TestExtras
using Random
using Random: randperm
using LinearAlgebra
using TupleTools
using Combinatorics: permutations
using TensorOperations
using MatrixAlgebraKit: left_polar, isunitary

using TensorKit
using TensorKit: type_repr, FusionTreeBlock, ℙ, PlanarTrivial, hassector, HomSpace, check_spacetype
import TensorKit as TK
using TensorKitSectors
using TensorKitSectors: ×

# Reuse TensorKitSectors's own sector-level test helpers
sectortestsuite_path = joinpath(
    dirname(dirname(pathof(TensorKitSectors))), "test", "testsuite.jl"
)
include(sectortestsuite_path)
using .SectorTestSuite: smallset, randsector, hasfusiontensor
using .SectorTestSuite: can_fuse, F_unitarity_test, R_unitarity_test
import .SectorTestSuite: random_fusion # TODO: is the method added below needed?

const testgroups = Dict{Symbol, Dict{String, Expr}}(
    :single_fusiontrees => Dict{String, Expr}(),
    :double_fusiontrees => Dict{String, Expr}(),
    :spaces => Dict{String, Expr}(),
    :tensors => Dict{String, Expr}(),
    :diagonal_tensors => Dict{String, Expr}(),
    :factorizations => Dict{String, Expr}(),
)

# cannot just esc() the body, because that would make it a closure, compile it at a fixed world age and break constprop=true
# workaround here is to store an unevaluated `Expr` and `Core.eval` it in a fresh `let` block every time
# this is at the cost of not reusing compiled code
function _run_testsuite_entry(lambda::Expr, arg)
    param, body = lambda.args[1], lambda.args[2]
    letex = Expr(:let, Expr(:(=), param, arg), body) # let block makes param local to the body
    return Core.eval(@__MODULE__, letex)
end

"""
    @testsuite testgroup name I -> begin
        # test code here
    end

Register a testsuite entry under `testgroup` (one of `:single_fusiontrees`, `:double_fusiontrees`, `:spaces`,`:tensors`, `:diagonal_tensors`).
The body is executed with a single argument: the concrete `Sector` type under test
(for `:single_fusiontrees`, `:double_fusiontrees` and `:spaces`), a space (for `:diagonal_tensors`),
or a 5-tuple of mutually compatible spaces (for `:tensors` and `:factorizations`). 

For the test groups involving spaces, see `setup.jl` for the space design considerations.

Run a registered entry via `run_testsuite(testgroup, name, arg)`.
"""
macro testsuite(testgroup, name, ex)
    Meta.isexpr(ex, :(->)) || error("@testsuite requires an `arg -> body` expression")
    testgroupsym = testgroup isa QuoteNode ? testgroup.value : testgroup
    group = QuoteNode(testgroupsym)
    return quote
        @assert !haskey(testgroups[$group], $name) "duplicate testsuite name: $($name) ($($group))"
        testgroups[$group][$name] = $(QuoteNode(ex))
        nothing
    end
end

"""
    run_testsuite(group::Symbol, name::String, arg)

Run a single registered testsuite entry by its `group` and `name`.
"""
run_testsuite(group::Symbol, name::String, arg) = _run_testsuite_entry(testgroups[group][name], arg)

# Sector utilities
# ----------------
"""
    random_fusion(I::Type{<:Sector}, ::Val{N}) where {N}

Returns an `NTuple{N,I}` of sectors that can consistently be used as the uncoupled
sectors of a fusion tree, i.e. consecutive sectors have a non-empty fusion product.
Thin wrapper around `SectorTestSuite.random_fusion(I, N::Int)`.
"""
function random_fusion(I::Type{<:Sector}, ::Val{N}) where {N}
    v = random_fusion(I, N)
    return ntuple(i -> v[i], Val(N))
end

"""
    force_planar(obj)

Replace an object with a planar equivalent -- i.e. one that disallows braiding.
"""
force_planar(V::ComplexSpace) = isdual(V) ? (ℙ^dim(V))' : ℙ^dim(V)
function force_planar(V::GradedSpace)
    return GradedSpace((c ⊠ PlanarTrivial() => dim(V, c) for c in sectors(V))..., isdual(V))
end
force_planar(V::ProductSpace) = mapreduce(force_planar, ⊗, V)
function force_planar(tsrc::TensorMap{<:Any, ComplexSpace})
    tdst = similar(tsrc, force_planar(codomain(tsrc)) ← force_planar(domain(tsrc)))
    copyto!(block(tdst, PlanarTrivial()), block(tsrc, Trivial()))
    return tdst
end
function force_planar(tsrc::TensorMap{<:Any, <:GradedSpace})
    tdst = similar(tsrc, force_planar(codomain(tsrc)) ← force_planar(domain(tsrc)))
    for (c, b) in blocks(tsrc)
        copyto!(block(tdst, c ⊠ PlanarTrivial()), b)
    end
    return tdst
end

# # helper function to check that d - dim(c) < dim(V) <= d where c is the largest sector
# # to allow for truncations to have some margin with larger sectors
# function dim_isapprox(V::ElementarySpace, d::Int)
#     dim_c_max = maximum(dim, sectors(V); init = 1)
#     return max(0, d - dim_c_max) ≤ dim(V) ≤ d + dim_c_max
# end
# function dim_isapprox(V::ProductSpace, d::Int)
#     dim_c_max = maximum(dim, blocksectors(V); init = 1)
#     return max(0, d - dim_c_max) ≤ dim(V) ≤ d + dim_c_max
# end

_isunitary(x::Number; kwargs...) = isapprox(x * x', one(x); kwargs...)
_isunitary(x; kwargs...) = isunitary(x; kwargs...)
_isone(x; kwargs...) = isapprox(x, one(x); kwargs...)

"""
    eval_show(x)

Use `show` to generate a string representation of `x`, then parse and evaluate the resulting expression.
"""
function eval_show(x)
    str = sprint(show, x; context = (:module => @__MODULE__))
    ex = Meta.parse(str)
    return eval(ex)
end

include("fusiontrees.jl")
include("spaces.jl")
include("tensors.jl")
include("diagonal.jl")
include("factorizations.jl")

end # module TensorKitTestSuite
