"""
    module TensorKitTestSuite

Test suite and utilities that ensure a reusable way of verifying that a custom `Sector`
type is correctly supported by fusion trees, `GradedSpace`, and `TensorMap`.

Downstream packages may include this test suite as follows:

```julia
import TensorKit
testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKit))), # TensorKit root
    "test", "TensorKitTestSuite.jl"
)
include(testsuite_path)
using .TensorKitTestSuite

TensorKitTestSuite.test_fusiontrees(MySector)
TensorKitTestSuite.test_spaces(MySector)
TensorKitTestSuite.test_tensors((V1, V2, V3, V4, V5)) # 5 mutually compatible spaces
```

The three entry points above are independent and may be run selectively.

Additionally, this test suite exports the following convenience testing utilities:
* [`smallset`](@ref)
* [`randsector`](@ref)
* [`random_fusion`](@ref)
* [`hasfusiontensor`](@ref)
* [`force_planar`](@ref)
"""
module TensorKitTestSuite

export test_fusiontrees, test_spaces, test_tensors
export smallset, randsector, random_fusion, hasfusiontensor, force_planar

using Test
using TestExtras
using Random
using Random: randperm
using LinearAlgebra
using TupleTools
using Combinatorics: permutations
using TensorOperations
using MatrixAlgebraKit: left_polar, isunitary
using Base.Iterators: take, product

using TensorKit
using TensorKit: type_repr, FusionTreeBlock, ℙ, PlanarTrivial, hassector, HomSpace
import TensorKit as TK
using TensorKitSectors

const testgroups = Dict{Symbol, Dict{String, Expr}}(
    :fusiontrees => Dict{String, Expr}(),
    :spaces => Dict{String, Expr}(),
    :tensors => Dict{String, Expr}(),
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

Register a testsuite entry under `testgroup` (one of `:fusiontrees`, `:spaces`,`:tensors`).
The body is executed with a single argument: the concrete `Sector` type
under test (for `:fusiontrees`/`:spaces`), or a 5-tuple/vector of mutually compatible spaces (for `:tensors`). 

Important: Whatever is passed as `name` becomes part of the generated function that must be called to run that body.
In particular, a `safe_name` is made where `name`'s spaces are replaced by underscores, and everything becomes lowercase.
One then calls `test_<testgroup>_<safe_name>`. This way, individual entries can be invoked without running the whole test group.
"""
macro testsuite(testgroup, name, ex) #TODO: rename
    Meta.isexpr(ex, :(->)) || error("@testsuite requires an `arg -> body` expression")
    testgroupsym = testgroup isa QuoteNode ? testgroup.value : testgroup
    safe_name = lowercase(replace(name, r"[^A-Za-z0-9]+" => "_"))
    fn = Symbol("test_", testgroupsym, "_", safe_name)
    group = QuoteNode(testgroupsym)
    return quote
        @assert !haskey(testgroups[$group], $name) "duplicate testsuite name: $($name) ($($group))"
        testgroups[$group][$name] = $(QuoteNode(ex))
        $(esc(fn))(arg) = _run_testsuite_entry(testgroups[$group][$name], arg)
        nothing
    end
end

"""
    test_fusiontrees(I::Type{<:Sector})

Runs the entire fusion-tree manipulation test suite (single and double fusion trees)
on sector type `I`.
"""
function test_fusiontrees(I::Type{<:Sector})
    return @testset "$(type_repr(I))" begin
        for (name, lambda) in testgroups[:fusiontrees]
            @testset "$name" begin
                _run_testsuite_entry(lambda, I)
            end
        end
    end
end

"""
    test_spaces(I::Type{<:Sector})

Runs the `GradedSpace` test suite on sector type `I`.
"""
function test_spaces(I::Type{<:Sector}) #TODO: change since it's just 1 testsuite, or change to get the hom-space tests in here
    return @testset "$(type_repr(I))" begin
        for (name, lambda) in testgroups[:spaces]
            @testset "$name" begin
                _run_testsuite_entry(lambda, I)
            end
        end
    end
end

"""
    test_tensors(V)

Runs the tensor operation test suite (construction, contractions, linear algebra,
index manipulations, braiding, `HomSpace`) on `V`, a 5-tuple of mutually
compatible `ElementarySpace`s. See `setup.jl` for space design considerations.
"""
function test_tensors(V::NTuple{5, GradedSpace{I, NTuple{N, Int}}}) where {I <: Sector, N}
    return @testset "$(type_repr(I))" begin
        for (name, lambda) in testgroups[:tensors]
            @testset "$name" begin
                _run_testsuite_entry(lambda, V)
            end
        end
    end
end

# Sector utilities
# ----------------
# TODO: replace with the ones from TensorKitSectors.SectorTestSuite
smallset(::Type{I}) where {I <: Sector} = take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1, I2}}}) where {I1, I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i, j) in iter if dim(i) * dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1, I2, I3}}}) where {I1, I2, I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i, j, k) in iter if dim(i) * dim(j) * dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end

function randsector(::Type{I}) where {I <: Sector}
    s = collect(smallset(I))
    a = rand(s)
    while isunit(a) # don't use trivial label
        a = rand(s)
    end
    return a
end

function hasfusiontensor(I::Type{<:Sector})
    try
        u = first(allunits(I))
        fusiontensor(u, u, u)
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

"""
    random_fusion(I::Type{<:Sector}, ::Val{N}) where {N}

Returns an `N`-tuple of sectors of type `I` that can consistently be used as the
uncoupled sectors of a fusion tree, i.e. consecutive sectors have a non-empty fusion product.
"""
function random_fusion(I::Type{<:Sector}, ::Val{N}) where {N} #TODO: merge with TKS random_fusion
    N == 1 && return (randsector(I),)
    tail = random_fusion(I, Val(N - 1))
    s = randsector(I)
    counter = 0
    while isempty(⊗(s, first(tail))) && counter < 20
        counter += 1
        s = (counter < 20) ? randsector(I) : leftunit(first(tail))
    end
    return (s, tail...)
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

include("testsuite/fusiontrees.jl")
include("testsuite/spaces.jl")
include("testsuite/tensors.jl")

end # module TensorKitTestSuite
