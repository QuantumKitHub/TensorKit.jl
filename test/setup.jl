module TestSetup

export randindextuple, randcircshift, _repartition, trivtuple
export default_tol
export smallset, randsector, hasfusiontensor, force_planar, eval_show
export random_fusion
export sectorlist, fast_sectorlist
# export dim_isapprox
export default_spacelist, factorization_spacelist, ad_spacelist
export test_ad_rrule
export _isunitary, _isone
export TensorKitTestSuite

using Random
using Test: @test
using TensorKit
using TensorKitSectors
using TensorOperations: IndexTuple, Index2Tuple
using TupleTools
using MatrixAlgebraKit: MatrixAlgebraKit
using ChainRulesCore: NoTangent
using ChainRulesTestUtils: ChainRulesTestUtils, test_rrule
using Zygote: Zygote, rrule_via_ad

include(joinpath(@__DIR__, "testsuite", "TensorKitTestSuite.jl"))
using .TensorKitTestSuite
# not exported by TensorKitTestSuite to avoid clash with TensorKitSectors.SectorTestSuite
using .TensorKitTestSuite: _isunitary, _isone
using .TensorKitTestSuite: smallset, randsector, hasfusiontensor, random_fusion
TensorKitTestSuite.fast_tests[] = fast_tests # inject fast_tests into TensorKitTestSuite

Random.seed!(123456)

# IndexTuple utility
# ------------------
function randindextuple(N::Int, k::Int = rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end
function randcircshift(N₁::Int, N₂::Int, k::Int = rand(0:(N₁ + N₂)))
    N = N₁ + N₂
    @assert 0 ≤ k ≤ N
    p = TupleTools.vcat(ntuple(identity, N₁), reverse(ntuple(identity, N₂) .+ N₁))
    n = rand(0:N)
    _p = TupleTools.circshift(p, n)
    return (tuple(_p[1:k]...), reverse(tuple(_p[(k + 1):end]...)))
end

trivtuple(N) = ntuple(identity, N)

Base.@constprop :aggressive function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return TupleTools.getindices(p, trivtuple(N₁)),
        TupleTools.getindices(p, trivtuple(length(p) - N₁) .+ N₁)
end
Base.@constprop :aggressive function _repartition(p::Index2Tuple, N₁::Int)
    return _repartition(linearize(p), N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, t::AbstractTensorMap)
    return _repartition(p, TensorKit.numout(t))
end

# Float32 and finite differences don't mix well
default_tol(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
default_tol(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-5

# Sector lists
# --------------
uniquefusionsectorlist = (
    Z2Irrep, Z3Irrep, Z4Irrep, Z3Irrep ⊠ Z4Irrep, U1Irrep,
    FermionParity, FermionParity ⊠ FermionParity, FermionNumber, # fermionic
    Z4Element{2}, ZNElement{6, 3}, # complex F symbols and anyonic braiding
    Z3Element{1}, ZNElement{5, 2}, # complex F symbols, no braiding
)
simplefusionsectorlist = (
    CU1Irrep, SU2Irrep, FibonacciAnyon, IsingAnyon,
    FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ SU2Irrep,
    ZNElement{6, 3} ⊠ SU2Irrep, # complex F symbols, anyonic braiding
    Z3Element{1} ⊠ FibonacciAnyon ⊠ FibonacciAnyon, # complex F symbols, no braiding
)
genericfusionsectorlist = (
    A4Irrep, A4Irrep ⊠ FermionParity, A4Irrep ⊠ SU2Irrep, A4Irrep ⊠ A4Irrep,
    A4Irrep ⊠ Z4Element{2}, # complex F symbols, anyonic braiding
    A4Irrep ⊠ Z3Element{1}, # complex F symbols, no braiding
)
multifusionsectorlist = (
    IsingBimodule, IsingBimodule ⊠ SU2Irrep, IsingBimodule ⊠ IsingBimodule,
    IsingBimodule ⊠ FibonacciAnyon ⊠ A4Irrep, # generic fusion
    IsingBimodule ⊠ A4Irrep ⊠ Z3Element{1}, # generic fusion and complex F symbols
)

sectorlist = (
    uniquefusionsectorlist...,
    simplefusionsectorlist...,
    genericfusionsectorlist...,
    multifusionsectorlist...,
)
fast_sectorlist = (Z2Irrep, SU2Irrep, FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FibonacciAnyon)

# spaces
# space design considerations:
# 1.    V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5 should exist (in multifusion case) in such a way that
#       V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5 ← () is non-empty, i.e. leftunitspace(V1) == rightunitspace(V5)
#       and blockdim(V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5, leftunit(first(sectors(V1)))) > 0:
#       this means that we can also make consistent maps in the homspaces that correspond to
#       repartitions and transpositions thereof,
#       i.e. V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)', V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)', V2 ⊗ V3 ⊗ V4 ← V1' ⊗ V5', etc,
#       as well as any endomorphism spaces V1 ⊗ V2 ← V1 ⊗ V2, V1 ⊗ V2 ⊗ V3 ← V1 ⊗ V2 ⊗ V3, etc.
#
# 2.    In order to construct isometries (QR etc), it is nice to have that
#       V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)' has tall blocks (few columns than rows), i.e. V1 ⊗ V2 ⊗ V3 ≿ (V4 ⊗ V5)'
#       and that V1 ⊗ V2 ← (V3 ⊗ V4 ⊗ V5)' has wide blocks (few rows than columns), i.e. V1 ⊗ V2 ≾ (V3 ⊗ V4 ⊗ V5)'.
#
# 3.    Tensor manipulations can depend on FusionStyle(I), BraidingStyle(I), UnitStyle(I) and sectorscalartype(I),
#       so it is good to have a variety of sectors in the spacelist to test different code paths. We also definitely
#       want to cover the most common cases: trivial symmetry, fermionic parity, U(1) charge, SU(2), Hubbard-like
#       symmetries, and some anyonic examples with nontrivial F symbols and braiding.
#
# 4.    V1 should not only have one-dimensional blocks as this causes some tests to fail, like `!ishermitian(thapprox)`.
#
# 5.    V2 and V4 are consistently made a dual space


# Trivial
Vtr = (ℂ^2, (ℂ^3)', ℂ^4, ℂ^3, (ℂ^2)')

# UniqueFusion, Bosonic
VRepℤ₂ = (
    Vect[Z2Irrep](0 => 2, 1 => 2),
    Vect[Z2Irrep](0 => 1, 1 => 1)',
    Vect[Z2Irrep](0 => 3, 1 => 1),
    Vect[Z2Irrep](0 => 1, 1 => 2)',
    Vect[Z2Irrep](0 => 3, 1 => 1),
)
VRepℤ₃ = (
    Vect[Z3Irrep](0 => 2, 1 => 1, 2 => 1),
    Vect[Z3Irrep](0 => 1, 1 => 1, 2 => 1)',
    Vect[Z3Irrep](0 => 3, 1 => 1, 2 => 1),
    Vect[Z3Irrep](0 => 0, 1 => 2, 2 => 1)',
    Vect[Z3Irrep](0 => 1, 1 => 0, 2 => 2),
)
VRepU₁ = (
    Vect[U1Irrep](0 => 2, 1 => 2, -1 => 2),
    Vect[U1Irrep](0 => 1, 1 => 1, -1 => 1)',
    Vect[U1Irrep](0 => 3, 1 => 1, -1 => 1),
    Vect[U1Irrep](0 => 1, 1 => 2, -1 => 1)',
    Vect[U1Irrep](0 => 1, 1 => 1, -1 => 2),
)

# UniqueFusion, Fermionic
VfRepℤ₂ = (
    Vect[FermionParity](0 => 2, 1 => 2),
    Vect[FermionParity](0 => 1, 1 => 1)',
    Vect[FermionParity](0 => 3, 1 => 1),
    Vect[FermionParity](0 => 1, 1 => 2)',
    Vect[FermionParity](0 => 2, 1 => 2),
)

# UniqueFusion, anyonic braiding
VTwistedVecℤ₄ = (
    Vect[Z4Element{2}](0 => 2, 1 => 2, 2 => 0, 3 => 0),
    Vect[Z4Element{2}](0 => 2, 1 => 0, 2 => 0, 3 => 1)',
    Vect[Z4Element{2}](0 => 2, 1 => 1, 2 => 1, 3 => 1),
    Vect[Z4Element{2}](0 => 1, 1 => 2, 2 => 0, 3 => 1)',
    Vect[Z4Element{2}](0 => 1, 1 => 0, 2 => 2, 3 => 0),
)

# UniqueFusion, no braiding
VTwistedVecℤ₃ = (
    Vect[Z3Element{1}](0 => 2, 1 => 2, 2 => 0),
    Vect[Z3Element{1}](0 => 2, 1 => 0, 2 => 1)',
    Vect[Z3Element{1}](0 => 2, 1 => 1, 2 => 1),
    Vect[Z3Element{1}](0 => 1, 1 => 0, 2 => 2)',
    Vect[Z3Element{1}](0 => 1, 1 => 2, 2 => 0),
)

# SimpleFusion, bosonic
VRepCU₁ = (
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 0),
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 0, 1 => 1)',
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 1)',
    Vect[CU1Irrep]((0, 0) => 1, (0, 1) => 1, 1 => 0),
)
VRepSU₂ = (
    Vect[SU2Irrep](0 => 3, 1 // 2 => 1),
    Vect[SU2Irrep](0 => 2, 1 => 1)',
    Vect[SU2Irrep](1 // 2 => 1, 1 => 1),
    Vect[SU2Irrep](0 => 2, 1 // 2 => 2)',
    Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1),
)

# SimpleFusion, fermionic
VfRepCU₁ = (
    Vect[FermionParity ⊠ CU1Irrep]((0, (0, 0)) => 2, (0, (0, 1)) => 1, (1, 1) => 0),
    Vect[FermionParity ⊠ CU1Irrep]((0, (0, 0)) => 2, (0, (0, 1)) => 0, (1, 1) => 1)',
    Vect[FermionParity ⊠ CU1Irrep]((0, (0, 0)) => 2, (0, (0, 1)) => 1, (1, 1) => 1),
    Vect[FermionParity ⊠ CU1Irrep]((0, (0, 0)) => 2, (0, (0, 1)) => 1, (1, 1) => 1)',
    Vect[FermionParity ⊠ CU1Irrep]((0, (0, 0)) => 1, (0, (0, 1)) => 1, (1, 1) => 0),
)

VfRepSU₂ = (
    Vect[FermionSpin](0 => 3, 1 // 2 => 1),
    Vect[FermionSpin](0 => 2, 1 => 1)',
    Vect[FermionSpin](1 // 2 => 1, 1 => 1),
    Vect[FermionSpin](0 => 2, 1 // 2 => 2)',
    Vect[FermionSpin](0 => 1, 1 // 2 => 1, 3 // 2 => 1),
)

I_Hubbard = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
VfHubbard = (
    Vect[I_Hubbard]((0, 0, 0) => 3, (1, 1 // 2, -1) => 1),
    Vect[I_Hubbard]((0, 0, 0) => 1, (0, 0, -2) => 1, (0, 1, 0) => 1)',
    Vect[I_Hubbard]((1, 1 // 2, -1) => 1, (1, 1 // 2, +1) => 1, (0, 1, -2) => 1, (0, 1, +2) => 1),
    Vect[I_Hubbard]((0, 0, 0) => 2, (1, 1 // 2, +1) => 1, (1, 1 // 2, -1) => 1)',
    Vect[I_Hubbard]((0, 0, 0) => 1, (1, 1 // 2, +1) => 1, (1, 3 // 2, -1) => 1),
)

# SimpleFusion, anyonic braiding
Vfib = (
    Vect[FibonacciAnyon](:I => 2, :τ => 1),
    Vect[FibonacciAnyon](:I => 1, :τ => 1)',
    Vect[FibonacciAnyon](:I => 3, :τ => 1),
    Vect[FibonacciAnyon](:I => 1, :τ => 2)',
    Vect[FibonacciAnyon](:I => 3, :τ => 1),
)

# Generic fusion, bosonic
VRepA4 = (
    Vect[A4Irrep](0 => 2, 1 => 1, 2 => 0, 3 => 1),
    Vect[A4Irrep](0 => 1, 1 => 1, 2 => 1, 3 => 0)',
    Vect[A4Irrep](0 => 2, 1 => 0, 2 => 0, 3 => 1),
    Vect[A4Irrep](0 => 0, 1 => 1, 2 => 1, 3 => 1)',
    Vect[A4Irrep](0 => 0, 1 => 2, 2 => 0, 3 => 1),
)

# Generic fusion, fermionic
VfRepA4 = (
    Vect[FermionParity ⊠ A4Irrep]((0, 0) => 2, (1, 1) => 1, (0, 2) => 0, (1, 3) => 1),
    Vect[FermionParity ⊠ A4Irrep]((1, 0) => 1, (0, 1) => 1, (0, 2) => 1, (1, 3) => 0)',
    Vect[FermionParity ⊠ A4Irrep]((0, 0) => 2, (0, 1) => 0, (1, 1) => 0, (1, 2) => 1, (0, 3) => 1),
    Vect[FermionParity ⊠ A4Irrep]((0, 0) => 0, (1, 1) => 1, (1, 2) => 1, (0, 3) => 1)',
    Vect[FermionParity ⊠ A4Irrep]((1, 0) => 0, (0, 1) => 2, (1, 2) => 0, (0, 3) => 1),
)

# Generic fusion, anyonic braiding
I_A4Z4 = A4Irrep ⊠ Z4Element{2}
VRepA4Twistedℤ₄ = (
    Vect[I_A4Z4]((0, 0) => 2, (1, 1) => 1, (2, 3) => 0, (3, 2) => 1),
    Vect[I_A4Z4]((0, 0) => 1, (1, 1) => 1, (2, 3) => 1, (3, 2) => 0)',
    Vect[I_A4Z4]((0, 0) => 2, (1, 1) => 1, (2, 3) => 1, (3, 2) => 1),
    Vect[I_A4Z4]((0, 0) => 0, (1, 1) => 1, (2, 3) => 1, (3, 2) => 1)',
    Vect[I_A4Z4]((0, 0) => 0, (1, 1) => 2, (2, 3) => 0, (3, 2) => 1),
)

# Multifusion categories: GenericUnit, SimpleFusion

C0, C1 = IsingBimodule(1, 1, 0), IsingBimodule(1, 1, 1)
D0, D1 = IsingBimodule(2, 2, 0), IsingBimodule(2, 2, 1)
M, Mop = IsingBimodule(1, 2, 0), IsingBimodule(2, 1, 0)
VIBM = (
    Vect[IsingBimodule](C0 => 2, C1 => 1),
    Vect[IsingBimodule](Mop => 1)',
    Vect[IsingBimodule](D0 => 2, D1 => 2),
    Vect[IsingBimodule](M => 2)',
    Vect[IsingBimodule](C0 => 3, C1 => 2),
)

VIBMRepA4 = (
    Vect[IsingBimodule ⊠ A4Irrep]((C0, 0) => 2, (C1, 3) => 1),
    Vect[IsingBimodule ⊠ A4Irrep]((Mop, 3) => 1)',
    Vect[IsingBimodule ⊠ A4Irrep]((D0, 1) => 2, (D1, 2) => 2),
    Vect[IsingBimodule ⊠ A4Irrep]((M, 3) => 2)',
    Vect[IsingBimodule ⊠ A4Irrep]((C0, 2) => 3, (C1, 3) => 2),
)

for V in (Vtr, VRepℤ₂, VRepℤ₃, VRepU₁, VfRepℤ₂, VTwistedVecℤ₄, VTwistedVecℤ₃, VRepCU₁, VRepSU₂, VfRepCU₁, VfRepSU₂, VfHubbard, Vfib, VRepA4, VfRepA4, VRepA4Twistedℤ₄, VIBM, VIBMRepA4)
    V1, V2, V3, V4, V5 = V
    for Vi in (V1, V2, V3, V4, V5)
        @test allequal(leftunit, sectors(Vi))
        @test allequal(rightunit, sectors(Vi))
    end
    @test dim(V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)') > 0
    @test V1 ⊗ V2 ≾ (V3 ⊗ V4 ⊗ V5)' # wide blocks for QR
    @test V1 ⊗ V2 ⊗ V3 ≿ (V4 ⊗ V5)' # tall blocks for isometries
end

# Spacelist selection
# -------------------
function default_spacelist(fast_tests::Bool)
    fast_tests && return (Vtr, VRepSU₂, VfHubbard, VRepA4Twistedℤ₄)
    if get(ENV, "CI", "false") == "true"
        println("Detected running on CI")
        Sys.iswindows() && return (Vtr, VRepU₁, VfRepℤ₂, VTwistedVecℤ₃, VfRepSU₂, Vfib, VIBM) # length 7
        Sys.isapple()   && return (Vtr, VRepℤ₂, VRepCU₁, VfHubbard, VRepA4Twistedℤ₄, VIBMRepA4) # length 6
        return (Vtr, VRepℤ₃, VTwistedVecℤ₄, VRepSU₂, VfRepCU₁, VRepA4, VfRepA4) # length 7
    end
    return (Vtr, VRepℤ₂, VRepℤ₃, VRepU₁, VfRepℤ₂, VTwistedVecℤ₄, VTwistedVecℤ₃, VRepCU₁, VRepSU₂, VfRepCU₁, VfRepSU₂, VfHubbard, Vfib, VRepA4, VfRepA4, VRepA4Twistedℤ₄, VIBM, VIBMRepA4)
end

function factorization_spacelist(fast_tests::Bool)
    fast_tests && return (Vtr, VRepU₁, VfHubbard, VRepA4Twistedℤ₄)
    if get(ENV, "CI", "false") == "true"
        println("Detected running on CI")
        Sys.iswindows() && return (Vtr, VRepU₁, VfRepℤ₂, VTwistedVecℤ₃, VfRepSU₂, Vfib, VIBM) # length 7
        Sys.isapple()   && return (Vtr, VRepℤ₂, VRepCU₁, VfHubbard, VRepA4Twistedℤ₄, VIBMRepA4) # length 6
        return (Vtr, VRepℤ₃, VTwistedVecℤ₄, VRepSU₂, VfRepCU₁, VRepA4, VfRepA4) # length 7
    end
    return (Vtr, VRepℤ₂, VRepℤ₃, VRepU₁, VfRepℤ₂, VTwistedVecℤ₄, VTwistedVecℤ₃, VRepCU₁, VRepSU₂, VfRepCU₁, VfRepSU₂, VfHubbard, Vfib, VRepA4, VfRepA4, VRepA4Twistedℤ₄, VIBM, VIBMRepA4)
end

function ad_spacelist(fast_tests::Bool)
    return fast_tests ? (Vtr, VRepU₁, VfHubbard, VRepA4Twistedℤ₄) : (Vtr, VRepℤ₂, VRepCU₁, VfHubbard, VRepA4Twistedℤ₄, VIBMRepA4)
end


# ChainRules test utilities
# -------------------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return randn!(similar(x))
end
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::DiagonalTensorMap)
    V = x.domain
    return DiagonalTensorMap(randn(eltype(x), reduceddim(V)), V)
end
ChainRulesTestUtils.rand_tangent(::AbstractRNG, ::VectorSpace) = NoTangent()
function ChainRulesTestUtils.test_approx(
        actual::AbstractTensorMap, expected::AbstractTensorMap, msg = ""; kwargs...
    )
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
    return nothing
end

function test_ad_rrule(f, args...; check_inferred = false, kwargs...)
    test_rrule(
        Zygote.ZygoteRuleConfig(), f, args...;
        rrule_f = rrule_via_ad, check_inferred, kwargs...
    )
    return nothing
end

end
