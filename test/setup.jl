module TestSetup

export smallset, randsector, hasfusiontensor, force_planar
export random_fusion
export sectorlist
export test_dim_isapprox
export Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, Vfib, VIB_diag, VIB_M

using Random
using Test: @test
using TensorKit
using TensorKit: ℙ, PlanarTrivial
using TensorKitSectors
using Base.Iterators: take, product

Random.seed!(123456)

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
    isa(UnitStyle(I), GenericUnit) && return false
    try
        TensorKit.fusiontensor(unit(I), unit(I), unit(I))
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
    force_planar(obj)

Replace an object with a planar equivalent -- i.e. one that disallows braiding.
"""
force_planar(V::ComplexSpace) = isdual(V) ? (ℙ^dim(V))' : ℙ^dim(V)
function force_planar(V::GradedSpace)
    return GradedSpace((c ⊠ PlanarTrivial() => dim(V, c) for c in sectors(V))..., isdual(V))
end
force_planar(V::ProductSpace) = mapreduce(force_planar, ⊗, V)
function force_planar(tsrc::TensorMap{<:Any, ComplexSpace})
    tdst = TensorMap{scalartype(tsrc)}(
        undef,
        force_planar(codomain(tsrc)) ←
            force_planar(domain(tsrc))
    )
    copyto!(block(tdst, PlanarTrivial()), block(tsrc, Trivial()))
    return tdst
end
function force_planar(tsrc::TensorMap{<:Any, <:GradedSpace})
    tdst = TensorMap{scalartype(tsrc)}(
        undef,
        force_planar(codomain(tsrc)) ←
            force_planar(domain(tsrc))
    )
    for (c, b) in blocks(tsrc)
        copyto!(block(tdst, c ⊠ PlanarTrivial()), b)
    end
    return tdst
end

function random_fusion(I::Type{<:Sector}, ::Val{N}) where {N} # for fusion tree tests
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

# helper function to check that d - dim(c) < dim(V) <= d where c is the largest sector
# to allow for truncations to have some margin with larger sectors
function test_dim_isapprox(V::ElementarySpace, d::Int)
    dim_c_max = maximum(dim, sectors(V); init = 1)
    return @test max(0, d - dim_c_max) ≤ dim(V) ≤ d + dim_c_max
end
function test_dim_isapprox(V::ProductSpace, d::Int)
    dim_c_max = maximum(dim, blocksectors(V); init = 1)
    return @test max(0, d - dim_c_max) ≤ dim(V) ≤ d + dim_c_max
end

uniquefusionsectorlist = (
    Z2Irrep, Z3Irrep, Z4Irrep, Z3Irrep ⊠ Z4Irrep,
    U1Irrep, FermionParity, FermionParity ⊠ FermionParity, FermionNumber,
    Z3Element{1}, ZNElement{5, 2},
)
simplefusionsectorlist = (
    CU1Irrep, SU2Irrep, FibonacciAnyon, IsingAnyon,
    FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ SU2Irrep,
    Z3Element{1} ⊠ FibonacciAnyon ⊠ FibonacciAnyon,
)
genericfusionsectorlist = (
    A4Irrep, A4Irrep ⊠ FermionParity, A4Irrep ⊠ SU2Irrep, A4Irrep ⊠ Z3Element{2}, A4Irrep ⊠ A4Irrep,
)
multifusionsectorlist = (
    IsingBimodule, IsingBimodule ⊠ SU2Irrep, IsingBimodule ⊠ IsingBimodule, IsingBimodule ⊠ Z3Element{1}, IsingBimodule ⊠ FibonacciAnyon ⊠ A4Irrep,
)

sectorlist = (
    uniquefusionsectorlist...,
    simplefusionsectorlist...,
    genericfusionsectorlist...,
    multifusionsectorlist...,
)

# spaces
Vtr = (ℂ^2, (ℂ^3)', ℂ^4, ℂ^3, (ℂ^2)')
Vℤ₂ = (
    Vect[Z2Irrep](0 => 1, 1 => 1),
    Vect[Z2Irrep](0 => 1, 1 => 2)',
    Vect[Z2Irrep](0 => 3, 1 => 2)',
    Vect[Z2Irrep](0 => 2, 1 => 3),
    Vect[Z2Irrep](0 => 2, 1 => 5),
)
Vfℤ₂ = (
    Vect[FermionParity](0 => 1, 1 => 1),
    Vect[FermionParity](0 => 1, 1 => 2)',
    Vect[FermionParity](0 => 2, 1 => 1)',
    Vect[FermionParity](0 => 2, 1 => 3),
    Vect[FermionParity](0 => 2, 1 => 5),
)
Vℤ₃ = (
    Vect[Z3Irrep](0 => 1, 1 => 2, 2 => 1),
    Vect[Z3Irrep](0 => 2, 1 => 1, 2 => 1),
    Vect[Z3Irrep](0 => 1, 1 => 2, 2 => 1)',
    Vect[Z3Irrep](0 => 1, 1 => 2, 2 => 3),
    Vect[Z3Irrep](0 => 1, 1 => 3, 2 => 3)',
)
VU₁ = (
    Vect[U1Irrep](0 => 1, 1 => 2, -1 => 2),
    Vect[U1Irrep](0 => 3, 1 => 1, -1 => 1),
    Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
    Vect[U1Irrep](0 => 1, 1 => 2, -1 => 3),
    Vect[U1Irrep](0 => 1, 1 => 3, -1 => 3)',
)
VfU₁ = (
    Vect[FermionNumber](0 => 1, 1 => 2, -1 => 2),
    Vect[FermionNumber](0 => 3, 1 => 1, -1 => 1),
    Vect[FermionNumber](0 => 2, 1 => 2, -1 => 1)',
    Vect[FermionNumber](0 => 1, 1 => 2, -1 => 3),
    Vect[FermionNumber](0 => 1, 1 => 3, -1 => 3)',
)
VCU₁ = (
    Vect[CU1Irrep]((0, 0) => 1, (0, 1) => 2, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 3, (0, 1) => 0, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 1, (0, 1) => 0, 1 => 2)',
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 2, 1 => 1),
    Vect[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 2)',
)
VSU₂ = (
    Vect[SU2Irrep](0 => 3, 1 // 2 => 1),
    Vect[SU2Irrep](0 => 2, 1 => 1),
    Vect[SU2Irrep](1 // 2 => 1, 1 => 1)',
    Vect[SU2Irrep](0 => 2, 1 // 2 => 2),
    Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
)
VfSU₂ = (
    Vect[FermionSpin](0 => 3, 1 // 2 => 1),
    Vect[FermionSpin](0 => 2, 1 => 1),
    Vect[FermionSpin](1 // 2 => 1, 1 => 1)',
    Vect[FermionSpin](0 => 2, 1 // 2 => 2),
    Vect[FermionSpin](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
)
VSU₂U₁ = (
    Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 1, (1 // 2, -1) => 1),
    Vect[SU2Irrep ⊠ U1Irrep](
        (0, 0) => 2, (0, 2) => 1, (1, 0) => 1, (1, -2) => 1,
        (1 // 2, -1) => 1
    ),
    Vect[SU2Irrep ⊠ U1Irrep]((1 // 2, 1) => 1, (1, -2) => 1)',
    Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 2, (0, 2) => 1, (1 // 2, 1) => 1),
    Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 1, (1 // 2, 1) => 1)',
)
Vfib = (
    Vect[FibonacciAnyon](:I => 1, :τ => 1),
    Vect[FibonacciAnyon](:I => 1, :τ => 2)',
    Vect[FibonacciAnyon](:I => 3, :τ => 2)',
    Vect[FibonacciAnyon](:I => 2, :τ => 3),
    Vect[FibonacciAnyon](:I => 2, :τ => 2),
)

C0, C1 = IsingBimodule(1, 1, 0), IsingBimodule(1, 1, 1)
D0, D1 = IsingBimodule(2, 2, 0), IsingBimodule(2, 2, 1)
M, Mop = IsingBimodule(1, 2, 0), IsingBimodule(2, 1, 0)
VIB_diag = (
    Vect[IsingBimodule](C0 => 1, C1 => 2),
    Vect[IsingBimodule](C0 => 2, C1 => 1),
    Vect[IsingBimodule](C0 => 3, C1 => 1),
    Vect[IsingBimodule](C0 => 2, C1 => 3),
    Vect[IsingBimodule](C0 => 3, C1 => 2),
)

# not a random ordering! designed to make V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 work (tensors)
# while V1 ⊗ V2 ← V4 isn't empty (factorizations)
VIB_M = (
    Vect[IsingBimodule](C0 => 1, C1 => 2),
    Vect[IsingBimodule](M => 3),
    Vect[IsingBimodule](C0 => 2, C1 => 3),
    Vect[IsingBimodule](M => 4),
    Vect[IsingBimodule](D0 => 3, D1 => 4),
)

end
