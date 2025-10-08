module TestSetup

export smallset, randsector, hasfusiontensor, force_planar
export sectorlist
export Vtr, Vℤ₂, Vfℤ₂, Vℤ₃, VU₁, VfU₁, VCU₁, VSU₂, VfSU₂, VSU₂U₁, Vfib

using Test
using TestExtras
using Random
using TensorKit
using Combinatorics
using TensorKit: ProductSector, fusiontensor, pentagon_equation, hexagon_equation
using TensorKit: ℙ, PlanarTrivial
using TensorOperations
using Base.Iterators: take, product
using LinearAlgebra: LinearAlgebra

Random.seed!(1234)

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
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(I::Type{<:Sector})
    try
        fusiontensor(one(I), one(I), one(I))
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

sectorlist = (
    Z2Irrep, Z3Irrep, Z4Irrep, Z3Irrep ⊠ Z4Irrep,
    U1Irrep, CU1Irrep, SU2Irrep,
    FermionParity, FermionParity ⊠ FermionParity,
    FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ SU2Irrep, # Hubbard-like
    FibonacciAnyon, IsingAnyon,
    Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon,
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

end
