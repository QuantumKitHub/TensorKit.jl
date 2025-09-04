using Test
using TestExtras
using Random
using TensorKit
using Combinatorics
using TensorKit: ProductSector, fusiontensor, pentagon_equation, hexagon_equation
using TensorOperations
using Base.Iterators: take, product
# using SUNRepresentations: SUNIrrep
# const SU3Irrep = SUNIrrep{3}
using LinearAlgebra: LinearAlgebra
using Zygote: Zygote

const TK = TensorKit

Random.seed!(1234)

# don't run all tests on GPU, only the GPU
# specific ones
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

smallset(::Type{I}) where {I<:Sector} = take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1,I2}}}) where {I1,I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i, j) in iter if dim(i) * dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1,I2,I3}}}) where {I1,I2,I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i, j, k) in iter if dim(i) * dim(j) * dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I<:Sector}
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

sectorlist = (Z2Irrep, Z3Irrep, Z4Irrep, Z3Irrep ⊠ Z4Irrep,
              U1Irrep, CU1Irrep, SU2Irrep,
              FermionParity, FermionParity ⊠ FermionParity,
              FermionParity ⊠ U1Irrep ⊠ SU2Irrep, FermionParity ⊠ SU2Irrep ⊠ SU2Irrep, # Hubbard-like
              FibonacciAnyon, IsingAnyon,
              Z2Irrep ⊠ FibonacciAnyon ⊠ FibonacciAnyon)

# spaces
Vtr = (ℂ^3,
       (ℂ^4)',
       ℂ^5,
       ℂ^6,
       (ℂ^7)')
Vℤ₂ = (ℂ[Z2Irrep](0 => 1, 1 => 1),
       ℂ[Z2Irrep](0 => 1, 1 => 2)',
       ℂ[Z2Irrep](0 => 3, 1 => 2)',
       ℂ[Z2Irrep](0 => 2, 1 => 3),
       ℂ[Z2Irrep](0 => 2, 1 => 5))
Vfℤ₂ = (ℂ[FermionParity](0 => 1, 1 => 1),
        ℂ[FermionParity](0 => 1, 1 => 2)',
        ℂ[FermionParity](0 => 3, 1 => 2)',
        ℂ[FermionParity](0 => 2, 1 => 3),
        ℂ[FermionParity](0 => 2, 1 => 5))
Vℤ₃ = (ℂ[Z3Irrep](0 => 1, 1 => 2, 2 => 2),
       ℂ[Z3Irrep](0 => 3, 1 => 1, 2 => 1),
       ℂ[Z3Irrep](0 => 2, 1 => 2, 2 => 1)',
       ℂ[Z3Irrep](0 => 1, 1 => 2, 2 => 3),
       ℂ[Z3Irrep](0 => 1, 1 => 3, 2 => 3)')
VU₁ = (ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 2),
       ℂ[U1Irrep](0 => 3, 1 => 1, -1 => 1),
       ℂ[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
       ℂ[U1Irrep](0 => 1, 1 => 2, -1 => 3),
       ℂ[U1Irrep](0 => 1, 1 => 3, -1 => 3)')
VfU₁ = (ℂ[FermionNumber](0 => 1, 1 => 2, -1 => 2),
        ℂ[FermionNumber](0 => 3, 1 => 1, -1 => 1),
        ℂ[FermionNumber](0 => 2, 1 => 2, -1 => 1)',
        ℂ[FermionNumber](0 => 1, 1 => 2, -1 => 3),
        ℂ[FermionNumber](0 => 1, 1 => 3, -1 => 3)')
VCU₁ = (ℂ[CU1Irrep]((0, 0) => 1, (0, 1) => 2, 1 => 1),
        ℂ[CU1Irrep]((0, 0) => 3, (0, 1) => 0, 1 => 1),
        ℂ[CU1Irrep]((0, 0) => 1, (0, 1) => 0, 1 => 2)',
        ℂ[CU1Irrep]((0, 0) => 2, (0, 1) => 2, 1 => 1),
        ℂ[CU1Irrep]((0, 0) => 2, (0, 1) => 1, 1 => 2)')
VSU₂ = (ℂ[SU2Irrep](0 => 3, 1 // 2 => 1),
        ℂ[SU2Irrep](0 => 2, 1 => 1),
        ℂ[SU2Irrep](1 // 2 => 1, 1 => 1)',
        ℂ[SU2Irrep](0 => 2, 1 // 2 => 2),
        ℂ[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)')
VfSU₂ = (ℂ[FermionSpin](0 => 3, 1 // 2 => 1),
         ℂ[FermionSpin](0 => 2, 1 => 1),
         ℂ[FermionSpin](1 // 2 => 1, 1 => 1)',
         ℂ[FermionSpin](0 => 2, 1 // 2 => 2),
         ℂ[FermionSpin](0 => 1, 1 // 2 => 1, 3 // 2 => 1)')
VSU₂U₁ = (Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 1, (1 // 2, -1) => 1),
          Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 2, (0, 2) => 1, (1, 0) => 1, (1, -2) => 1,
                                   (1 // 2, -1) => 1),
          Vect[SU2Irrep ⊠ U1Irrep]((1 // 2, 1) => 1, (1, -2) => 1)',
          Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 2, (0, 2) => 1, (1 // 2, 1) => 1),
          Vect[SU2Irrep ⊠ U1Irrep]((0, 0) => 1, (1 // 2, 1) => 1)')
# VSU₃ = (ℂ[SU3Irrep]((0, 0, 0) => 3, (1, 0, 0) => 1),
#     ℂ[SU3Irrep]((0, 0, 0) => 3, (2, 0, 0) => 1)',
#     ℂ[SU3Irrep]((1, 1, 0) => 1, (2, 1, 0) => 1),
#     ℂ[SU3Irrep]((1, 0, 0) => 1, (2, 0, 0) => 1),
#     ℂ[SU3Irrep]((0, 0, 0) => 1, (1, 0, 0) => 1, (1, 1, 0) => 1)')

if !is_buildkite
    Ti = time()
    include("fusiontrees.jl")
    include("spaces.jl")
    include("tensors.jl")
    include("diagonal.jl")
    include("planar.jl")
    # TODO: remove once we know AD is slow on macOS CI
    if !(Sys.isapple() && get(ENV, "CI", "false") == "true") && isempty(VERSION.prerelease)
        include("ad.jl")
    end
    include("bugfixes.jl")
    Tf = time()
    printstyled("Finished all tests in ",
                string(round((Tf - Ti) / 60; sigdigits=3)),
                " minutes."; bold=true, color=Base.info_color())
    println()
    @testset "Aqua" verbose = true begin
        using Aqua
        Aqua.test_all(TensorKit)
    end
else
    Ti = time()
    #=using CUDA
    if CUDA.functional()
    end
    using AMDGPU
    if AMDGPU.functional()
    end=#
    Tf = time()
    printstyled("Finished all GPU tests in ",
                string(round((Tf - Ti) / 60; sigdigits=3)),
                " minutes."; bold=true, color=Base.info_color())
    println()
end
