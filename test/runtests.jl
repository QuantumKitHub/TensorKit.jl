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
        fusiontensor(unit(I), unit(I), unit(I))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
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

if !is_buildkite
    Ti = time()
    @time include("fusiontrees.jl")
    @time include("spaces.jl")
    @time include("tensors.jl")
    @time include("factorizations.jl")
    @time include("diagonal.jl")
    @time include("planar.jl")
    if !(Sys.isapple() && get(ENV, "CI", "false") == "true") && isempty(VERSION.prerelease)
        @time include("ad.jl")
    end
    @time include("bugfixes.jl")
    Tf = time()
    printstyled(
        "Finished all tests in ",
        string(round((Tf - Ti) / 60; sigdigits = 3)),
        " minutes."; bold = true, color = Base.info_color()
    )
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
    printstyled(
        "Finished all GPU tests in ",
        string(round((Tf - Ti) / 60; sigdigits = 3)),
        " minutes."; bold = true, color = Base.info_color()
    )
    println()
end
