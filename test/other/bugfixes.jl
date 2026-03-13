using Test, TestExtras
using TensorKit
using TensorOperations
using LinearAlgebra: LinearAlgebra

@testset "BugfixConvert" begin
    v = randn(
        ComplexF64,
        (
            Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)](
                (-3, 1 / 2, 1) => 3, (-5, 1 / 2, 1) => 10, (-7, 1 / 2, 1) => 13,
                (-9, 1 / 2, 1) => 9, (-11, 1 / 2, 1) => 1, (-5, 3 / 2, 1) => 3,
                (-7, 3 / 2, 1) => 3, (-9, 3 / 2, 1) => 1
            ) ⊗ Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((1, 1 / 2, 1) => 1)'
        ) ← Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)](
            (-3, 1 / 2, 1) => 3, (-5, 1 / 2, 1) => 10, (-7, 1 / 2, 1) => 13,
            (-9, 1 / 2, 1) => 9, (-11, 1 / 2, 1) => 1, (-5, 3 / 2, 1) => 3,
            (-7, 3 / 2, 1) => 3, (-9, 3 / 2, 1) => 1
        )
    )
    w = convert(typeof(real(v)), v)
    @test w == v
    @test scalartype(w) == Float64
end

# https://github.com/quantumkithub/TensorKit.jl/issues/178
@testset "Issue #178" begin
    t = rand(U1Space(1 => 1) ← U1Space(1 => 1)')
    a = convert(Array, t)
    @test a == zeros(size(a))
end

# https://github.com/quantumkithub/TensorKit.jl/issues/194
@testset "Issue #194" begin
    t1 = rand(ℂ^4 ← ℂ^4)
    t2 = tensoralloc(typeof(t1), space(t1), Val(true), TensorOperations.ManualAllocator())
    t3 = similar(t2, ComplexF64, space(t1))
    @test storagetype(t3) == Vector{ComplexF64}
    t4 = similar(t2, domain(t1))
    @test storagetype(t4) == Vector{Float64}
    t5 = similar(t2)
    @test storagetype(t5) == Vector{Float64}
    tensorfree!(t2)
end
