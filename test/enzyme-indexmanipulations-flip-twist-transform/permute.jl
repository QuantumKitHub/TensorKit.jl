using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
Tαs = is_ci ? (Active,) : (Active, Const)
Tβs = is_ci ? (Active,) : (Active, Const)

if VERSION >= v"1.11.0-rc" # segfault issues on 1.10
    @timedtestset "Enzyme - Index Manipulations (permute!): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        println(TensorKit.type_repr(sectortype(eltype(V))))
        atol = default_tol(T)
        rtol = default_tol(T)
        symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

        symmetricbraiding && @timedtestset "permute! Tα $Tα, Tβ $Tβ" for Tα in Tαs, Tβ in Tβs
            A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
            α = randn(T)
            β = randn(T)
            p = randindextuple(numind(A))
            C = randn!(permute(A, p))
            EnzymeTestUtils.test_reverse(TensorKit.permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
        end
    end
end
