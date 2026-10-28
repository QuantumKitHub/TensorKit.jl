using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
rTαs = is_ci ? (Active,) : (Active, Const)
rTβs = is_ci ? (Active,) : (Active, Const)
fTαs = is_ci ? (Duplicated,) : (Duplicated, Const)
fTβs = is_ci ? (Duplicated,) : (Duplicated, Const)

@timedtestset "Enzyme - Index Manipulations (permute!): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

    symmetricbraiding && @timedtestset "permute!" begin
        A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        α = randn(T)
        β = randn(T)
        p = randindextuple(numind(A))
        C = randn!(permute(A, p))
        @static if VERSION ≥ v"1.11.0-rc"
            for Tα in rTαs, Tβ in rTβs
                EnzymeTestUtils.test_reverse(TensorKit.permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "permute! reverse Tα: $Tα, Tβ: $Tβ")
            end
            for Tα in fTαs, Tβ in fTβs
                EnzymeTestUtils.test_forward(TensorKit.permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "permute! forward Tα: $Tα, Tβ: $Tβ")
            end
        end
    end
end
