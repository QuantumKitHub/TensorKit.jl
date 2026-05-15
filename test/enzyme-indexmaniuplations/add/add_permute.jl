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

@timedtestset "Enzyme - Index Manipulations (add_permute!):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

        symmetricbraiding && @timedtestset "add_permute!" begin
            # repeat a couple times to get some distribution of arrows
            for ri in 1:5
                @testset for Tα in Tαs, Tβ in Tβs
                    A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
                    α = randn(T)
                    β = randn(T)
                    p = randindextuple(numind(A))
                    C = randn!(permute(A, p))
                    EnzymeTestUtils.test_reverse(TensorKit.add_permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                end
            end
        end
    end
end
