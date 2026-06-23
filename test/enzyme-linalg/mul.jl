using Test, TestExtras
using TensorKit
using VectorInterface
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"

@timedtestset verbose = true "Enzyme - LinearAlgebra (mul):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)

        # see https://github.com/QuantumKitHub/TensorKit.jl/issues/457
        @static if VERSION < v"1.11.0-rc"
            C = randn(T, V[1] ⊗ V[2] ← (V[4] ⊗ V[5])')
        else
            C = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        end
        A = randn(T, codomain(C) ← V[5]' ⊗ V[4]')
        B = randn(T, domain(A) ← domain(C))
        zero_αβs = ((Zero(), Zero()), (randn(T), Zero()), (Zero(), randn(T)))
        αβs = !is_ci ? vcat(zero_αβs..., (randn(T), randn(T))) : ((randn(T), randn(T)),)
        for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,)
            for (α, β) in αβs
                rTαs = if α === Zero()
                    (Const,)
                elseif !is_ci
                    (Active, Const)
                else
                    (Active,)
                end
                rTβs = if β === Zero()
                    (Const,)
                elseif !is_ci
                    (Active, Const)
                else
                    (Active,)
                end
                for Tα in rTαs, Tβ in rTβs
                    EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB), (α, Tα), (β, Tβ); atol, rtol, testset_name = "mul! reverse Tα $Tα, Tβ $Tβ")
                end
                fTαs = if α === Zero()
                    (Const,)
                elseif !is_ci
                    (Duplicated, Const)
                else
                    (Duplicated,)
                end
                fTβs = if β === Zero()
                    (Const,)
                elseif !is_ci
                    (Duplicated, Const)
                else
                    (Duplicated,)
                end
                for Tα in fTαs, Tβ in fTβs
                    EnzymeTestUtils.test_forward(mul!, TC, (C, TC), (A, TA), (B, TB), (α, Tα), (β, Tβ); atol, rtol, testset_name = "mul! forward Tα $Tα, Tβ $Tβ")
                end
            end
            if !is_ci
                EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB); atol, rtol, testset_name = "mul! reverse no α no β")
                EnzymeTestUtils.test_forward(mul!, TC, (C, TC), (A, TA), (B, TB); atol, rtol, testset_name = "mul! forward no α no β")
            end
        end
    end
end
