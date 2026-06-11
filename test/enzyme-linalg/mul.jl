using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
rTs = is_ci ? (Active,) : (Const, Active)
fTs = is_ci ? (Duplicated,) : (Const, Duplicated)

@timedtestset verbose = true "Enzyme - LinearAlgebra (mul):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)

        C = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        A = randn(T, codomain(C) ← V[5]' ⊗ V[4]')
        B = randn(T, domain(A) ← domain(C))
        α = randn(T)
        β = randn(T)
        @testset "mul: TC $TC, TA $TA, TB $TB" for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,)
            @testset "Tα $Tα, Tβ $Tβ" for Tα in rTs, Tβ in rTs
                EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB), (α, Tα), (β, Tβ); atol, rtol)
            end
            @testset "Tα $Tα, Tβ $Tβ" for Tα in fTs, Tβ in fTs
                EnzymeTestUtils.test_forward(mul!, TC, (C, TC), (A, TA), (B, TB), (α, Tα), (β, Tβ); atol, rtol)
            end
            EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB); atol, rtol)
            EnzymeTestUtils.test_forward(mul!, TC, (C, TC), (A, TA), (B, TB); atol, rtol)
        end
    end
end
