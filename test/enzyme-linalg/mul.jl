using Test, TestExtras
using TensorKit
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset verbose = true "Enzyme - LinearAlgebra (mul):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)

        C = randn(T, V[1] ⊗ V[2] ← V[5])
        A = randn(T, codomain(C) ← V[3] ⊗ V[4])
        B = randn(T, domain(A) ← domain(C))
        α = randn(T)
        β = randn(T)

        @testset "mul: TC $TC, TA $TA, TB $TB" for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,)
            @testset "Tα $Tα, Tβ $Tβ" for Tα in (Active, Const), Tβ in (Active, Const)
                EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB), (α, Tα), (β, Tβ); atol, rtol)
            end
            EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB); atol, rtol)
        end
    end
end
