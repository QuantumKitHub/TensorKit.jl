using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"

rTs(::Number, is_ci::Bool) = is_ci ? (Active,) : (Active, Const)
rTs(::Zero, is_ci::Bool) = (Const,)
rTs(::One, is_ci::Bool) = (Const,)
fTs(::Number, is_ci::Bool) = is_ci ? (Duplicated,) : (Duplicated, Const)
fTs(::Zero, is_ci::Bool) = (Const,)
fTs(::One, is_ci::Bool) = (Const,)

@timedtestset verbose = true "Enzyme - LinearAlgebra (mul) and planarcontract!:" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)

        zero_αβs = ((Zero(), Zero()), (randn(T), Zero()), (Zero(), randn(T)))
        αβs = !is_ci ? vcat(zero_αβs..., (randn(T), randn(T))) : ((randn(T), randn(T)),)
        @timedtestset "mul" begin
            C = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
            A = randn(T, codomain(C) ← V[5]' ⊗ V[4]')
            B = randn(T, domain(A) ← domain(C))
            for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,)
                for (α, β) in αβs
                    rTαs = rTs(α, is_ci)
                    rTβs = rTs(β, is_ci)
                    fTαs = fTs(α, is_ci)
                    fTβs = fTs(β, is_ci)
                    for Tα in rTαs, Tβ in rTβs
                        EnzymeTestUtils.test_reverse(mul!, TC, (C, TC), (A, TA), (B, TB), (α, Tα), (β, Tβ); atol, rtol, testset_name = "mul! reverse Tα $Tα, Tβ $Tβ")
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
        @timedtestset "planarcontract!" begin
            V1, V2, V3, V4, V5 = V
            k1 = 3
            k2 = 2
            k3 = 3
            k′ = rand(0:(k1 + k2))
            pA = randcircshift(k′, k1 + k2 - k′, k1)
            ipA = _repartition(invperm(linearize(pA)), k′)
            k′ = rand(0:(k2 + k3))
            pB = randcircshift(k′, k2 + k3 - k′, k2)
            ipB = _repartition(invperm(linearize(pB)), k′)
            # TODO: primal value already is broken for this?
            # pAB = randcircshift(k1, k3)
            pAB = _repartition(tuple((1:(k1 + k3))...), k1)

            α_ = randn(T)
            β_ = randn(T)

            A = randn(T, permute(V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)', ipA))
            B = randn(T, permute((V4 ⊗ V5)' ← V1 ⊗ V2 ⊗ V3, ipB))
            C = randn!(
                TensorOperations.tensoralloc_contract(
                    T, A, pA, false, B, pB, false, pAB, Val(false)
                )
            )
            αβs = !is_ci ? ((One(), Zero()), (α_, β_)) : ((α_, β_),)
            @testset for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,), (α, β) in αβs
                rTαs = rTs(α, is_ci)
                rTβs = rTs(β, is_ci)
                fTαs = fTs(α, is_ci)
                fTβs = fTs(β, is_ci)
                for Tα in rTαs, Tβ in rTβs
                    EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "planarcontract! Tα = $Tα, Tβ = $Tβ")
                end
                # TODO broken internally in Enzyme?
                #for Tα in fTαs, Tβ in fTβs
                #    EnzymeTestUtils.test_forward(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "planarcontract! Tα = $Tα, Tβ = $Tβ")
                #end
            end
        end
    end
end
