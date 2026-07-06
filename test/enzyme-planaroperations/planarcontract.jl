using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - PlanarOperations (planarcontract): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
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

    α = randn(T)
    β = randn(T)

    A = randn(T, permute(V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)', ipA))
    B = randn(T, permute((V4 ⊗ V5)' ← V1 ⊗ V2 ⊗ V3, ipB))
    C = randn!(
        TensorOperations.tensoralloc_contract(
            T, A, pA, false, B, pB, false, pAB, Val(false)
        )
    )

    α = randn(T)
    β = randn(T)

    A = randn(T, permute(V1 ⊗ V2 ⊗ V3 ← (V4 ⊗ V5)', ipA))
    B = randn(T, permute((V4 ⊗ V5)' ← V1 ⊗ V2 ⊗ V3, ipB))
    C = randn!(
        TensorOperations.tensoralloc_contract(
            T, A, pA, false, B, pB, false, pAB, Val(false)
        )
    )
    @testset for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (One(), Const), (Zero(), Const); atol, rtol, testset_name = "planarcontract! α = One, β = Zero")
    end
    @testset for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,), Tα in (Const, Active), Tβ in (Const, Active)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "planarcontract! Tα = $Tα, Tβ = $Tβ")
    end
end
