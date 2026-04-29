using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)
#=
@timedtestset "Enzyme - PlanarOperations (planarcontract): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    d = 0
    local V1, V2, V3, k1, k2, k3
    # retry a couple times to make sure there are at least some nonzero elements
    for _ in 1:10
        k1 = rand(0:3)
        k2 = rand(0:2)
        k3 = rand(0:2)
        V1 = prod(v -> rand(Bool) ? v' : v, rand(V, k1); init = one(V[1]))
        V2 = prod(v -> rand(Bool) ? v' : v, rand(V, k2); init = one(V[1]))
        V3 = prod(v -> rand(Bool) ? v' : v, rand(V, k3); init = one(V[1]))
        d = min(dim(V1 ← V2), dim(V1' ← V2), dim(V2 ← V3), dim(V2' ← V3))
        d > 1 && break
    end
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

    A = randn(T, permute(V1 ← V2, ipA))
    B = randn(T, permute(V2 ← V3, ipB))
    C = randn!(
        TensorOperations.tensoralloc_contract(
            T, A, pA, false, B, pB, false, pAB, Val(false)
        )
    )
    for TC in (Duplicated,), TA in (Duplicated,), TB in (Duplicated,)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (One(), Const), (Zero(), Const); atol, rtol)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Const), (β, Const); atol, rtol)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Const), (β, Active); atol, rtol)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Active), (β, Const); atol, rtol)
        EnzymeTestUtils.test_reverse(TensorKit.planarcontract!, TC, (C, TC), (A, TA), (pA, Const), (B, TB), (pB, Const), (pAB, Const), (α, Active), (β, Active); atol, rtol)
    end
end=#
