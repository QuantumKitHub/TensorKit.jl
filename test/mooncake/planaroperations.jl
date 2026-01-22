using Test, TestExtras
using TensorKit
using TensorOperations
using Mooncake
using Random

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup
using .TestSetup: _repartition

mode = Mooncake.ReverseMode
rng = Random.default_rng()

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
    (
        Vect[Z2Irrep](0 => 1, 1 => 1),
        Vect[Z2Irrep](0 => 1, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 2)',
        Vect[Z2Irrep](0 => 2, 1 => 3),
        Vect[Z2Irrep](0 => 2, 1 => 2),
    ),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2)',
        Vect[FermionParity](0 => 2, 1 => 1)',
        Vect[FermionParity](0 => 2, 1 => 3),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
    (
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 1, -1 => 1),
        Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)',
        Vect[U1Irrep](0 => 1, 1 => 1, -1 => 2),
        Vect[U1Irrep](0 => 1, 1 => 2, -1 => 1)',
    ),
    (
        Vect[SU2Irrep](0 => 2, 1 // 2 => 1),
        Vect[SU2Irrep](0 => 1, 1 => 1),
        Vect[SU2Irrep](1 // 2 => 1, 1 => 1)',
        Vect[SU2Irrep](1 // 2 => 2),
        Vect[SU2Irrep](0 => 1, 1 // 2 => 1, 3 // 2 => 1)',
    ),
    # (
    #     Vect[FibonacciAnyon](:I => 2, :τ => 1),
    #     Vect[FibonacciAnyon](:I => 1, :τ => 2)',
    #     Vect[FibonacciAnyon](:I => 2, :τ => 2)',
    #     Vect[FibonacciAnyon](:I => 2, :τ => 3),
    #     Vect[FibonacciAnyon](:I => 2, :τ => 2),
    # ),
)
eltypes = (Float64,) # no complex support yet

@timedtestset "Mooncake - PlanarOperations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = precision(T)
    rtol = precision(T)

    @timedtestset "planarcontract!" begin
        for _ in 1:5
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
            Mooncake.TestUtils.test_rule(
                rng, TensorKit.planarcontract!, C, A, pA, B, pB, pAB, α, β;
                atol, rtol, mode, is_primitive = false
            )
        end
    end

    # TODO: currently broken
    # @timedtestset "planartrace!" begin
    #     for _ in 1:5
    #         k1 = rand(0:2)
    #         k2 = rand(0:1)
    #         V1 = map(v -> rand(Bool) ? v' : v, rand(V, k1))
    #         V2 = map(v -> rand(Bool) ? v' : v, rand(V, k2))
    #         V3 = prod(x -> x ⊗ x', V2[1:k2]; init = one(V[1]))
    #         V4 = prod(x -> x ⊗ x', V2[(k2 + 1):end]; init = one(V[1]))
    #
    #         k′ = rand(0:(k1 + 2k2))
    #         (_p, _q) = randcircshift(k′, k1 + 2k2 - k′, k1)
    #         p = _repartition(_p, rand(0:k1))
    #         q = (tuple(_q[1:2:end]...), tuple(_q[2:2:end]...))
    #         ip = _repartition(invperm(linearize((_p, _q))), k′)
    #         A = randn(T, permute(prod(V1) ⊗ V3 ← V4, ip))
    #
    #         α = randn(T)
    #         β = randn(T)
    #         C = randn!(TensorOperations.tensoralloc_add(T, A, p, false, Val(false)))
    #         Mooncake.TestUtils.test_rule(
    #             rng, TensorKit.planartrace!,
    #             C, A, p, q, α, β,
    #             TensorOperations.DefaultBackend(), TensorOperations.DefaultAllocator();
    #             atol, rtol, mode
    #         )
    #     end
    # end
end
