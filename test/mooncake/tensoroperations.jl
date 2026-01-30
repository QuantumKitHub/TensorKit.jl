using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: One, Zero
using Mooncake
using Random

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

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
    (
        Vect[FibonacciAnyon](:I => 2, :τ => 1),
        Vect[FibonacciAnyon](:I => 1, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 2)',
        Vect[FibonacciAnyon](:I => 2, :τ => 3),
        Vect[FibonacciAnyon](:I => 2, :τ => 2),
    ),
)
eltypes = (Float64, ComplexF64)

@timedtestset "Mooncake - TensorOperations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

    symmetricbraiding && @timedtestset "tensorcontract!" begin
        for _ in 1:5
            d = 0
            local V1, V2, V3
            # retry a couple times to make sure there are at least some nonzero elements
            for _ in 1:10
                k1 = rand(0:3)
                k2 = rand(0:2)
                k3 = rand(0:2)
                V1 = prod(v -> rand(Bool) ? v' : v, rand(V, k1); init = one(V[1]))
                V2 = prod(v -> rand(Bool) ? v' : v, rand(V, k2); init = one(V[1]))
                V3 = prod(v -> rand(Bool) ? v' : v, rand(V, k3); init = one(V[1]))
                d = min(dim(V1 ← V2), dim(V1' ← V2), dim(V2 ← V3), dim(V2' ← V3))
                d > 0 && break
            end
            ipA = randindextuple(length(V1) + length(V2))
            pA = _repartition(invperm(linearize(ipA)), length(V1))
            ipB = randindextuple(length(V2) + length(V3))
            pB = _repartition(invperm(linearize(ipB)), length(V2))
            pAB = randindextuple(length(V1) + length(V3))

            α = randn(T)
            β = randn(T)
            V2_conj = prod(conj, V2; init = one(V[1]))

            A = randn(T, permute(V1 ← V2, ipA))
            B = randn(T, permute(V2 ← V3, ipB))
            C = randn!(
                TensorOperations.tensoralloc_contract(
                    T, A, pA, false, B, pB, false, pAB, Val(false)
                )
            )
            Mooncake.TestUtils.test_rule(
                rng, TensorKit.blas_contract!,
                C, A, pA, B, pB, pAB, One(), Zero(),
                TensorOperations.DefaultBackend(), TensorOperations.DefaultAllocator();
                atol, rtol, mode
            )
            Mooncake.TestUtils.test_rule(
                rng, TensorKit.blas_contract!,
                C, A, pA, B, pB, pAB, α, β,
                TensorOperations.DefaultBackend(), TensorOperations.DefaultAllocator();
                atol, rtol, mode
            )
            T <: Complex && Mooncake.TestUtils.test_rule(
                rng, TensorKit.blas_contract!,
                C, A, pA, B, pB, pAB, real(α), real(β),
                TensorOperations.DefaultBackend(), TensorOperations.DefaultAllocator();
                atol, rtol, mode
            )
        end
    end

    symmetricbraiding && @timedtestset "trace_permute!" begin
        for _ in 1:5
            k1 = rand(0:2)
            k2 = rand(1:2)
            V1 = map(v -> rand(Bool) ? v' : v, rand(V, k1))
            V2 = map(v -> rand(Bool) ? v' : v, rand(V, k2))

            (_p, _q) = randindextuple(k1 + 2 * k2, k1)
            p = _repartition(_p, rand(0:k1))
            q = _repartition(_q, k2)
            ip = _repartition(invperm(linearize((_p, _q))), rand(0:(k1 + 2 * k2)))
            A = randn(T, permute(prod(V1) ⊗ prod(V2) ← prod(V2), ip))

            α = randn(T)
            β = randn(T)
            C = randn!(TensorOperations.tensoralloc_add(T, A, p, false, Val(false)))
            Mooncake.TestUtils.test_rule(
                rng, TensorKit.trace_permute!, C, A, p, q, α, β, TensorOperations.DefaultBackend();
                atol, rtol, mode
            )
        end
    end
end
