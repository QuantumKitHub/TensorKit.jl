using Test, TestExtras
using TensorKit
using TensorOperations
using Mooncake
using Random

mode = Mooncake.ReverseMode
rng = Random.default_rng()
is_primitive = false

function randindextuple(N::Int, k::Int = rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end

const _repartition = @static if isdefined(Base, :get_extension)
    Base.get_extension(TensorKit, :TensorKitMooncakeExt)._repartition
else
    TensorKit.TensorKitMooncakeExt._repartition
end

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

for V in spacelist
    I = sectortype(eltype(V))
    Istr = TensorKit.type_repr(I)

    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
    println("---------------------------------------")
    println("Mooncake with symmetry: $Istr")
    println("---------------------------------------")
    eltypes = (Float64,) # no complex support yet

    @timedtestset "VectorInterface with scalartype $T" for T in eltypes
        atol = precision(T)
        rtol = precision(T)

        C = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        A = randn(T, V[1] ⊗ V[2] ← V[3] ⊗ V[4] ⊗ V[5])
        α = randn(T)
        β = randn(T)

        Mooncake.TestUtils.test_rule(rng, scale!, C, α; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, scale!, C', α; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, scale!, C, A, α; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, scale!, C', A', α; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, scale!, copy(C'), A', α; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, scale!, C', copy(A'), α; atol, rtol, mode)

        Mooncake.TestUtils.test_rule(rng, add!, C, A; atol, rtol, mode, is_primitive = false)
        Mooncake.TestUtils.test_rule(rng, add!, C, A, α; atol, rtol, mode, is_primitive = false)
        Mooncake.TestUtils.test_rule(rng, add!, C, A, α, β; atol, rtol, mode)

        Mooncake.TestUtils.test_rule(rng, inner, C, A; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, inner, C', A'; atol, rtol, mode)
    end

    symmetricbraiding && @timedtestset "TensorOperations with scalartype $T" for T in eltypes
        atol = precision(T)
        rtol = precision(T)

        @timedtestset "tensorcontract!" begin
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

                for conjA in (false, true), conjB in (false, true)
                    A = randn(T, permute(V1 ← (conjA ? V2_conj : V2), ipA))
                    B = randn(T, permute((conjB ? V2_conj : V2) ← V3, ipB))
                    C = randn!(
                        TensorOperations.tensoralloc_contract(
                            T, A, pA, conjA, B, pB, conjB, pAB, Val(false)
                        )
                    )
                    Mooncake.TestUtils.test_rule(
                        rng, tensorcontract!, C, A, pA, conjA, B, pB, conjB, pAB, α, β;
                        atol, rtol, mode, is_primitive
                    )

                end
            end
        end
    end
end
