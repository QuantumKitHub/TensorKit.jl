using Test, TestExtras
using TensorKit
using TensorKit: type_repr, SectorDict
using TensorOperations
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences: FiniteDifferences, central_fdm, forward_fdm
using Random
using LinearAlgebra
using Zygote
using MatrixAlgebraKit


# Tests
# -----

spacelist = (Vtr, Vℤ₂, Vfℤ₂, VSU₂)

for V in spacelist
    I = sectortype(eltype(V))
    Istr = type_repr(I)
    eltypes = isreal(sectortype(eltype(V))) ? (Float64, ComplexF64) : (ComplexF64,)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
    println("---------------------------------------")
    println("Auto-diff with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "AD with symmetry $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V

        symmetricbraiding &&
            @timedtestset "TensorOperations with scalartype $T" for T in eltypes
            atol = default_tol(T)
            rtol = default_tol(T)

            @timedtestset "tensortrace!" begin
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
                    for conjA in (false, true)
                        C = randn!(TensorOperations.tensoralloc_add(T, A, p, conjA, Val(false)))
                        test_rrule(tensortrace!, C, A, p, q, conjA, α, β; atol, rtol)
                    end
                end
            end

            @timedtestset "tensoradd!" begin
                A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
                α = randn(T)
                β = randn(T)

                # repeat a couple times to get some distribution of arrows
                for _ in 1:5
                    p = randindextuple(numind(A))

                    C1 = randn!(TensorOperations.tensoralloc_add(T, A, p, false, Val(false)))
                    test_rrule(tensoradd!, C1, A, p, false, α, β; atol, rtol)

                    C2 = randn!(TensorOperations.tensoralloc_add(T, A, p, true, Val(false)))
                    test_rrule(tensoradd!, C2, A, p, true, α, β; atol, rtol)

                    A = rand(Bool) ? C1 : C2
                end
            end

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
                        test_rrule(
                            tensorcontract!, C, A, pA, conjA, B, pB, conjB, pAB, α, β;
                            atol, rtol
                        )
                    end
                end
            end

            @timedtestset "tensorscalar" begin
                A = randn(T, ProductSpace{typeof(V[1]), 0}())
                test_rrule(tensorscalar, A)
            end
        end
    end
end

# https://github.com/quantumkithub/TensorKit.jl/issues/209
@testset "Issue #209" begin
    function f(T, D)
        @tensor T[1, 4, 1, 3] * D[3, 4]
    end
    V = Z2Space(2, 2)
    D = DiagonalTensorMap(randn(4), V)
    T = randn(V ⊗ V ← V ⊗ V)
    g1, = Zygote.gradient(f, T, D)
    g2, = Zygote.gradient(f, T, TensorMap(D))
    @test g1 ≈ g2
end
