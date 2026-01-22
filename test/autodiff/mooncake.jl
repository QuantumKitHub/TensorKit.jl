using Test, TestExtras
using TensorKit
using TensorOperations
using Mooncake
using Random
using TupleTools

mode = Mooncake.ReverseMode
rng = Random.default_rng()
is_primitive = false

function randindextuple(N::Int, k::Int = rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end
function randcircshift(N₁::Int, N₂::Int, k::Int = rand(0:(N₁ + N₂)))
    N = N₁ + N₂
    @assert 0 ≤ k ≤ N
    p = TupleTools.vcat(ntuple(identity, N₁), reverse(ntuple(identity, N₂) .+ N₁))
    n = rand(0:N)
    _p = TupleTools.circshift(p, n)
    return (tuple(_p[1:k]...), reverse(tuple(_p[(k + 1):end]...)))
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
    # (
    #     Vect[FibonacciAnyon](:I => 2, :τ => 1),
    #     Vect[FibonacciAnyon](:I => 1, :τ => 2)',
    #     Vect[FibonacciAnyon](:I => 2, :τ => 2)',
    #     Vect[FibonacciAnyon](:I => 2, :τ => 3),
    #     Vect[FibonacciAnyon](:I => 2, :τ => 2),
    # ),
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

    @timedtestset "LinearAlgebra with scalartype $T" for T in eltypes
        atol = precision(T)
        rtol = precision(T)

        C = randn(T, V[1] ⊗ V[2] ← V[5])
        A = randn(T, codomain(C) ← V[3] ⊗ V[4])
        B = randn(T, domain(A) ← domain(C))
        α = randn(T)
        β = randn(T)

        Mooncake.TestUtils.test_rule(rng, mul!, C, A, B, α, β; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, mul!, C, A, B; atol, rtol, mode, is_primitive = false)

        Mooncake.TestUtils.test_rule(rng, norm, C, 2; atol, rtol, mode)
        Mooncake.TestUtils.test_rule(rng, norm, C', 2; atol, rtol, mode)
    end


    @timedtestset "Index manipulations with scalartype $T" for T in eltypes
        atol = precision(T)
        rtol = precision(T)

        symmetricbraiding && @timedtestset "add_permute!" begin
            A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
            α = randn(T)
            β = randn(T)

            # repeat a couple times to get some distribution of arrows
            for _ in 1:5
                p = randindextuple(numind(A))
                C = randn!(permute(A, p))
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_permute!, C, A, p, α, β; atol, rtol, mode)
                A = C
            end
        end

        @timedtestset "add_transpose!" begin
            A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
            α = randn(T)
            β = randn(T)

            # repeat a couple times to get some distribution of arrows
            for _ in 1:5
                p = randcircshift(numout(A), numin(A))
                C = randn!(transpose(A, p))
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, A, p, α, β; atol, rtol, mode)
                A = C
            end
        end

        @timedtestset "add_braid!" begin
            A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
            α = randn(T)
            β = randn(T)

            # repeat a couple times to get some distribution of arrows
            for _ in 1:5
                p = randcircshift(numout(A), numin(A))
                levels = tuple(randperm(numind(A)))
                C = randn!(transpose(A, p))
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, A, p, α, β; atol, rtol, mode)
                A = C
            end
        end

        @timedtestset "flip_n_twist!" begin
            A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; inv = false), twist!, A, 1; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; inv = true), twist!, A, [1, 3]; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, twist!, A, 1; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, twist!, A, [1, 3]; atol, rtol, mode)

            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; inv = false), flip, A, 1; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; inv = true), flip, A, [1, 3]; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, flip, A, 1; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, flip, A, [1, 3]; atol, rtol, mode)
        end

        @timedtestset "insert and remove units" begin
            A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])

            for insertunit in (insertleftunit, insertrightunit)
                Mooncake.TestUtils.test_rule(rng, insertunit, A, Val(1); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, insertunit, A, Val(4); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, insertunit, A', Val(2); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = false), insertunit, A, Val(1); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = true), insertunit, A, Val(2); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = false, dual = true, conj = true), insertunit, A, Val(3); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = false, dual = true, conj = true), insertunit, A', Val(3); atol, rtol, mode)
            end

            for i in 1:4
                B = insertleftunit(A, i; dual = rand(Bool))
                Mooncake.TestUtils.test_rule(rng, removeunit, B, Val(i); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = false), removeunit, B, Val(i); atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = true), removeunit, B, Val(i); atol, rtol, mode)
            end
        end
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
                        atol, rtol, mode
                    )

                end
            end
        end

        @timedtestset "trace_permute!" begin
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

    @timedtestset "PlanarOperations with scalartype $T" for T in eltypes
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

        @timedtestset "planartrace!" begin
            for _ in 1:5
                k1 = rand(0:2)
                k2 = rand(1:2)
                V1 = map(v -> rand(Bool) ? v' : v, rand(V, k1))
                V2 = map(v -> rand(Bool) ? v' : v, rand(V, k2))

                k′ = rand(0:(k1 + 2k2))
                (_p, _q) = randcircshift(k′, k1 + 2 * k2 - k′, k1)
                p = _repartition(_p, rand(0:k1))
                q = _repartition(_q, k2)
                ip = _repartition(invperm(linearize((_p, _q))), k′)
                A = randn(T, permute(prod(V1) ⊗ prod(V2) ← prod(V2), ip))

                α = randn(T)
                β = randn(T)
                C = randn!(TensorOperations.tensoralloc_add(T, A, p, false, Val(false)))
                Mooncake.TestUtils.test_rule(
                    rng, TensorKit.planartrace!,
                    C, A, p, q, α, β,
                    TensorOperations.DefaultBackend(), TensorOperations.DefaultAllocator();
                    atol, rtol, mode
                )
            end
        end
    end
end
