using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Mooncake
using Random

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

mode = Mooncake.ReverseMode
rng = Random.default_rng()

spacelist = (
    (ℂ^2, (ℂ^3)', ℂ^3, ℂ^2, (ℂ^2)'),
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

@timedtestset "Mooncake - Index Manipulations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

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
        for _ in 1:2
            p = randcircshift(numout(A), numin(A))
            C = randn!(transpose(A, p))
            Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, A, p, One(), Zero(); atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, A, p, α, β; atol, rtol, mode)
            if !(T <: Real)
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, real(A), p, α, β; atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, A, p, real(α), β; atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_transpose!, C, real(A), p, real(α), β; atol, rtol, mode)
            end
            A = C
        end
    end

    @timedtestset "add_braid!" begin
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        α = randn(T)
        β = randn(T)

        # repeat a couple times to get some distribution of arrows
        for _ in 1:2
            p = randcircshift(numout(A), numin(A))
            levels = Tuple(randperm(numind(A)))
            C = randn!(transpose(A, p))
            Mooncake.TestUtils.test_rule(rng, TensorKit.add_braid!, C, A, p, levels, α, β; atol, rtol, mode)
            if !(T <: Real)
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_braid!, C, real(A), p, levels, α, β; atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_braid!, C, A, p, levels, real(α), β; atol, rtol, mode)
                Mooncake.TestUtils.test_rule(rng, TensorKit.add_braid!, C, A, p, levels, real(α), real(β); atol, rtol, mode)
            end
            A = C
        end
    end

    @timedtestset "flip_n_twist!" begin
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])

        if !(T <: Real && !(sectorscalartype(sectortype(A)) <: Real))
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; inv = false), twist!, A, 1; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; inv = true), twist!, A, [1, 3]; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, twist!, A, 1; atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, twist!, A, [1, 3]; atol, rtol, mode)
        end

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

        for i in 1:2
            B = insertleftunit(A, i; dual = rand(Bool))
            Mooncake.TestUtils.test_rule(rng, removeunit, B, Val(i); atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = false), removeunit, B, Val(i); atol, rtol, mode)
            Mooncake.TestUtils.test_rule(rng, Core.kwcall, (; copy = true), removeunit, B, Val(i); atol, rtol, mode)
        end
    end
end
