using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

mode = Enzyme.ReverseMode
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

@timedtestset "Enzyme - Index Manipulations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding

    symmetricbraiding && @timedtestset "add_permute!" begin
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        α = randn(T)
        β = randn(T)

        # repeat a couple times to get some distribution of arrows
        for _ in 1:5
            for Tα in (Active, Const), Tβ in (Active, Const)
                p = randindextuple(numind(A))
                C = randn!(permute(A, p))
                EnzymeTestUtils.test_reverse(TensorKit.add_permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                A = C
            end
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
            EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
            for Tα in (Const, Active), Tβ in (Const, Active)
                EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                if !(T <: Real)
                    EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.add_transpose!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                end
                A = C
            end
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
            for Tα in (Active, Const), Tβ in (Active, Const)
                EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (A, DuplicateD), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol)
                if !(T <: Real)
                    EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (real(A), DuplicateD), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (A, DuplicateD), (p, Const), (levels, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (A, DuplicateD), (p, Const), (levels, Const), (real(α), Tα), (real(β), Tβ); atol, rtol)
                end
                A = C
            end
        end
    end

    @timedtestset "flip_n_twist!" begin
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        for TA in (Const, Duplicated)
            if !(T <: Real && !(sectorscalartype(sectortype(A)) <: Real))
                EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
                EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
                EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), (1, Const); atol, rtol)
                EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol)
            end

            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), [1, 3]; atol, rtol, fkwargs = (inv = true,))
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol)
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), ([1, 3], Const); atol, rtol)
        end
    end

    @timedtestset "insert and remove units" begin
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])

        for TA in (Const, Duplicated)
            for insertunit in (insertleftunit, insertrightunit)
                EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(1), Const); atol, rtol)
                EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(4), Const); atol, rtol)
                EnzymeTestUtils.test_reverse(insertunit, TA, (A', TA), (Val(2), Const); atol, rtol)
                EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(1), Const); atol, rtol, fkwargs = (copy = false,))
                EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(2), Const); atol, rtol, fkwargs = (copy = true,))
                EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(3), Const); atol, rtol, fkwargs = (copy = false, dual = true, conj = true))
                EnzymeTestUtils.test_reverse(insertunit, TA, (A', TA), (Val(3), Const); atol, rtol, fkwargs = (copy = false, dual = true, conj = true))
            end
        end

        for TB in (Const, Duplicated)
            for i in 1:2
                B = insertleftunit(A, i; dual = rand(Bool))
                EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol)
                EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol, fkwargs = (copy = false,))
                EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol, fkwargs = (copy = true,))
            end
        end
    end
end
