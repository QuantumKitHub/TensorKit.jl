using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: One, Zero
using Enzyme, EnzymeTestUtils
Enzyme.Compiler.VERBOSE_ERRORS[] = true

@isdefined(TestSetup) || include("../../setup.jl")
using .TestSetup

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

@timedtestset "Enzyme - TensorOperations (trace)" begin
    @timedtestset verbose = true "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
        symmetricbraiding && @timedtestset "trace_permute!" begin
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
            for TC in (Const, Duplicated), TA in (Const, Duplicated), Tα in (Const, Active), Tβ in (Const, Active)
                EnzymeTestUtils.test_reverse(
                    TensorKit.trace_permute!, TC,
                    (copy(C), TC), (A, TA), (p, Const), (q, Const),
                    (α, Tα), (β, Tβ), (TensorOperations.DefaultBackend(), Const);
                    atol, rtol,
                    testset_name = "trace_permute! TC $TC TA $TA Tα $Tα Tβ $Tβ",
                )
            end
        end
    end
end
