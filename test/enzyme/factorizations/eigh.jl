using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using MatrixAlgebraKit
using Enzyme, EnzymeTestUtils
using Random

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

function remove_eighgauge_dependence!(
        ΔV, D, V; degeneracy_atol = MatrixAlgebraKit.default_pullback_degeneracy_atol(D)
    )
    gaugepart = project_antihermitian!(V' * ΔV)
    for (c, b) in blocks(gaugepart)
        Dc = diagview(block(D, c))
        # for some reason this fails only on tests, and I cannot reproduce it in an
        # interactive session.
        # b[abs.(transpose(diagview(Dc)) .- diagview(Dc)) .>= degeneracy_atol] .= 0
        for j in axes(b, 2), i in axes(b, 1)
            abs(Dc[i] - Dc[j]) >= degeneracy_atol && (b[i, j] = 0)
        end
    end
    mul!(ΔV, V, gaugepart, -1, 1)
    return ΔV
end

@timedtestset "Enzyme - Factorizations (EIGH): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes, t in (randn(T, V[1] ← V[1]), rand(T, V[1] ⊗ V[2] ← V[1] ⊗ V[2]))
    atol = default_tol(T)
    rtol = default_tol(T)
    th = project_hermitian(t)
    DV = eigh_full(th)
    ΔDV = EnzymeTestUtils.rand_tangent(DV)
    remove_eighgauge_dependence!(ΔDV[2], DV...)
    EnzymeTestUtils.test_reverse(eigh_full ∘ project_hermitian, Duplicated, (t, Duplicated); output_tangent = ΔDV, atol, rtol)

    #D = eigh_vals(th)
    #EnzymeTestUtils.test_reverse(eigh_vals ∘ project_hermitian, Duplicated, (t, Duplicated); atol, rtol)

    V_trunc = spacetype(th)(c => min(size(b)...) ÷ 2 for (c, b) in blocks(t))
    trunc = truncspace(V_trunc)
    alg = MatrixAlgebraKit.select_algorithm(eigh_trunc_no_error, th, nothing; trunc)
    DVtrunc = eigh_trunc_no_error(th, alg)
    ΔDVtrunc = EnzymeTestUtils.rand_tangent(DVtrunc)
    remove_eighgauge_dependence!(ΔDVtrunc[2], DVtrunc...)
    proj_eigh(t, alg) = eigh_trunc_no_error(project_hermitian(t), alg)
    EnzymeTestUtils.test_reverse(proj_eigh, Duplicated, (t, Duplicated), (alg, Const); output_tangent = ΔDVtrunc, atol, rtol)
end
