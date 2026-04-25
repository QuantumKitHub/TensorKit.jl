using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
Tαs = is_ci ? (Active,) : (Active, Const)
Tβs = is_ci ? (Active,) : (Active, Const)

@timedtestset "Enzyme - Index Manipulations (add_braid!):" begin
    @timedtestset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T) Tα $Tα Tβ $Tβ" for V in spacelist, T in eltypes, Tα in Tαs, Tβ in Tβs
        atol = default_tol(T)
        rtol = default_tol(T)
        Vstr = TensorKit.type_repr(sectortype(eltype(V)))
        A = randn(T, V[1] ⊗ V[2] ← V[4] ⊗ V[5])
        α = randn(T)
        β = randn(T)
        p = randcircshift(numout(A), numin(A))
        levels = Tuple(randperm(numind(A)))
        C = randn!(transpose(A, p))
        EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add_braid! V $Vstr Tα $Tα Tβ $Tβ")
        if !(T <: Real) && !is_ci
            EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "add_braid! V $Vstr Tα $Tα Tβ $Tβ")
            EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (β, Tβ); atol, rtol, testset_name = "add_braid! V $Vstr Tα $Tα Tβ $Tβ")
            EnzymeTestUtils.test_reverse(TensorKit.add_braid!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (real(β), Tβ); atol, rtol, testset_name = "add_braid! V $Vstr Tα $Tα Tβ $Tβ")
        end
    end
end
