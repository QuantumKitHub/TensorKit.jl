using Test, TestExtras
using TensorKit
using TensorOperations
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
rTαs = is_ci ? (Active,) : (Active, Const)
rTβs = is_ci ? (Active,) : (Active, Const)
fTαs = is_ci ? (Duplicated,) : (Duplicated, Const)
fTβs = is_ci ? (Duplicated,) : (Duplicated, Const)

@timedtestset "Enzyme - Index Manipulations (braid!) $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    Vstr = TensorKit.type_repr(sectortype(eltype(V)))
    has_braiding = BraidingStyle(sectortype(eltype(V))) isa HasBraiding
    if has_braiding
        A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        α = randn(T)
        β = randn(T)
        p = randcircshift(numout(A), numin(A))
        levels = Tuple(randperm(numind(A)))
        C = randn!(transpose(A, p))
        for Tα in rTαs, Tβ in rTβs
            EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "braid! reverse V $Vstr Tα $Tα Tβ $Tβ")
            if !(T <: Real) && !is_ci
                EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "braid! reverse V $Vstr Tα $Tα Tβ $Tβ")
                EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (β, Tβ); atol, rtol, testset_name = "braid! reverse V $Vstr Tα $Tα Tβ $Tβ")
                EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (real(β), Tβ); atol, rtol, testset_name = "braid! reverse V $Vstr Tα $Tα Tβ $Tβ")
            end
        end
        for Tα in fTαs, Tβ in fTβs
            C = randn!(transpose(A, p))
            EnzymeTestUtils.test_forward(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "braid! forward V $Vstr Tα $Tα Tβ $Tβ")
            if !(T <: Real) && !is_ci
                EnzymeTestUtils.test_forward(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "braid! forward V $Vstr Tα $Tα Tβ $Tβ")
                EnzymeTestUtils.test_forward(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (β, Tβ); atol, rtol, testset_name = "braid! forward V $Vstr Tα $Tα Tβ $Tβ")
                EnzymeTestUtils.test_forward(TensorKit.braid!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (real(β), Tβ); atol, rtol, testset_name = "braid! forward V $Vstr Tα $Tα Tβ $Tβ")
            end
        end
    end
end
