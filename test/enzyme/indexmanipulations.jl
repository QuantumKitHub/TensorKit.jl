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

@timedtestset "Enzyme - Index Manipulations: $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    has_braiding = BraidingStyle(sectortype(eltype(V))) isa HasBraiding
    symmetric_braiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
    Vstr = TensorKit.type_repr(sectortype(eltype(V)))
    atol = default_tol(T)
    rtol = default_tol(T)
    if VERSION >= v"1.11.0-rc" # segfault issues on 1.10
        symmetric_braiding && @timedtestset "permute! Tα $Tα, Tβ $Tβ" for Tα in Tαs, Tβ in Tβs
            A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
            α = randn(T)
            β = randn(T)
            p = randindextuple(numind(A))
            C = randn!(permute(A, p))
            EnzymeTestUtils.test_reverse(TensorKit.permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
        end
    end
    if VERSION > v"1.11.0-rc" # https://github.com/QuantumKitHub/TensorKit.jl/issues/457
        @timedtestset "transpose!" begin
            A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
            α = randn(T)
            β = randn(T)

            # repeat a couple times to get some distribution of arrows
            p = randcircshift(numout(A), numin(A))
            C = randn!(transpose(A, p))
            !is_ci && EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
            @testset for Tα in Tαs, Tβ in Tβs
                EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                if !(T <: Real) && !is_ci
                    EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                end
            end
        end
    end
    if !Sys.iswindows() && VERSION > v"1.11.0-rc"
        @timedtestset "braid! Tα $Tα Tβ $Tβ" for Tα in Tαs, Tβ in Tβs
            if has_braiding
                A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
                α = randn(T)
                β = randn(T)
                p = randcircshift(numout(A), numin(A))
                levels = Tuple(randperm(numind(A)))
                C = randn!(transpose(A, p))
                EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "braid! V $Vstr Tα $Tα Tβ $Tβ")
                if !(T <: Real) && !is_ci
                    EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (C, Duplicated), (real(A), Duplicated), (p, Const), (levels, Const), (α, Tα), (β, Tβ); atol, rtol, testset_name = "braid! V $Vstr Tα $Tα Tβ $Tβ")
                    EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (β, Tβ); atol, rtol, testset_name = "braid! V $Vstr Tα $Tα Tβ $Tβ")
                    EnzymeTestUtils.test_reverse(TensorKit.braid!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (levels, Const), (real(α), Tα), (real(β), Tβ); atol, rtol, testset_name = "braid! V $Vstr Tα $Tα Tβ $Tβ")
                end
            end
        end
    end
    if !Sys.iswindows() && VERSION > v"1.11.0-rc"
        has_braiding && @timedtestset "flip_n_twist! TA ($TA)" for TA in (Duplicated,)
            A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
            if !(T <: Real && !(sectorscalartype(sectortype(A)) <: Real))
                EnzymeTestUtils.test_reverse(twist!, TA, (copy(A), TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
                EnzymeTestUtils.test_reverse(twist!, TA, (copy(A), TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
                EnzymeTestUtils.test_reverse(twist!, TA, (copy(A), TA), (1, Const); atol, rtol)
                EnzymeTestUtils.test_reverse(twist!, TA, (copy(A), TA), ([1, 3], Const); atol, rtol)
            end
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), [1, 3]; atol, rtol, fkwargs = (inv = true,))
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol)
            EnzymeTestUtils.test_reverse(flip, TA, (A, TA), ([1, 3], Const); atol, rtol)
        end
    end

    @timedtestset "insert and remove units TA ($TA)" for TA in (Duplicated,)
        A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        @testset for insertunit in (insertleftunit, insertrightunit)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(1), Const); atol, rtol)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(4), Const); atol, rtol)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A', TA), (Val(2), Const); atol, rtol)
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(1), Const); atol, rtol, fkwargs = (copy = false,))
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(2), Const); atol, rtol, fkwargs = (copy = true,))
            EnzymeTestUtils.test_reverse(insertunit, TA, (A, TA), (Val(3), Const); atol, rtol, fkwargs = (copy = false, dual = true, conj = true))
            EnzymeTestUtils.test_reverse(insertunit, TA, (A', TA), (Val(3), Const); atol, rtol, fkwargs = (copy = false, dual = true, conj = true))
        end
        for i in 1:2
            B = insertleftunit(A, i; dual = rand(Bool))
            EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol, fkwargs = (copy = false,))
            EnzymeTestUtils.test_reverse(removeunit, TB, (B, TB), (Val(i), Const); atol, rtol, fkwargs = (copy = true,))
        end
    end
end
