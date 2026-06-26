using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: One, Zero
using Enzyme, EnzymeTestUtils

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

is_ci = get(ENV, "CI", "false") == "true"
rTαs = is_ci ? (Active,) : (Const, Active)
rTβs = is_ci ? (Active,) : (Const, Active)
fTαs = is_ci ? (Duplicated,) : (Const, Duplicated)
fTβs = is_ci ? (Duplicated,) : (Const, Duplicated)
TCs = is_ci ? (Duplicated,) : (Const, Duplicated)
TAs = is_ci ? (Duplicated,) : (Const, Duplicated)

@timedtestset "Enzyme - TensorOperations (trace)" begin
    @timedtestset verbose = true "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        println(TensorKit.type_repr(sectortype(eltype(V)))) # just some printing for test purposes
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
            for TC in TCs, TA in TAs
                for Tα in rTαs, Tβ in rTβs
                    EnzymeTestUtils.test_reverse(
                        TensorKit.trace_permute!, TC,
                        (copy(C), TC), (A, TA), (p, Const), (q, Const),
                        (α, Tα), (β, Tβ), (TensorOperations.DefaultBackend(), Const);
                        atol, rtol,
                        testset_name = "trace_permute! reverse TC $TC TA $TA Tα $Tα Tβ $Tβ",
                    )
                end
                for Tα in fTαs, Tβ in fTβs
                    EnzymeTestUtils.test_forward(
                        TensorKit.trace_permute!, TC,
                        (copy(C), TC), (A, TA), (p, Const), (q, Const),
                        (α, Tα), (β, Tβ), (TensorOperations.DefaultBackend(), Const);
                        atol, rtol,
                        testset_name = "trace_permute! forward TC $TC TA $TA Tα $Tα Tβ $Tβ",
                    )
                end
            end
        end
    end
end
