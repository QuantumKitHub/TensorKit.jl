using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: One, Zero
using Enzyme, EnzymeTestUtils

is_ci = get(ENV, "CI", "false") == "true"

spacelist = ad_spacelist(fast_tests)
eltypes = (Float64, ComplexF64)

@timedtestset "Enzyme - TensorOperations" begin
    @timedtestset verbose = true "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
        atol = default_tol(T)
        rtol = default_tol(T)
        symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
        symmetricbraiding && @timedtestset "tensorcontract!" begin
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
            A = randn(T, permute(V1 ← V2, ipA))
            B = randn(T, permute(V2 ← V3, ipB))
            C = randn!(
                TensorOperations.tensoralloc_contract(
                    T, A, pA, false, B, pB, false, pAB, Val(false)
                )
            )

            αβs = is_ci ? (((α, Active), (β, Active)),) : Iterators.product(((One(), Const), (α, Const), (α, Active)), ((Zero(), Const), (β, Const), (β, Active)))
            for (α_, β_) in αβs
                EnzymeTestUtils.test_reverse(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (A, Duplicated), (pA, Const),
                    (B, Duplicated), (pB, Const), (pAB, Const),
                    α_, β_,
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! reverse α $α_ β $β_",
                )
            end
            αβs = is_ci ? (((α, Duplicated), (β, Duplicated)),) : Iterators.product(((One(), Const), (α, Const), (α, Duplicated)), ((Zero(), Const), (β, Const), (β, Duplicated)))
            for (α_, β_) in αβs
                EnzymeTestUtils.test_forward(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (A, Duplicated), (pA, Const),
                    (B, Duplicated), (pB, Const), (pAB, Const),
                    α_, β_,
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! forward α $α_ β $β_",
                )
            end
            if !(T <: Real) && !is_ci
                EnzymeTestUtils.test_reverse(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (A, Duplicated), (pA, Const),
                    (B, Duplicated), (pB, Const), (pAB, Const),
                    (real(α), Active), (real(β), Active),
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! reverse real(α) real(β)",
                )
                EnzymeTestUtils.test_reverse(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (real(A), Duplicated), (pA, Const),
                    (B, Duplicated), (pB, Const), (pAB, Const),
                    (real(α), Active), (real(β), Active),
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! reverse real(A) real(α) real(β)",
                )
                EnzymeTestUtils.test_reverse(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (A, Duplicated), (pA, Const),
                    (real(B), Duplicated), (pB, Const), (pAB, Const),
                    (real(α), Active), (real(β), Active),
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! reverse real(B) real(α) real(β)",
                )
                EnzymeTestUtils.test_forward(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (A, Duplicated), (pA, Const),
                    (B, Duplicated), (pB, Const), (pAB, Const),
                    (real(α), Active), (real(β), Active),
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! forward real(α) real(β)",
                )
                EnzymeTestUtils.test_forward(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (real(A), Duplicated), (pA, Const),
                    (B, Duplicated), (pB, Const), (pAB, Const),
                    (real(α), Active), (real(β), Active),
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! forward real(A) real(α) real(β)",
                )
                EnzymeTestUtils.test_forward(
                    TensorKit.blas_contract!, Duplicated,
                    (copy(C), Duplicated), (A, Duplicated), (pA, Const),
                    (real(B), Duplicated), (pB, Const), (pAB, Const),
                    (real(α), Active), (real(β), Active),
                    (TensorOperations.DefaultBackend(), Const),
                    (TensorOperations.DefaultAllocator(), Const);
                    atol, rtol,
                    testset_name = "blas_contract! forward real(B) real(α) real(β)",
                )
            end
        end
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
