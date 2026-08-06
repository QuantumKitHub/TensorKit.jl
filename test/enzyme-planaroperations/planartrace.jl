using Test, TestExtras
using TensorKit
using TensorOperations
using VectorInterface: Zero, One
using Enzyme, EnzymeTestUtils
using Random

spacelist = ad_spacelist(fast_tests)
#eltypes = (Float64, ComplexF64)
eltypes = (ComplexF64,)

@timedtestset "Enzyme - PlanarOperations (planartrace): $(TensorKit.type_repr(sectortype(eltype(V)))) ($T)" for V in spacelist, T in eltypes
    atol = default_tol(T)
    rtol = default_tol(T)
    for _ in 1:5
        k1 = rand(0:2)
        k2 = rand(0:1)
        V1 = map(v -> rand(Bool) ? v' : v, rand(V, k1))
        V2 = map(v -> rand(Bool) ? v' : v, rand(V, k2))
        V3 = prod(x -> x ⊗ x', V2[1:k2]; init = one(V[1]))
        V4 = prod(x -> x ⊗ x', V2[(k2 + 1):end]; init = one(V[1]))

        k′ = rand(0:(k1 + 2k2))
        (_p, _q) = randcircshift(k′, k1 + 2k2 - k′, k1)
        p = _repartition(_p, rand(0:k1))
        q = (tuple(_q[1:2:end]...), tuple(_q[2:2:end]...))
        if !all(isempty, p) && !all(isempty, q)
            α = randn(T)
            β = randn(T)
            ip = _repartition(invperm(linearize((_p, _q))), k′)
            A = randn(T, permute(prod(V1) ⊗ V3 ← V4, ip))
            C = randn!(TensorOperations.tensoralloc_add(T, A, p, false, Val(false)))
            for Tα in (Const, Active), Tβ in (Const, Active)
                EnzymeTestUtils.test_reverse(TensorKit.planartrace!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (q, Const), (α, Tα), (β, Tβ), (TensorOperations.DefaultBackend(), Const), (TensorOperations.DefaultAllocator(), Const); atol, rtol, testset_name = "planartrace reverse Tα $Tα Tβ $Tβ")
            end
            for Tα in (Const, Duplicated), Tβ in (Const, Duplicated)
                EnzymeTestUtils.test_forward(TensorKit.planartrace!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (q, Const), (α, Tα), (β, Tβ), (TensorOperations.DefaultBackend(), Const), (TensorOperations.DefaultAllocator(), Const); atol, rtol, testset_name = "planartrace forward Tα $Tα Tβ $Tβ")
            end
        end
    end
end
