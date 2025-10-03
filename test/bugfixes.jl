@timedtestset "Bugfixes" verbose = true begin
    @testset "BugfixConvert" begin
        v = randn(
            ComplexF64,
            (
                Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)](
                    (-3, 1 / 2, 1) => 3, (-5, 1 / 2, 1) => 10, (-7, 1 / 2, 1) => 13,
                    (-9, 1 / 2, 1) => 9, (-11, 1 / 2, 1) => 1, (-5, 3 / 2, 1) => 3,
                    (-7, 3 / 2, 1) => 3, (-9, 3 / 2, 1) => 1
                ) ⊗ Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((1, 1 / 2, 1) => 1)'
            ) ← Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)](
                (-3, 1 / 2, 1) => 3, (-5, 1 / 2, 1) => 10, (-7, 1 / 2, 1) => 13,
                (-9, 1 / 2, 1) => 9, (-11, 1 / 2, 1) => 1, (-5, 3 / 2, 1) => 3,
                (-7, 3 / 2, 1) => 3, (-9, 3 / 2, 1) => 1
            )
        )
        w = convert(typeof(real(v)), v)
        @test w == v
        @test scalartype(w) == Float64
    end

    # https://github.com/quantumkithub/TensorKit.jl/issues/178
    @testset "Issue #178" begin
        t = rand(U1Space(1 => 1) ← U1Space(1 => 1)')
        a = convert(Array, t)
        @test a == zeros(size(a))
    end

    # https://github.com/quantumkithub/TensorKit.jl/issues/194
    @testset "Issue #194" begin
        t1 = rand(ℂ^4 ← ℂ^4)
        t2 = tensoralloc(typeof(t1), space(t1), Val(true), TensorOperations.ManualAllocator())
        t3 = similar(t2, ComplexF64, space(t1))
        @test storagetype(t3) == Vector{ComplexF64}
        t4 = similar(t2, domain(t1))
        @test storagetype(t4) == Vector{Float64}
        t5 = similar(t2)
        @test storagetype(t5) == Vector{Float64}
        tensorfree!(t2)
    end

    # https://github.com/quantumkithub/TensorKit.jl/issues/201
    @testset "Issue #201" begin
        function f(A::AbstractTensorMap)
            U, S, V, = svd_compact(A)
            return tr(S)
        end
        function f(A::AbstractMatrix)
            S = LinearAlgebra.svdvals(A)
            return sum(S)
        end
        A₀ = randn(Z2Space(4, 4) ← Z2Space(4, 4))
        grad1, = Zygote.gradient(f, A₀)
        grad2, = Zygote.gradient(f, convert(Array, A₀))
        @test convert(Array, grad1) ≈ grad2

        function g(A::AbstractTensorMap)
            U, S, V, = svd_compact(A)
            return tr(U * V)
        end
        function g(A::AbstractMatrix)
            U, S, V, = LinearAlgebra.svd(A)
            return tr(U * V')
        end
        B₀ = randn(ComplexSpace(4) ← ComplexSpace(4))
        grad3, = Zygote.gradient(g, B₀)
        grad4, = Zygote.gradient(g, convert(Array, B₀))
        @test convert(Array, grad3) ≈ grad4
    end

    # https://github.com/quantumkithub/TensorKit.jl/issues/209
    @testset "Issue #209" begin
        function f(T, D)
            @tensor T[1, 4, 1, 3] * D[3, 4]
        end
        V = Z2Space(2, 2)
        D = DiagonalTensorMap(randn(4), V)
        T = randn(V ⊗ V ← V ⊗ V)
        g1, = Zygote.gradient(f, T, D)
        g2, = Zygote.gradient(f, T, TensorMap(D))
        @test g1 ≈ g2
    end
end
