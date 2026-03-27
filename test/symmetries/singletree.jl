using Test, TestExtras
using TensorKit
using TensorKit: FusionTreeBlock, ×
import TensorKit as TK
using Random: randperm
using TensorOperations
using MatrixAlgebraKit: isunitary
using LinearAlgebra
using TupleTools

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

@timedtestset "Fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in (fast_tests ? fast_sectorlist : sectorlist)
    Istr = TensorKit.type_repr(I)
    N = 5
    out = random_fusion(I, Val(N))
    isdual = ntuple(n -> rand(Bool), N)
    in = rand(collect(⊗(out...)))
    numtrees = length(fusiontrees(out, in, isdual))
    @test numtrees == count(n -> true, fusiontrees(out, in, isdual))
    while !(0 < numtrees < 30) && !(one(in) in ⊗(out...))
        out = ntuple(n -> randsector(I), N)
        in = rand(collect(⊗(out...)))
        numtrees = length(fusiontrees(out, in, isdual))
        @test numtrees == count(n -> true, fusiontrees(out, in, isdual))
    end
    it = @constinferred fusiontrees(out, in, isdual)
    @constinferred Nothing iterate(it)
    f, s = iterate(it)
    @constinferred Nothing iterate(it, s)
    @test f == @constinferred first(it)

    @testset "Fusion tree: printing" begin
        @test eval(Meta.parse(sprint(show, f; context = (:module => @__MODULE__)))) == f
    end

    @testset "Fusion tree: constructor properties" begin
        for u in allunits(I)
            @constinferred FusionTree((), u, (), (), ())
            @constinferred FusionTree((u,), u, (false,), (), ())
            @constinferred FusionTree((u, u), u, (false, false), (), (1,))
            @constinferred FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
            @constinferred FusionTree(
                (u, u, u, u), u, (false, false, false, false), (u, u), (1, 1, 1)
            )
            @test_throws MethodError FusionTree((u, u, u), u, (false, false), (u,), (1, 1))
            @test_throws MethodError FusionTree(
                (u, u, u), u, (false, false, false), (u, u), (1, 1)
            )
            @test_throws MethodError FusionTree(
                (u, u, u), u, (false, false, false), (u,), (1, 1, 1)
            )
            @test_throws MethodError FusionTree((u, u, u), u, (false, false, false), (), (1,))

            f = FusionTree((u, u, u), u, (false, false, false), (u,), (1, 1))
            @test sectortype(f) == I
            @test length(f) == 3
            @test FusionStyle(f) == FusionStyle(I)
            @test BraidingStyle(f) == BraidingStyle(I)

            if FusionStyle(I) isa UniqueFusion
                @constinferred FusionTree((), u, ())
                @constinferred FusionTree((u,), u, (false,))
                @constinferred FusionTree((u, u), u, (false, false))
                @constinferred FusionTree((u, u, u), u)
                if UnitStyle(I) isa SimpleUnit
                    @constinferred FusionTree((u, u, u, u))
                else
                    @test_throws ArgumentError FusionTree((u, u, u, u))
                end
                @test_throws MethodError FusionTree((u, u), u, (false, false, false))
            else
                @test_throws ArgumentError FusionTree((), u, ())
                @test_throws ArgumentError FusionTree((u,), u, (false,))
                @test_throws ArgumentError FusionTree((u, u), u, (false, false))
                @test_throws ArgumentError FusionTree((u, u, u), u)
                if I <: ProductSector && UnitStyle(I) isa GenericUnit
                    @test_throws DomainError FusionTree((u, u, u, u))
                else
                    @test_throws ArgumentError FusionTree((u, u, u, u))
                end
            end
        end
    end

    # Basic associativity manipulations of individual fusion trees
    @testset "Fusion tree: split and join" begin
        N = 6
        uncoupled = random_fusion(I, Val(N))
        coupled = rand(collect(⊗(uncoupled...)))
        isdual = ntuple(n -> rand(Bool), N)
        f = rand(collect(fusiontrees(uncoupled, coupled, isdual)))
        for i in 0:N
            f₁, f₂ = @constinferred TK.split(f, $i)
            @test length(f₁) == i
            @test length(f₂) == N - i + 1
            f′ = @constinferred TK.join(f₁, f₂)
            @test f′ == f
        end
    end

    @testset "Fusion tree: multi_Fmove" begin
        N = 6
        uncoupled = random_fusion(I, Val(N))
        coupled = rand(collect(⊗(uncoupled...)))
        isdualrest = ntuple(n -> rand(Bool), N - 1)
        for isdual in ((false, isdualrest...), (true, isdualrest...))
            trees = collect(fusiontrees(uncoupled, coupled, isdual))
            # trees = rand(trees, min(5, length(trees))) # limit number of tests?
            for f in trees
                a = f.uncoupled[1]
                isduala = f.isdual[1]
                c = f.coupled
                f′s, coeffs = @constinferred TK.multi_Fmove(f)
                @test norm(coeffs) ≈ 1 atol = 1.0e-12 # expansion should have unit norm
                d = Dict(f => -one(eltype(eltype(coeffs))))
                for (f′, coeff) in zip(f′s, coeffs)
                    @test coeff ≈ TK.multi_associator(f, f′)
                    f′′s, coeff′s = @constinferred TK.multi_Fmove_inv(a, c, f′, isduala)
                    if FusionStyle(I) isa MultiplicityFreeFusion
                        @test norm(coeff′s) ≈ 1 atol = 1.0e-12 # expansion should have unit norm
                    else
                        for i in 1:Nsymbol(a, f′.coupled, c)
                            @test norm(getindex.(coeff′s, i)) ≈ 1 atol = 1.0e-12 # expansion should have unit norm for every possible fusion channel at the top vertex
                        end
                    end
                    for (f′′, coeff′) in zip(f′′s, coeff′s)
                        @test coeff′ ≈ conj(TK.multi_associator(f′′, f′))
                        d[f′′] = get(d, f′′, zero(eltype(coeff′))) + sum(coeff .* coeff′)
                    end
                end
                @test norm(values(d)) < 1.0e-12
            end
        end

        if hasfusiontensor(I) # because no permutations are involved, this also works for fermionic braiding
            N = 4
            uncoupled = random_fusion(I, Val(N))
            coupled = rand(collect(⊗(uncoupled...)))
            isdualrest = ntuple(n -> rand(Bool), N - 1)
            for isdual in ((false, isdualrest...), (true, isdualrest...))
                trees = collect(fusiontrees(uncoupled, coupled, isdual))
                for f in trees
                    ftensor = fusiontensor(f)
                    ftensor′ = zero(ftensor)
                    a = f.uncoupled[1]
                    isduala = f.isdual[1]
                    c = f.coupled
                    f′s, coeffs = @constinferred TK.multi_Fmove(f)
                    for (f′, coeff) in zip(f′s, coeffs)
                        f′tensor = fusiontensor(f′)
                        for i in 1:Nsymbol(a, f′.coupled, c)
                            f′′ = FusionTree{I}((a, f′.coupled), c, (isduala, false), (), (i,))
                            f′′tensor = fusiontensor(f′′)
                            ftensor′ += coeff[i] * tensorcontract(1:(N + 1), f′tensor, [(2:N)..., -1], f′′tensor, [1, -1, N + 1])
                        end
                    end
                    @test ftensor′ ≈ ftensor atol = 1.0e-12
                end
            end
        end
    end

    @testset "Fusion tree: insertat" begin
        # just check some basic consistency properties here
        # correctness should follow from multi_Fmove tests
        N = 4
        out2 = random_fusion(I, Val(N))
        in2 = rand(collect(⊗(out2...)))
        isdual2 = ntuple(n -> rand(Bool), N)
        f2 = rand(collect(fusiontrees(out2, in2, isdual2)))
        for i in 1:N
            out1 = random_fusion(I, Val(N)) # guaranteed good fusion
            out1 = Base.setindex(out1, in2, i) # can lead to poor fusion
            while isempty(⊗(out1...)) # TODO: better way to do this?
                out1 = random_fusion(I, Val(N))
                out1 = Base.setindex(out1, in2, i)
            end
            in1 = rand(collect(⊗(out1...)))
            isdual1 = ntuple(n -> rand(Bool), N)
            isdual1 = Base.setindex(isdual1, false, i)
            f1 = rand(collect(fusiontrees(out1, in1, isdual1)))

            trees = @constinferred TK.insertat(f1, i, f2)
            @test norm(values(trees)) ≈ 1

            if hasfusiontensor(I)
                Af1 = fusiontensor(f1)
                Af2 = fusiontensor(f2)
                Af = tensorcontract(
                    1:(2N), Af1,
                    [1:(i - 1); -1; N - 1 .+ ((i + 1):(N + 1))],
                    Af2, [i - 1 .+ (1:N); -1]
                )
                Af′ = zero(Af)
                for (f, coeff) in trees
                    Af′ .+= coeff .* fusiontensor(f)
                end
                @test Af ≈ Af′
            end
        end
    end

    @testset "Fusion tree: merging" begin
        N = 3
        out1 = random_fusion(I, Val(N))
        out2 = random_fusion(I, Val(N))
        in1 = rand(collect(⊗(out1...)))
        in2 = rand(collect(⊗(out2...)))
        tp = ⊗(in1, in2) # messy solution but it works
        while isempty(tp)
            out1 = random_fusion(I, Val(N))
            out2 = random_fusion(I, Val(N))
            in1 = rand(collect(⊗(out1...)))
            in2 = rand(collect(⊗(out2...)))
            tp = ⊗(in1, in2)
        end

        f1 = rand(collect(fusiontrees(out1, in1)))
        f2 = rand(collect(fusiontrees(out2, in2)))

        d = @constinferred TK.merge(f1, f2, first(in1 ⊗ in2), 1)
        @test norm(values(d)) ≈ 1
        if !(FusionStyle(I) isa GenericFusion)
            @constinferred TK.merge(f1, f2, first(in1 ⊗ in2))
        end
        @test dim(in1) * dim(in2) ≈ sum(
            abs2(coeff) * dim(c) for c in in1 ⊗ in2
                for μ in 1:Nsymbol(in1, in2, c)
                for (f, coeff) in TK.merge(f1, f2, c, μ)
        )

        if hasfusiontensor(I)
            for c in in1 ⊗ in2
                for μ in 1:Nsymbol(in1, in2, c)
                    Af1 = fusiontensor(f1)
                    Af2 = fusiontensor(f2)
                    Af0 = fusiontensor(FusionTree((in1, in2), c, (false, false), (), (μ,)))
                    _Af = tensorcontract(
                        1:(N + 2), Af1, [1:N; -1], Af0, [-1; N + 1; N + 2]
                    )
                    Af = tensorcontract(
                        1:(2N + 1), Af2, [N .+ (1:N); -1], _Af, [1:N; -1; 2N + 1]
                    )
                    Af′ = zero(Af)
                    for (f, coeff) in TK.merge(f1, f2, c, μ)
                        Af′ .+= coeff .* fusiontensor(f)
                    end
                    @test Af ≈ Af′
                end
            end
        end
    end

    # Duality tests
    @testset "Fusion tree: elementary planar trace" begin
        N = 5
        uncoupled = random_fusion(I, Val(N))
        coupled = rand(collect(⊗(uncoupled...)))
        isdual = ntuple(n -> rand(Bool), N)
        f = rand(collect(fusiontrees(uncoupled, coupled, isdual)))
        for i in 0:N # insert a (b b̄ ← 1) vertex in the tree after ith uncoupled sector and then trace it away
            f₁, f₂ = TK.split(f, i)
            c = f₁.coupled
            funit = FusionTree{I}((c, rightunit(c)), c, (false, false), (), (1,))
            f′ = TK.join(TK.join(f₁, funit), f₂)
            for b in smallset(I)
                leftunit(b) == rightunit(c) || continue
                out = Dict(f => -sqrtdim(b) * one(fusionscalartype(I)))
                fbb = FusionTree{I}((b, dual(b)), leftunit(b), (false, true), (), (1,))
                for (f′′, coeff) in TK.insertat(f′, i + 1, fbb)
                    d = @constinferred TK.elementary_trace(f′′, i + 1)
                    for (tree, coeff2) in d
                        out[tree] = get(out, tree, zero(eltype(coeff2))) + coeff * coeff2
                    end
                end
                @test norm(values(out)) < 1.0e-12
                out = Dict(f => -frobenius_schur_phase(b) * sqrtdim(b) * one(fusionscalartype(I)))
                fbb = FusionTree{I}((b, dual(b)), leftunit(b), (true, false), (), (1,))
                for (f′′, coeff) in TK.insertat(f′, i + 1, fbb)
                    for (tree, coeff2) in TK.elementary_trace(f′′, i + 1)
                        out[tree] = get(out, tree, zero(eltype(coeff2))) + coeff * coeff2
                    end
                end
                @test norm(values(out)) < 1.0e-12
            end
        end
        # insert f′ in between the two legs of a (b b̄ ← 1) vertex and then trace the outer legs away
        f′ = TK.join(f, FusionTree{I}((coupled, dual(coupled)), leftunit(coupled), (false, true), (), (1,)))
        for b in smallset(I)
            rightunit(b) == leftunit(coupled) || continue
            fbb = FusionTree{I}((b, rightunit(b), dual(b)), leftunit(b), (false, false, true), (b,), (1, 1))
            out = Dict(f′ => -sqrtdim(b) * one(fusionscalartype(I)))
            for (f′′, coeff) in TK.insertat(fbb, 2, f′)
                d = @constinferred TK.elementary_trace(f′′, N + 3)
                for (tree, coeff2) in d
                    out[tree] = get(out, tree, zero(eltype(coeff2))) + coeff * coeff2
                end
            end
            @test norm(values(out)) < 1.0e-12
            fbb = FusionTree{I}((b, rightunit(b), dual(b)), leftunit(b), (true, false, false), (b,), (1, 1))
            out = Dict(f′ => -frobenius_schur_phase(b) * sqrtdim(b) * one(fusionscalartype(I)))
            for (f′′, coeff) in TK.insertat(fbb, 2, f′)
                for (tree, coeff2) in TK.elementary_trace(f′′, N + 3)
                    out[tree] = get(out, tree, zero(eltype(coeff2))) + coeff * coeff2
                end
            end
            @test norm(values(out)) < 1.0e-12
        end
    end

    TK.empty_globalcaches!()
end
