using Test, TestExtras
using TensorKit
using TensorKit: FusionTreeBlock
import TensorKit as TK
using Random: randperm
using TensorOperations
using MatrixAlgebraKit: isunitary
using LinearAlgebra

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

_isunitary(x::Number; kwargs...) = isapprox(x * x', one(x); kwargs...)
_isunitary(x; kwargs...) = isunitary(x; kwargs...)
_isone(x; kwargs...) = isapprox(x, one(x); kwargs...)

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

@timedtestset "Fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in sectorlist
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
    @testset "Fusion tree $Istr: printing" begin
        @test eval(Meta.parse(sprint(show, f; context = (:module => @__MODULE__)))) == f
    end
    @testset "Fusion tree $Istr: constructor properties" begin
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
    function _reinsert_partial_tree(t, f)
        (t′, c′) = first(TK.insertat(t, 1, f))
        @test c′ == one(c′)
        return t′
    end
    @testset "Fusion tree $Istr: insertat" begin
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

            f1a, f1b = @constinferred TK.split(f1, $i)
            @test length(TK.insertat(f1b, 1, f1a)) == 1
            @test first(TK.insertat(f1b, 1, f1a)) == (f1 => 1)

            levels = ntuple(identity, N)

            # TODO: restore this?
            # if UnitStyle(I) isa SimpleUnit
            #     braid_i_to_1 = braid(f1, (i, (1:(i - 1))..., ((i + 1):N)...), levels)
            #     trees2 = Dict(_reinsert_partial_tree(t, f2) => c for (t, c) in braid_i_to_1)
            #     trees3 = empty(trees2)
            #     p = (((N + 1):(N + i - 1))..., (1:N)..., ((N + i):(2N - 1))...)
            #     levels = ((i:(N + i - 1))..., (1:(i - 1))..., ((i + N):(2N - 1))...)
            #     for (t, coeff) in trees2
            #         for (t′, coeff′) in braid(t, p, levels)
            #             trees3[t′] = get(trees3, t′, zero(coeff′)) + coeff * coeff′
            #         end
            #     end
            #     for (t, coeff) in trees3
            #         coeff′ = get(trees, t, zero(coeff))
            #         @test isapprox(coeff′, coeff; atol = 1.0e-12, rtol = 1.0e-12)
            #     end
            # end

            if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
                Af1 = convert(Array, f1)
                Af2 = convert(Array, f2)
                Af = tensorcontract(
                    1:(2N), Af1,
                    [1:(i - 1); -1; N - 1 .+ ((i + 1):(N + 1))],
                    Af2, [i - 1 .+ (1:N); -1]
                )
                Af′ = zero(Af)
                for (f, coeff) in trees
                    Af′ .+= coeff .* convert(Array, f)
                end
                @test isapprox(Af, Af′; atol = 1.0e-12, rtol = 1.0e-12)
            end
        end
    end

    @testset "Fusion tree $Istr: planar trace" begin
        if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
            s = randsector(I)
            N = 6
            outgoing = (s, dual(s), s, dual(s), s, dual(s))
            for bool in (true, false)
                isdual = (bool, !bool, bool, !bool, bool, !bool)
                for f in fusiontrees(outgoing, unit(s), isdual)
                    af = convert(Array, f)
                    T = eltype(af)

                    for i in 1:N
                        d = @constinferred TK.elementary_trace(f, i)
                        j = mod1(i + 1, N)
                        inds = collect(1:(N + 1))
                        inds[i] = inds[j]
                        bf = tensortrace(af, inds)
                        bf′ = zero(bf)
                        for (f′, coeff) in d
                            bf′ .+= coeff .* convert(Array, f′)
                        end
                        @test bf ≈ bf′ atol = 1.0e-12
                    end

                    d2 = @constinferred TK.planar_trace(f, ((1, 3), (2, 4)))
                    oind2 = (5, 6, 7)
                    bf2 = tensortrace(af, (:a, :a, :b, :b, :c, :d, :e))
                    bf2′ = zero(bf2)
                    for (f2′, coeff) in d2
                        bf2′ .+= coeff .* convert(Array, f2′)
                    end
                    @test bf2 ≈ bf2′ atol = 1.0e-12

                    d2 = @constinferred TK.planar_trace(f, ((5, 6), (2, 1)))
                    oind2 = (3, 4, 7)
                    bf2 = tensortrace(af, (:a, :b, :c, :d, :b, :a, :e))
                    bf2′ = zero(bf2)
                    for (f2′, coeff) in d2
                        bf2′ .+= coeff .* convert(Array, f2′)
                    end
                    @test bf2 ≈ bf2′ atol = 1.0e-12

                    d2 = @constinferred TK.planar_trace(f, ((1, 4), (6, 3)))
                    bf2 = tensortrace(af, (:a, :b, :c, :c, :d, :a, :e))
                    bf2′ = zero(bf2)
                    for (f2′, coeff) in d2
                        bf2′ .+= coeff .* convert(Array, f2′)
                    end
                    @test bf2 ≈ bf2′ atol = 1.0e-12

                    q1 = (1, 3, 5)
                    q2 = (2, 4, 6)
                    d3 = @constinferred TK.planar_trace(f, (q1, q2))
                    bf3 = tensortrace(af, (:a, :a, :b, :b, :c, :c, :d))
                    bf3′ = zero(bf3)
                    for (f3′, coeff) in d3
                        bf3′ .+= coeff .* convert(Array, f3′)
                    end
                    @test bf3 ≈ bf3′ atol = 1.0e-12

                    q1 = (1, 3, 5)
                    q2 = (6, 2, 4)
                    d3 = @constinferred TK.planar_trace(f, (q1, q2))
                    bf3 = tensortrace(af, (:a, :b, :b, :c, :c, :a, :d))
                    bf3′ = zero(bf3)
                    for (f3′, coeff) in d3
                        bf3′ .+= coeff .* convert(Array, f3′)
                    end
                    @test bf3 ≈ bf3′ atol = 1.0e-12

                    q1 = (1, 2, 3)
                    q2 = (6, 5, 4)
                    d3 = @constinferred TK.planar_trace(f, (q1, q2))
                    bf3 = tensortrace(af, (:a, :b, :c, :c, :b, :a, :d))
                    bf3′ = zero(bf3)
                    for (f3′, coeff) in d3
                        bf3′ .+= coeff .* convert(Array, f3′)
                    end
                    @test bf3 ≈ bf3′ atol = 1.0e-12

                    q1 = (1, 2, 4)
                    q2 = (6, 3, 5)
                    d3 = @constinferred TK.planar_trace(f, (q1, q2))
                    bf3 = tensortrace(af, (:a, :b, :b, :c, :c, :a, :d))
                    bf3′ = zero(bf3)
                    for (f3′, coeff) in d3
                        bf3′ .+= coeff .* convert(Array, f3′)
                    end
                    @test bf3 ≈ bf3′ atol = 1.0e-12
                end
            end
        end
    end

    (BraidingStyle(I) isa HasBraiding) && @testset "Fusion tree $Istr: elementary artin braid" begin
        N = length(out)
        isdual = ntuple(n -> rand(Bool), N)
        if FusionStyle(I) isa UniqueFusion
            for in in ⊗(out...)
                src = only(fusiontrees(out, in, isdual))
                for i in 1:(N - 1)
                    dst, U = @constinferred TK.artin_braid(src, i)
                    @test _isunitary(U)
                    dst′, U′ = @constinferred TK.artin_braid(dst, i; inv = true)
                    @test U' ≈ U′
                end
            end
        else
            src = FusionTreeBlock{I}((out, ()), (isdual, ()))
            length(src) > 0 && for i in 1:(N - 1)
                dst, U = @constinferred TK.artin_braid(src, i)
                @test _isunitary(U)
                dst′, U′ = @constinferred TK.artin_braid(dst, i; inv = true)
                @test U' ≈ U′
            end
        end

        # Test double braid-unbraid
        if FusionStyle(I) isa UniqueFusion
            in = rand(collect(⊗(out...)))
            src = only(fusiontrees(out, in, isdual))
            dst, U = TK.artin_braid(src, 2)
            dst′, U′ = TK.artin_braid(dst, 3)
            dst″, U″ = TK.artin_braid(dst′, 3; inv = true)
            dst‴, U‴ = TK.artin_braid(dst″, 2; inv = true)
            @test U * U′ * U″ * U‴ ≈ 1
        else
            src = FusionTreeBlock{I}((out, ()), (isdual, ()))
            if length(src) > 0
                dst, U = TK.artin_braid(src, 2)
                dst′, U′ = TK.artin_braid(dst, 3)
                dst″, U″ = TK.artin_braid(dst′, 3; inv = true)
                dst‴, U‴ = TK.artin_braid(dst″, 2; inv = true)
                @test  _isone(U * U′ * U″ * U‴)
            end
        end
    end
    (BraidingStyle(I) isa HasBraiding) && @testset "Fusion tree $Istr: braiding and permuting" begin
        f = rand(collect(it))
        p = tuple(randperm(N)...)
        ip = invperm(p)
        levels = ntuple(identity, N)
        levels2 = p

        if FusionStyle(I) isa UniqueFusion
            f = only(fusiontrees(out, in, isdual))
        else
            f = FusionTreeBlock{I}((out, ()), (isdual, ()))
        end

        dst, U = @constinferred braid(f, (p, ()), (levels, ()))
        @test _isunitary(U)
        dst′, U′ = braid(dst, (ip, ()), (levels2, ()))
        @test U' ≈ U′


        if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
            if FusionStyle(I) isa UniqueFusion
                A = permutedims(fusiontensor(f), (p..., N + 1))
                A′ = U * fusiontensor(dst)
                @test A ≈ A′
            else
                A = map(x -> permutedims(fusiontensor(x[1]), (p..., N + 1)), fusiontrees(f))
                A′ = map(fusiontensor ∘ first, fusiontrees(dst))
                for (i, Ai) in enumerate(A)
                    Aj = sum(A′ .* U[:, i])
                    @test Ai ≈ Aj
                end
            end
        end
    end

    FusionStyle(I) isa UniqueFusion && @testset "Fusion tree $Istr: merging" begin
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

        @constinferred TK.merge(f1, f2, first(in1 ⊗ in2), 1)
        if !(FusionStyle(I) isa GenericFusion)
            @constinferred TK.merge(f1, f2, first(in1 ⊗ in2), 1)
            @constinferred TK.merge(f1, f2, first(in1 ⊗ in2))
        end
        @test dim(in1) * dim(in2) ≈ sum(
            abs2(coeff) * dim(c) for c in in1 ⊗ in2
                for μ in 1:Nsymbol(in1, in2, c)
                for (f, coeff) in TK.merge(f1, f2, c, μ)
        )

        if BraidingStyle(I) isa HasBraiding
            for c in in1 ⊗ in2
                R = Rsymbol(in1, in2, c)
                for μ in 1:Nsymbol(in1, in2, c)
                    trees1 = TK.merge(f1, f2, c, μ)

                    # test merge and braid interplay
                    trees2 = Dict{keytype(trees1), complex(valtype(trees1))}()
                    trees3 = Dict{keytype(trees1), complex(valtype(trees1))}()
                    for ν in 1:Nsymbol(in2, in1, c)
                        for (t, coeff) in TK.merge(f2, f1, c, ν)
                            trees2[t] = get(trees2, t, zero(valtype(trees2))) + coeff * R[μ, ν]
                        end
                    end
                    perm = ((N .+ (1:N))..., (1:N)...)
                    levels = ntuple(identity, 2 * N)
                    for (t, coeff) in trees1
                        t′, coeff′ = braid(t, perm, levels)
                        trees3[t′] = get(trees3, t′, zero(valtype(trees3))) + coeff * coeff′
                    end
                    for (t, coeff) in trees3
                        coeff′ = get(trees2, t, zero(coeff))
                        @test isapprox(coeff, coeff′; atol = 1.0e-12, rtol = 1.0e-12)
                    end

                    # test via conversion
                    if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
                        Af1 = fusiontensor(f1)
                        Af2 = fusiontensor(f2)
                        Af0 = fusiontensor(FusionTree((f1.coupled, f2.coupled), c, (false, false), (), (μ,)))
                        _Af = TensorOperations.tensorcontract(
                            1:(N + 2), Af1, [1:N; -1], Af0, [-1; N + 1; N + 2]
                        )
                        Af = TensorOperations.tensorcontract(
                            1:(2N + 1), Af2, [N .+ (1:N); -1], _Af, [1:N; -1; 2N + 1]
                        )
                        Af′ = zero(Af)
                        for (f, coeff) in trees1
                            Af′ .+= coeff .* convert(Array, f)
                        end
                        @test Af ≈ Af′
                    end
                end
            end
        end
    end

    if I <: ProductSector
        N = 3
    else
        N = 4
    end
    if UnitStyle(I) isa SimpleUnit
        out = random_fusion(I, Val(N))
        numtrees = count(n -> true, fusiontrees((out..., map(dual, out)...)))
        while !(0 < numtrees < 100)
            out = random_fusion(I, Val(N))
            numtrees = count(n -> true, fusiontrees((out..., map(dual, out)...)))
        end
        incoming = rand(collect(⊗(out...)))
        f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
        f2 = rand(collect(fusiontrees(out[randperm(N)], incoming, ntuple(n -> rand(Bool), N))))
    else
        out = random_fusion(I, Val(N))
        out2 = random_fusion(I, Val(N))
        tp = ⊗(out...)
        tp2 = ⊗(out2...)
        while isempty(intersect(tp, tp2)) # guarantee fusion to same coloring
            out2 = random_fusion(I, Val(N))
            tp2 = ⊗(out2...)
        end
        @test_throws ArgumentError fusiontrees((out..., map(dual, out)...))
        incoming = rand(collect(intersect(tp, tp2)))
        f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
        f2 = rand(collect(fusiontrees(out2, incoming, ntuple(n -> rand(Bool), N)))) # no permuting
    end
    incoming = rand(collect(⊗(out...)))

    if FusionStyle(I) isa UniqueFusion
        f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
        f2 = rand(collect(fusiontrees(out[randperm(N)], incoming, ntuple(n -> rand(Bool), N))))
        src = (f1, f2)
        if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
            A = fusiontensor(src)
        end
    else
        src = FusionTreeBlock{I}((out, out), (ntuple(n -> rand(Bool), N), ntuple(n -> rand(Bool), N)))
        if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
            A = map(fusiontensor, fusiontrees(src))
        end
    end

    @testset "Double fusion tree $Istr: repartitioning" begin
        for n in 0:(2 * N)
            dst, U = @constinferred TK.repartition(src, $n)
            # @test _isunitary(U)

            dst′, U′ = repartition(dst, N)
            @test _isone(U * U′)

            if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
                all_inds = (ntuple(identity, numout(src))..., reverse(ntuple(i -> i + numout(src), numin(src)))...)
                p₁ = ntuple(i -> all_inds[i], numout(dst))
                p₂ = reverse(ntuple(i -> all_inds[i + numout(dst)], numin(dst)))
                if FusionStyle(I) isa UniqueFusion
                    @test permutedims(A, (p₁..., p₂...)) ≈ U * fusiontensor(dst)
                else
                    A′ = map(Base.Fix2(permutedims, (p₁..., p₂...)), A)
                    A″ = map(fusiontensor, fusiontrees(dst))
                    for (i, Ai) in enumerate(A′)
                        Aj = sum(A″ .* U[:, i])
                        @test Ai ≈ Aj
                    end
                end
            end
        end
    end

    BraidingStyle(I) isa SymmetricBraiding && @testset "Double fusion tree $Istr: permutation" begin
        for n in 0:(2N)
            p = (randperm(2 * N)...,)
            p1, p2 = p[1:n], p[(n + 1):(2N)]
            ip = invperm(p)
            ip1, ip2 = ip[1:N], ip[(N + 1):(2N)]

            dst, U = @constinferred TensorKit.permute(src, (p1, p2))
            # @test _isunitary(U)
            dst′, U′ = @constinferred TensorKit.permute(dst, (ip1, ip2))
            # @test U' ≈ U′
            @test _isone(U * U′)

            if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
                if FusionStyle(I) isa UniqueFusion
                    @test permutedims(A, (p1..., p2...)) ≈ U * fusiontensor(dst)
                else
                    A′ = map(Base.Fix2(permutedims, (p1..., p2...)), A)
                    A″ = map(fusiontensor, fusiontrees(dst))
                    for (i, Ai) in enumerate(A′)
                        Aj = sum(A″ .* U[:, i])
                        @test Ai ≈ Aj
                    end
                end
            end
        end
    end
    @testset "Double fusion tree $Istr: transposition" begin
        for n in 0:(2N)
            i0 = rand(1:(2N))
            p = mod1.(i0 .+ (1:(2N)), 2N)
            ip = mod1.(-i0 .+ (1:(2N)), 2N)
            p′ = tuple(getindex.(Ref(vcat(1:N, (2N):-1:(N + 1))), p)...)
            p1, p2 = p′[1:n], p′[(2N):-1:(n + 1)]
            ip′ = tuple(getindex.(Ref(vcat(1:n, (2N):-1:(n + 1))), ip)...)
            ip1, ip2 = ip′[1:N], ip′[(2N):-1:(N + 1)]

            dst, U = @constinferred transpose(src, (p1, p2))
            # @test _isunitary(U)
            dst′, U′ = @constinferred transpose(dst, (ip1, ip2))
            # @test U' ≈ U′
            @test _isone(U * U′)

            if BraidingStyle(I) isa Bosonic
                dst″, U″ = permute(src, (p1, p2))
                @test U″ ≈ U
            end

            if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
                if FusionStyle(I) isa UniqueFusion
                    @test permutedims(A, (p1..., p2...)) ≈ U * fusiontensor(dst)
                else
                    A′ = map(Base.Fix2(permutedims, (p1..., p2...)), A)
                    A″ = map(fusiontensor, fusiontrees(dst))
                    for (i, Ai) in enumerate(A′)
                        Aj = sum(U[:, i] .* A″)
                        @test Ai ≈ Aj
                    end
                end
            end
        end
    end
    @testset "Double fusion tree $Istr: planar trace" begin
        if FusionStyle(I) isa UniqueFusion
            f1, f1 = src
            dst, U = transpose((f1, f1), ((N + 1, 1:N..., ((2N):-1:(N + 3))...), (N + 2,)))
            d1 = zip((dst,), (U,))
        else
            f1, f1 = first(fusiontrees(src))
            src′ = FusionTreeBlock{I}((f1.uncoupled, f1.uncoupled), (f1.isdual, f1.isdual))
            dst, U = transpose(src′, ((N + 1, 1:N..., ((2N):-1:(N + 3))...), (N + 2,)))
            d1 = zip(fusiontrees(dst), U[:, 1])
        end

        f1front, = TK.split(f1, N - 1)
        T = sectorscalartype(I)
        d2 = Dict{typeof((f1front, f1front)), T}()
        for ((f1′, f2′), coeff′) in d1
            for ((f1′′, f2′′), coeff′′) in TK.planar_trace(
                    (f1′, f2′), ((2:N...,), (1, ((2N):-1:(N + 3))...)), ((N + 1,), (N + 2,))
                )
                coeff = coeff′ * coeff′′
                d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff
            end
        end
        for ((f1_, f2_), coeff) in d2
            if (f1_, f2_) == (f1front, f1front)
                @test coeff ≈ dim(f1.coupled) / dim(f1front.coupled)
            else
                @test abs(coeff) < 1.0e-12
            end
        end
    end
    TK.empty_globalcaches!()
end
