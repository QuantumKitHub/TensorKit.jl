using Test, TestExtras
using TensorKit
using TensorKit: FusionTreeBlock
import TensorKit as TK
using Random: randperm
using TensorOperations
using TupleTools

# TODO: remove this once type_repr works for all included types
using TensorKitSectors


@timedtestset "Double fusion trees for $(TensorKit.type_repr(I))" verbose = true for I in (fast_tests ? fast_sectorlist : sectorlist)
    Istr = TensorKit.type_repr(I)
    N = I <: ProductSector ? 3 : 4

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

    @testset "Double fusion tree: bending" begin
        # single bend
        dst, U = @constinferred TK.bendright(src)
        dst2, U2 = @constinferred TK.bendleft(dst)
        @test src == dst2
        @test _isone(U2 * U)
        # double bend
        dst1, U1 = @constinferred TK.bendleft(src)
        dst2, U2 = @constinferred TK.bendleft(dst1)
        dst3, U3 = @constinferred TK.bendright(dst2)
        dst4, U4 = @constinferred TK.bendright(dst3)
        @test src == dst4
        @test _isone(U4 * U3 * U2 * U1)

        if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
            all_inds = (ntuple(identity, numout(src))..., reverse(ntuple(i -> i + numout(src), numin(src)))...)
            p₁ = ntuple(i -> all_inds[i], numout(dst2))
            p₂ = reverse(ntuple(i -> all_inds[i + numout(dst2)], numin(dst2)))
            U = U2 * U1
            if FusionStyle(I) isa UniqueFusion
                @test permutedims(A, (p₁..., p₂...)) ≈ U * fusiontensor(dst)
            else
                A′ = map(Base.Fix2(permutedims, (p₁..., p₂...)), A)
                A″ = map(fusiontensor, fusiontrees(dst2))
                for (i, Ai) in enumerate(A′)
                    @test Ai ≈ sum(A″ .* U[:, i])
                end
            end
        end
    end

    @testset "Double fusion tree: folding" begin
        # single bend
        dst, U = @constinferred TK.foldleft(src)
        dst2, U2 = @constinferred TK.foldright(dst)
        @test src == dst2
        @test _isone(U2 * U)
        # double bend
        dst1, U1 = @constinferred TK.foldright(src)
        dst2, U2 = @constinferred TK.foldright(dst1)
        dst3, U3 = @constinferred TK.foldleft(dst2)
        dst4, U4 = @constinferred TK.foldleft(dst3)
        @test src == dst4
        @test _isone(U4 * U3 * U2 * U1)

        if (BraidingStyle(I) isa Bosonic) && hasfusiontensor(I)
            all_inds = TupleTools.circshift((ntuple(identity, numout(src))..., reverse(ntuple(i -> i + numout(src), numin(src)))...), -2)
            p₁ = ntuple(i -> all_inds[i], numout(dst2))
            p₂ = reverse(ntuple(i -> all_inds[i + numout(dst2)], numin(dst2)))
            U = U2 * U1
            if FusionStyle(I) isa UniqueFusion
                @test permutedims(A, (p₁..., p₂...)) ≈ U * fusiontensor(dst2)
            else
                A′ = map(Base.Fix2(permutedims, (p₁..., p₂...)), A)
                A″ = map(fusiontensor, fusiontrees(dst2))
                for (i, Ai) in enumerate(A′)
                    @test Ai ≈ sum(A″ .* U[:, i])
                end
            end
        end
    end

    @testset "Double fusion tree: repartitioning" begin
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
                        @test Ai ≈ sum(A″ .* U[:, i])
                    end
                end
            end
        end
    end

    @testset "Double fusion tree: transposition" begin
        for n in 0:(2N)
            i0 = rand(1:(2N))
            p = mod1.(i0 .+ (1:(2N)), 2N)
            ip = mod1.(-i0 .+ (1:(2N)), 2N)
            p′ = tuple(getindex.(Ref(vcat(1:N, (2N):-1:(N + 1))), p)...)
            p1, p2 = p′[1:n], p′[(2N):-1:(n + 1)]
            ip′ = tuple(getindex.(Ref(vcat(1:n, (2N):-1:(n + 1))), ip)...)
            ip1, ip2 = ip′[1:N], ip′[(2N):-1:(N + 1)]

            dst, U = @constinferred transpose(src, (p1, p2))
            dst′, U′ = @constinferred transpose(dst, (ip1, ip2))
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
                        @test Ai ≈ sum(U[:, i] .* A″)
                    end
                end
            end
        end
    end

    BraidingStyle(I) isa HasBraiding && @testset "Double fusion tree: permutation and braiding" begin
        for n in 0:(2N)
            p = (randperm(2 * N)...,)
            p1, p2 = p[1:n], p[(n + 1):(2N)]
            ip = invperm(p)
            ip1, ip2 = ip[1:N], ip[(N + 1):(2N)]
            levels = ntuple(identity, 2N)
            l1, l2 = levels[1:N], levels[(N + 1):(2N)]
            ilevels = TupleTools.getindices(levels, p)
            il1, il2 = ilevels[1:n], ilevels[(n + 1):(2N)]

            if BraidingStyle(I) isa SymmetricBraiding
                dst, U = @constinferred TensorKit.permute(src, (p1, p2))
            else
                dst, U = @constinferred TensorKit.braid(src, (p1, p2), (l1, l2))
            end

            # check norm-preserving
            if FusionStyle(I) isa UniqueFusion
                @test abs(U) ≈ 1
            else
                dim1 = map(fusiontrees(src)) do (f1, f2)
                    return dim(f1.coupled)
                end
                dim2 = map(fusiontrees(dst)) do (f1, f2)
                    return dim(f1.coupled)
                end
                @test vec(sum(abs2.(U) .* dim2; dims = 1)) ≈ dim1
            end

            # check reversible
            if BraidingStyle(I) isa SymmetricBraiding
                dst′, U′ = @constinferred TensorKit.permute(dst, (ip1, ip2))
            else
                dst′, U′ = @constinferred TensorKit.braid(dst, (ip1, ip2), (il1, il2))
            end
            @test _isone(U * U′)

            # check fusiontensor compatibility
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

    @testset "Double fusion tree: planar trace" begin
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
