# Fusion tree manipulations

# Single fusion trees
# -------------------
@testsuite :fusiontrees "iterate and printing" I -> begin
    N = 5
    out = random_fusion(I, Val(N))
    isdual = ntuple(n -> rand(Bool), N)
    incoupled = rand(collect(⊗(out...))) # renamed to not break infix `in` in this scope
    numtrees = length(fusiontrees(out, incoupled, isdual))
    @test numtrees == count(n -> true, fusiontrees(out, incoupled, isdual))
    while !(0 < numtrees < 30) && !(one(incoupled) in ⊗(out...))
        out = ntuple(n -> randsector(I), N)
        incoupled = rand(collect(⊗(out...)))
        numtrees = length(fusiontrees(out, incoupled, isdual))
        @test numtrees == count(n -> true, fusiontrees(out, incoupled, isdual))
    end
    it = @constinferred fusiontrees(out, incoupled, isdual)
    @constinferred Nothing iterate(it)
    f, s = iterate(it)
    @constinferred Nothing iterate(it, s)
    @test f == @constinferred first(it)
    @test eval(Meta.parse(sprint(show, f; context = (:module => @__MODULE__)))) == f
end

@testsuite :fusiontrees "constructor properties" I -> begin
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
@testsuite :fusiontrees "split and join" I -> begin
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

@testsuite :fusiontrees "multi Fmove" I -> begin
    N = 6
    uncoupled = random_fusion(I, Val(N))
    coupled = rand(collect(⊗(uncoupled...)))
    isdualrest = ntuple(n -> rand(Bool), N - 1)
    for isdual in ((false, isdualrest...), (true, isdualrest...))
        trees = collect(fusiontrees(uncoupled, coupled, isdual))
        trees = trees[randperm(length(trees))[1:rand(1:min(5, length(trees)))]] # limit number of tests?
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
            trees = trees[randperm(length(trees))[1:rand(1:min(5, length(trees)))]] # limit number of tests?
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

@testsuite :fusiontrees "insertat" I -> begin
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

@testsuite :fusiontrees "merging" I -> begin
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
@testsuite :fusiontrees "elementary planar trace" I -> begin
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

# Double fusion trees
# --------------------
function _random_doubletree_setup(I::Type{<:Sector})
    N = I <: ProductSector ? 3 : 4
    A = nothing

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
    return N, src, A
end

@testsuite :fusiontrees "bending" I -> begin
    _, src, A = _random_doubletree_setup(I)
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

@testsuite :fusiontrees "folding" I -> begin
    _, src, A = _random_doubletree_setup(I)
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

@testsuite :fusiontrees "repartitioning" I -> begin
    N, src, A = _random_doubletree_setup(I)
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

@testsuite :fusiontrees "transposition" I -> begin
    N, src, A = _random_doubletree_setup(I)
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

@testsuite :fusiontrees "permutation and braiding" I -> begin
    BraidingStyle(I) isa HasBraiding || return nothing
    N, src, A = _random_doubletree_setup(I)
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

@testsuite :fusiontrees "planar trace" I -> begin
    N, src, A = _random_doubletree_setup(I)
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
