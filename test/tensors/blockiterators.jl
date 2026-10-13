using Test
using TensorKit
using LinearAlgebra: Diagonal

@testset "Positional block collections" begin
    spaces = (
        ComplexSpace(4), Z2Space(0 => 3, 1 => 2), U1Space(0 => 3, 1 => 2),
        SU2Space(0 => 3, 1 // 2 => 2), Vect[FibonacciAnyon](:I => 3, :τ => 2),
    )
    collection_types = []
    for V in spaces, T in (Float64, ComplexF64)
        t = randn(T, V ← V)
        bs = @inferred TensorKit.positionalblocks(t)
        T === Float64 && push!(collection_types, typeof(bs))
        @test bs.structure === TensorKit.degeneracystructure(space(t)).blockstructure
        @test length(bs) == length(blocks(t))
        for (i, (_, b)) in enumerate(blocks(t))
            @test bs[i] == b
            @test typeof(bs[i]) === typeof(b)
            @test TensorKit.positionalblocks(t')[i] == b'
            old = b[1, 1]
            bs[i][1, 1] = old + 1
            @test b[1, 1] == old + 1
            b[1, 1] = old
        end
        @test_throws BoundsError bs[0]
        @test_throws BoundsError bs[length(bs) + 1]

        # Exercise the actual forward kernels with matrix, diagonal and vector outputs.
        for f in (qr_compact, qr_full, lq_compact, lq_full, svd_compact, svd_full, left_polar, right_polar)
            @test reduce(*, f(t)) ≈ t
        end
        @test svd_vals(t) ≈ svd_vals(t')
        h = project_hermitian(t)
        D, Q = eigh_full(h)
        @test Q * D * Q' ≈ h
        vals = svd_vals(t)
        for (i, (_, b)) in enumerate(blocks(vals))
            @test TensorKit.positionalblocks(vals)[i] == b
        end
        for (i, (_, b)) in enumerate(blocks(D))
            @test TensorKit.positionalblocks(D)[i] == b
            @test TensorKit.positionalblocks(D)[i] isa Diagonal
        end
        @test exponential((0.5, h)) ≈ exponential(0.5 * h)
    end
    @test all(==(first(collection_types)), collection_types)

    # Reuse metadata for matching sector sets and erase tensor arity at the boundary.
    V = Z2Space(0 => 2, 1 => 2)
    t = randn(Float64, V ⊗ V ← V)
    Q, R = qr_compact(t)
    aligned = @inferred TensorKit.alignedblocks(t, Q, R)
    @test all(bs -> bs isa TensorKit.ReshapedBlocks{Vector{Float64}}, aligned)
    @test aligned[1].structure === TensorKit.positionalblocks(t).structure
    @test Q * R ≈ t
    @test_throws SectorMismatch TensorKit.alignedblocks(t, randn(Float64, ComplexSpace(4) ← ComplexSpace(4)))

    # Equal numbers of blocks with different labels must not be zipped by position.
    V = U1Space(0 => 3, 1 => 2)
    W = U1Space(0 => 2, 2 => 4)
    a = randn(Float64, V ← V)
    b = randn(Float64, W ← W)
    aligned = TensorKit.alignedblocks(a, b)
    sectors = union(blocksectors(a), blocksectors(b))
    @test length(aligned[1]) == 3
    for (i, c) in enumerate(sectors), (ts, bs) in zip((a, b), aligned)
        @test bs[i] == block(ts, c)
    end

    # Preserve nonzero dimensions of absent blocks in full factorizations.
    t = randn(ComplexF64, V ← W)
    for f in (qr_full, lq_full, svd_full, qr_compact, svd_compact)
        out = f(t)
        @test reduce(*, out) ≈ t
        tensors = (t, out...)
        aligned = TensorKit.alignedblocks(tensors...)
        sectors = union(blocksectors.(tensors)...)
        for (i, c) in enumerate(sectors), (ts, bs) in zip(tensors, aligned)
            @test size(bs[i]) == size(block(ts, c))
            @test bs[i] == block(ts, c)
        end
    end

    # Empty tensors and generic implementations retain their original behavior.
    empty = zeros(Float64, U1Space() ← U1Space())
    @test isempty(collect(TensorKit.positionalblocks(empty)))
    calls = Ref(0)
    TensorKit.foreachblockvalue(empty) do _
        calls[] += 1
    end
    @test calls[] == 0
    braid = BraidingTensor{Float64}(V, W)
    @test TensorKit.positionalblocks(braid) == map(last, collect(blocks(braid)))
end
