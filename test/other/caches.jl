using Test, TestExtras
using TensorKit
using TensorKit: MemoizationKit, sectorstructure, degeneracystructure

@testset "Cache structure keys" begin
    V1 = SU2Space(0 => 2, 1 // 2 => 3)
    V2 = SU2Space(0 => 4, 1 // 2 => 1)
    W1, W2 = V1 ⊗ V1 ← V1, V2 ⊗ V2 ← V2
    empty_globalcaches!()
    s1 = @constinferred sectorstructure(W1)
    @test s1 === @constinferred sectorstructure(W2)
    fresh = MemoizationKit.uncached(sectorstructure, W1)
    @test s1.blocksectors == fresh.blocksectors
    @test s1.fusiontrees == fresh.fusiontrees
    @test parent(only(MemoizationKit.cachekey(sectorstructure, W1))) === W1
    @test length(only(MemoizationKit.cache_info(sectorstructure)).second) == 1
    @test (@constinferred degeneracystructure(W1)).totaldim !=
        (@constinferred degeneracystructure(W2)).totaldim
    @test length(only(MemoizationKit.cache_info(degeneracystructure)).second) == 2
end

@testset "TensorKit cached-value sizes" begin
    for I in (Z2Irrep, SU2Irrep, A4Irrep, FibonacciAnyon)
        labels = collect(Iterators.take(values(I), 2))
        V = GradedSpace{I}(labels[1] => 2, labels[2] => 2)
        W = V ⊗ V ← V ⊗ V
        p, levels = ((2, 1), (4, 3)), (1, 2, 3, 4)
        transformer = TensorKit.treebraider(Vector{Float64}, W, W, p, false, levels)
        for value in (sectorstructure(W), degeneracystructure(W), transformer)
            bytes = @constinferred MemoizationKit.cachesize(value)
            @test 0 < bytes <= Base.summarysize(value)
        end
        for src in TensorKit.fusionblocks(W)
            result = TensorKit.fusiontreebraid(src, p, ((1, 2), (3, 4)))
            @test 0 < (@constinferred MemoizationKit.cachesize(result)) <= Base.summarysize(result)
        end

        # Two distinct transformer entries share structure buffers. Each entry must
        # retain its own size budget even if the degeneracy cache has been cleared.
        MemoizationKit.set_cache_size!(TensorKit.treebraider, MemoizationKit.cachesize(transformer); by = MemoizationKit.cachesize)
        try
            TensorKit.treebraider(Vector{Float64}, W, W, p, false, levels)
            TensorKit.treebraider(Vector{Float64}, W, W, ((1, 2), (3, 4)), false, levels)
            MemoizationKit.empty_caches!(degeneracystructure)
            cache = only(MemoizationKit.cache_info(TensorKit.treebraider)).second
            stats = MemoizationKit.cache_stats(cache)
            @test stats.by === MemoizationKit.cachesize
            @test stats.currentsize <= stats.maxsize
            @test stats.length <= 1
            @test stats.currentsize == sum(MemoizationKit.cachesize, values(cache); init = 0)

            MemoizationKit.set_cache_size!(TensorKit.treebraider, 0; by = MemoizationKit.cachesize)
            @test isempty(cache)
            @test TensorKit.treebraider(Vector{Float64}, W, W, p, false, levels).data == transformer.data
            @test isempty(cache)
        finally
            MemoizationKit.set_cache_size!(TensorKit.treebraider, 10_000)
        end
    end

    # An empty transformer still costs space, and arbitrary-precision coefficients
    # include their referenced storage instead of just the array's pointer slots.
    strides = TensorKit.StridedStructure{2}[]
    empty_transformer = TensorKit.UniqueTreeTransformer(Tuple{Float64, Int, Int}[], strides, strides)
    @test MemoizationKit.cachesize(empty_transformer) > 0
    big_transformer = TensorKit.UniqueTreeTransformer([(BigFloat(1), 1, 1)], strides, strides)
    @test MemoizationKit.cachesize(big_transformer) >= Base.summarysize(big_transformer.data)
end

# A cache owned by another module must survive TensorKit's cache clearing.
MemoizationKit.@cached unrelated_cache(x) = [x]

@testset "TensorKit cache management" begin
    retained = unrelated_cache(1)
    try
        V = SU2Space(0 => 2, 1 // 2 => 2)
        t = rand(V ⊗ V ← V ⊗ V)
        @test (@constinferred transpose(transpose(t))) ≈ t
        p = ((2, 1), (4, 3))
        @test (@constinferred permute(permute(t, p), p)) ≈ t
        info = sprint(TensorKit.global_cache_info)
        @test occursin("treebraider", info)
        @test empty_globalcaches!() === nothing
        @test all(isempty(cache) for (_, cache) in MemoizationKit.cache_info(TensorKit))
        @test unrelated_cache(1) === retained
    finally
        MemoizationKit.empty_caches!(unrelated_cache)
    end
end
