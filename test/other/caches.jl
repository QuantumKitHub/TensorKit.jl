using Test, TestExtras
using TensorKit
using TensorKit: Cached, sectorstructure, degeneracystructure

@testset "Cached structure keys" begin
    V1 = SU2Space(0 => 2, 1 // 2 => 3)
    V2 = SU2Space(0 => 4, 1 // 2 => 1)
    W1, W2 = V1 ⊗ V1 ← V1, V2 ⊗ V2 ← V2
    empty_globalcaches!()
    s1 = @constinferred sectorstructure(W1)
    @test s1 === @constinferred sectorstructure(W2)
    fresh = Cached.uncached(sectorstructure, W1)
    @test s1.blocksectors == fresh.blocksectors
    @test s1.fusiontrees == fresh.fusiontrees
    @test parent(only(Cached.cachekey(sectorstructure, W1))) === W1
    @test length(only(Cached.cache_info(sectorstructure)).second) == 1
    @test (@constinferred degeneracystructure(W1)).totaldim !=
        (@constinferred degeneracystructure(W2)).totaldim
    @test length(only(Cached.cache_info(degeneracystructure)).second) == 2
end

@testset "TensorKit cached-value sizes" begin
    for I in (Z2Irrep, SU2Irrep, A4Irrep, FibonacciAnyon)
        labels = collect(Iterators.take(values(I), 2))
        V = GradedSpace{I}(labels[1] => 2, labels[2] => 2)
        W = V ⊗ V ← V ⊗ V
        p, levels = ((2, 1), (4, 3)), (1, 2, 3, 4)
        transformer = TensorKit.treebraider(W, W, p, false, levels)
        for value in (sectorstructure(W), degeneracystructure(W), transformer)
            bytes = @constinferred Cached.cachesize(value)
            @test 0 < bytes <= Base.summarysize(value)
        end
        for src in TensorKit.fusionblocks(W)
            result = TensorKit.fsbraid(src, p, ((1, 2), (3, 4)))
            @test 0 < (@constinferred Cached.cachesize(result)) <= Base.summarysize(result)
        end

        # Two distinct transformer entries share structure buffers. Each entry must
        # retain its own size budget even if the degeneracy cache has been cleared.
        Cached.set_cache_size!(TensorKit.treebraider, Cached.cachesize(transformer); by = Cached.cachesize)
        try
            TensorKit.treebraider(W, W, p, false, levels)
            TensorKit.treebraider(W, W, ((1, 2), (3, 4)), false, levels)
            Cached.empty_caches!(degeneracystructure)
            cache = only(Cached.cache_info(TensorKit.treebraider)).second
            stats = Cached.cache_stats(cache)
            @test stats.by === Cached.cachesize
            @test stats.currentsize <= stats.maxsize
            @test stats.length <= 1
            @test stats.currentsize == sum(Cached.cachesize, values(cache); init = 0)

            Cached.set_cache_size!(TensorKit.treebraider, 0; by = Cached.cachesize)
            @test isempty(cache)
            @test TensorKit.treebraider(W, W, p, false, levels).data == transformer.data
            @test isempty(cache)
        finally
            Cached.set_cache_size!(TensorKit.treebraider, 10_000)
        end
    end

    # An empty transformer still costs space, and arbitrary-precision coefficients
    # include their referenced storage instead of just the array's pointer slots.
    strides = TensorKit.StridedStructure{2}[]
    empty_transformer = TensorKit.UniqueTreeTransformer(Tuple{Float64, Int, Int}[], strides, strides)
    @test Cached.cachesize(empty_transformer) > 0
    big_transformer = TensorKit.UniqueTreeTransformer([(BigFloat(1), 1, 1)], strides, strides)
    @test Cached.cachesize(big_transformer) >= Base.summarysize(big_transformer.data)
end

# A cache owned by another module must survive TensorKit's cache clearing.
Cached.@cached unrelated_cache(x) = [x]

@testset "TensorKit cache management" begin
    retained = unrelated_cache(1)
    extra = Dict(:key => :value)
    push!(TensorKit.GLOBAL_CACHES, :extension_cache => extra)
    try
        V = SU2Space(0 => 2, 1 // 2 => 2)
        t = rand(V ⊗ V ← V ⊗ V)
        @test (@constinferred transpose(transpose(t))) ≈ t
        p = ((2, 1), (4, 3))
        @test (@constinferred permute(permute(t, p), p)) ≈ t
        info = sprint(TensorKit.global_cache_info)
        @test occursin("treebraider", info)
        @test occursin("extension_cache", info)
        @test empty_globalcaches!() === nothing
        @test all(isempty(cache) for (_, cache) in Cached.cache_info(TensorKit))
        @test isempty(extra)
        @test unrelated_cache(1) === retained
    finally
        pop!(TensorKit.GLOBAL_CACHES)
        Cached.empty_caches!(unrelated_cache)
    end
end
