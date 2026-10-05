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
