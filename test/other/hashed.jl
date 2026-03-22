using Test, TestExtras
using TensorKit
using TensorKit: Hashed

@testset "Hashed" begin
    @testset "default constructor" begin
        h1 = @constinferred Hashed(42)
        h2 = Hashed(42)
        @test isequal(h1, h2)
        @test hash(h1) == hash(h2)
        @test parent(h1) == 42
    end

    @testset "custom hash function" begin
        # hash only the length, ignoring contents
        lenhash = (v, seed) -> hash(length(v), seed)
        h1 = Hashed([1, 2, 3], lenhash)
        h2 = Hashed([4, 5, 6], lenhash)
        @test hash(h1) == hash(h2)
        h3 = Hashed([1, 2], lenhash)
        @test hash(h1) != hash(h3)
    end

    @testset "custom isequal" begin
        # consider vectors equal if they have the same length
        lenequal = (a, b) -> length(a) == length(b)
        h1 = Hashed([1, 2, 3], Base.hash, lenequal)
        h2 = Hashed([4, 5, 6], Base.hash, lenequal)
        h3 = Hashed([1, 2], Base.hash, lenequal)
        @test isequal(h1, h2)
        @test !isequal(h1, h3)
    end

    @testset "Dict key usage" begin
        d = Dict(Hashed(1) => "one", Hashed(2) => "two")
        @test d[Hashed(1)] == "one"
        @test d[Hashed(2)] == "two"
        @test length(d) == 2
    end

    @testset "Dict with custom hash and isequal" begin
        lenhash = (v, seed) -> hash(length(v), seed)
        lenequal = (a, b) -> length(a) == length(b)
        d = Dict(Hashed([1, 2, 3], lenhash, lenequal) => "length3")
        # lookup with different contents but same length should succeed
        @test d[Hashed([7, 8, 9], lenhash, lenequal)] == "length3"
    end
end
