using Test, TestExtras
using TensorKit
using TensorKit: SectorVector, diagonalblockstructure

spacelist = (
    (ℂ^4)',
    Vect[Z2Irrep](0 => 2, 1 => 3),
    Vect[SU2Irrep](0 => 2, 1 => 1)',
    Vect[FibonacciAnyon](:I => 2, :τ => 2),
)

@testset "SectorVector with space $V" for V in spacelist
    I = sectortype(V)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @timedtestset "Constructors ($T)" begin
            v = @testinferred SectorVector{T}(undef, V)
            @test v isa SectorVector{T, I, Vector{T}}
            @test length(v) == reduceddim(V)
            @test sectortype(v) == I
            @test storagetype(typeof(v)) == Vector{T}

            structure = diagonalblockstructure(V ← V)
            data = rand(T, reduceddim(V))
            v2 = @testinferred SectorVector(data, structure)
            @test parent(v2) === data
            @test v2.structure == structure

            v3 = @testinferred SectorVector{T, I, Vector{T}}(undef, V)
            @test length(v3) == reduceddim(V)
        end

        @timedtestset "AbstractVector interface ($T)" begin
            v = SectorVector{T}(undef, V)
            rand!(parent(v))
            @test @testinferred(eltype(v)) == T
            @test length(v) == length(parent(v))
            oldval = v[1]
            v[1] = one(T)
            @test v[1] == one(T) == parent(v)[1]
        end

        @timedtestset "similar and copy ($T)" begin
            v = SectorVector{T}(undef, V)
            rand!(parent(v))

            v1 = @testinferred similar(v)
            @test v1 isa typeof(v)
            @test v1.structure == v.structure
            @test length(v1) == length(v)

            v2 = @testinferred similar(v, ComplexF64)
            @test eltype(v2) == ComplexF64
            @test v2.structure == v.structure

            v3 = @testinferred similar(v, V)
            @test v3 isa SectorVector{T, I, Vector{T}}
            @test v3.structure == diagonalblockstructure(V ← V)

            v4 = @testinferred similar(v, ComplexF64, V)
            @test v4 isa SectorVector{ComplexF64, I, Vector{ComplexF64}}
            @test v4.structure == diagonalblockstructure(V ← V)
            @test length(v4) == reduceddim(V)

            v5 = @testinferred copy(v)
            @test v5 == v
            @test parent(v5) !== parent(v)
            @test v5.structure == v.structure
        end

        @timedtestset "AbstractDict interface ($T)" begin
            v = SectorVector{T}(undef, V)
            rand!(parent(v))
            @test @testinferred(keytype(v)) == I
            @test @testinferred(keytype(typeof(v))) == I
            @test valtype(v) == SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}

            for c in sectors(V)
                @test @testinferred(haskey(v, c))
                @test v[c] == view(parent(v), v.structure[c])
                newvals = rand(T, length(v[c]))
                v[c] = newvals
                @test v[c] == newvals
            end
            @test collect(@testinferred(keys(v))) == collect(sectors(V))
            @test collect(values(v)) == [v[c] for c in keys(v)]
            @test pairs(v) == Dict(c => v[c] for c in keys(v))
        end

        @timedtestset "VectorInterface ($T)" begin
            v1 = SectorVector{T}(undef, V)
            rand!(parent(v1))
            v2 = SectorVector{T}(undef, V)
            rand!(parent(v2))

            z = @testinferred zerovector(v1, T)
            @test all(iszero, parent(z))
            @test z.structure == v1.structure
            v1c = deepcopy(v1)
            zerovector!(v1c)
            @test all(iszero, parent(v1c))
            zerovector!!(v1c)
            @test all(iszero, parent(v1c))

            α = rand(T)
            vs = @testinferred scale(v1, α)
            @test parent(vs) ≈ α * parent(v1)
            v1c = deepcopy(v1)
            scale!(v1c, α)
            @test parent(v1c) ≈ α * parent(v1)
            v1c = deepcopy(v1)
            scale!!(v1c, α)
            @test parent(v1c) ≈ α * parent(v1)

            β = rand(T)
            # VectorInterface convention: add(x, y, α, β) computes β*x + α*y
            va = @testinferred add(v1, v2, α, β)
            @test parent(va) ≈ β * parent(v1) + α * parent(v2)
            v1c = deepcopy(v1)
            add!(v1c, v2, α, β)
            @test parent(v1c) ≈ β * parent(v1) + α * parent(v2)
            v1c = deepcopy(v1)
            add!!(v1c, v2, α, β)
            @test parent(v1c) ≈ β * parent(v1) + α * parent(v2)

            @test @testinferred(inner(v1, v2)) ≈ dot(v1, v2)
            v3 = SectorVector{T}(undef, V ⊕ V)
            @test_throws SpaceMismatch inner(v1, v3)
        end

        @timedtestset "LinearAlgebra ($T)" begin
            v = SectorVector{T}(undef, V)
            rand!(parent(v))
            @test norm(v)^2 ≈ dot(v, v)
            @test norm(v) ≈ @testinferred(norm(v, 2))
            for p in (1, 2, Inf)
                @test norm(v, p) isa real(T)
                @test norm(v, p) >= 0
            end
        end
    end
end
