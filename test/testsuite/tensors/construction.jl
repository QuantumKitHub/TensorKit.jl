function basic_tensor_properties(t, W, T::Type, AT::Type)
    return @testset "Basic tensor properties $AT" begin
        #@test @testinferred(hash(t)) == hash(deepcopy(t))
        @test scalartype(t) == T
        @test norm(t) == 0
        @test codomain(t) == W
        @test space(t) == (W ← one(W))
        @test domain(t) == one(W)
        @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, AT}
    end
end

function basic_blocks_properties(t, W)
    bs = @testinferred blocks(t)
    (c, b1), state = @testinferred Nothing iterate(bs)
    @test c == first(blocksectors(W))
    next = @testinferred Nothing iterate(bs, state)
    b2 = @testinferred block(t, first(blocksectors(t)))
    @test b1 == b2
    @test eltype(bs) === Pair{typeof(c), typeof(b1)}
    @test typeof(b1) === TensorKit.blocktype(t)
    return @test typeof(c) === sectortype(t)
end

function tensor_dict_conversion(t)
    return @testset "Tensor Dict conversion" begin
        @test adapt(Vector{scalartype(t)}, t) ≈ convert(TensorMap, convert(Dict, t))
    end
end

function tensor_array_conversion(t, W)
    return @testset "Tensor Array conversion" begin
        a = @testinferred convert(Array, t)
        b = reshape(a, dim(codomain(W)), dim(domain(W)))
        @test adapt(Vector{scalartype(t)}, t) ≈ @testinferred TensorMap(a, W)
        @test adapt(Vector{scalartype(t)}, t) ≈ @testinferred TensorMap(b, W)
        @test t === @testinferred TensorMap(t.data, W)
    end
end

function empty_tensor_array_conversion(t, AT)
    return @testset "Empty tensor array conversion" begin
        a = convert(Array, t)
        @test norm(a) == 0
    end
end

function real_and_imaginary_parts(t)
    return @testset "Real and imaginary parts" begin
        tr = @testinferred real(t)
        @test scalartype(tr) <: Real
        @test real(convert(Array, t)) == convert(Array, tr)

        ti = @testinferred imag(t)
        @test scalartype(ti) <: Real
        @test imag(convert(Array, t)) == convert(Array, ti)

        tc = @inferred complex(t)
        @test scalartype(tc) <: Complex
        @test complex(convert(Array, t)) == convert(Array, tc)

        tc2 = @inferred complex(tr, ti)
        @test tc2 ≈ tc
    end
end

function tensor_conversion(t)
    return @testset "Tensor conversion" begin
        @test typeof(convert(TensorMap, t')) == typeof(t)
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
        @test typeof(convert(typeof(tc), t')) == typeof(tc)
        @test Base.promote_typeof(t, tc) == typeof(tc)
        @test Base.promote_typeof(tc, t) == typeof(tc + t)
    end
end
