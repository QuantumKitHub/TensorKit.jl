using Test, TestExtras
using TensorKit
using TensorKit: type_repr


spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (fast_tests ? (Float64, ComplexF64) : (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat))
                t = @constinferred zeros(T, W)
                @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, Vector{T}}
                # Array type input
                t = @constinferred zeros(Vector{T}, W)
                @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, Vector{T}}
                # blocks
                bs = @constinferred blocks(t)
                if !isempty(blocksectors(t)) # multifusion space ending on module gives empty data
                    (c, b1), state = @constinferred Nothing iterate(bs)
                    @test c == first(blocksectors(W))
                    next = @constinferred Nothing iterate(bs, state)
                    b2 = @constinferred block(t, first(blocksectors(t)))
                    @test b1 == b2
                    @test eltype(bs) === Pair{typeof(c), typeof(b1)}
                    @test typeof(b1) === TensorKit.blocktype(t)
                    @test typeof(c) === sectortype(t)
                end
            end
        end
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                t = @constinferred rand(T, W)
                d = convert(Dict, t)
                @test t == convert(TensorMap, d)
            end
        end
        if hasfusiontensor(I) || I == Trivial
            @timedtestset "Tensor Array conversion" begin
                W1 = V1 ← one(V1)
                W2 = one(V2) ← V2
                W3 = V1 ⊗ V2 ← one(V1)
                W4 = V1 ← V2
                W5 = one(V1) ← V1 ⊗ V2
                W6 = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for W in (W1, W2, W3, W4, W5, W6)
                    for T in (Int, Float32, ComplexF64)
                        if T == Int
                            t = TensorMap{T}(undef, W)
                            for (_, b) in blocks(t)
                                rand!(b, -20:20)
                            end
                        else
                            t = @constinferred randn(T, W)
                        end
                        a = @constinferred convert(Array, t)
                        b = reshape(a, dim(codomain(W)), dim(domain(W)))
                        @test t ≈ @constinferred TensorMap(a, W)
                        @test t ≈ @constinferred TensorMap(b, W)
                        @test t === @constinferred TensorMap(t.data, W)
                    end
                end
                for T in (Int, Float32, ComplexF64)
                    t = randn(T, V1 ⊗ V2 ← zerospace(V1))
                    a = convert(Array, t)
                    @test norm(a) == 0
                end
            end
        end
        if hasfusiontensor(I)
            @timedtestset "Real and imaginary parts" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64, ComplexF32)
                    t = @constinferred randn(T, W, W)

                    tr = @constinferred real(t)
                    @test scalartype(tr) <: Real
                    @test real(convert(Array, t)) == convert(Array, tr)

                    ti = @constinferred imag(t)
                    @test scalartype(ti) <: Real
                    @test imag(convert(Array, t)) == convert(Array, ti)

                    tc = @inferred complex(t)
                    @test scalartype(tc) <: Complex
                    @test complex(convert(Array, t)) == convert(Array, tc)

                    tc2 = @inferred complex(tr, ti)
                    @test tc2 ≈ tc
                end
            end
        end
        @timedtestset "Tensor conversion" begin
            W = V1 ⊗ V2
            t = @constinferred randn(W ← W)
            @test typeof(convert(TensorMap, t')) == typeof(t)
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            @test typeof(convert(typeof(tc), t')) == typeof(tc)
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
        end
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "show tensors" begin
    for V in (ℂ^2, Z2Space(0 => 2, 1 => 2), SU2Space(0 => 2, 1 => 2))
        t1 = ones(Float32, V ⊗ V, V)
        t2 = randn(ComplexF64, V ⊗ V ⊗ V)
        t3 = randn(Float64, zero(V), zero(V))
        # test unlimited output
        for t in (t1, t2, t1', t2', t3)
            output = IOBuffer()
            summary(output, t)
            print(output, ":\n codomain: ")
            show(output, MIME("text/plain"), codomain(t))
            print(output, "\n domain: ")
            show(output, MIME("text/plain"), domain(t))
            print(output, "\n blocks: \n")
            first = true
            for (c, b) in blocks(t)
                first || print(output, "\n\n")
                print(output, " * ")
                show(output, MIME("text/plain"), c)
                print(output, " => ")
                show(output, MIME("text/plain"), b)
                first = false
            end
            outputstr = String(take!(output))
            @test outputstr == sprint(show, MIME("text/plain"), t)
        end

        # test limited output with a single block
        t = randn(Float64, V ⊗ V, V)' # we know there is a single space in the codomain, so that blocks have 2 rows
        output = IOBuffer()
        summary(output, t)
        print(output, ":\n codomain: ")
        show(output, MIME("text/plain"), codomain(t))
        print(output, "\n domain: ")
        show(output, MIME("text/plain"), domain(t))
        print(output, "\n blocks: \n")
        c = unit(sectortype(t))
        b = block(t, c)
        print(output, " * ")
        show(output, MIME("text/plain"), c)
        print(output, " => ")
        show(output, MIME("text/plain"), b)
        if length(blocks(t)) > 1
            print(output, "\n\n *   …   [output of 1 more block(s) truncated]")
        end
        outputstr = String(take!(output))
        @test outputstr == sprint(show, MIME("text/plain"), t; context = (:limit => true, :displaysize => (12, 100)))
    end
end
