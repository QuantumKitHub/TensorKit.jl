# GradedSpace

@testsuite :spaces "graded space" I -> begin
    if Base.IteratorSize(values(I)) === Base.IsInfinite()
        set = unique(vcat(allunits(I)..., [randsector(I) for k in 1:10]))
        gen = (c => 2 for c in set)
    else
        gen = (values(I)[k] => (k + 1) for k in 1:length(values(I)))
    end
    V = GradedSpace(gen)
    @test eval(Meta.parse(type_repr(typeof(V)))) == typeof(V)
    @test eval_show(V) == V
    @test eval_show(V') == V'
    @test V' == GradedSpace(gen; dual = true)
    @test V' == GradedSpace{I}(gen; dual = true)
    @test V == @testinferred GradedSpace(gen...)
    @test V' == @testinferred GradedSpace(gen...; dual = true)
    @test V == @testinferred GradedSpace(tuple(gen...))
    @test V' == @testinferred GradedSpace(tuple(gen...); dual = true)
    @test V == @testinferred GradedSpace(Dict(gen))
    @test V' == @testinferred GradedSpace(Dict(gen); dual = true)
    @test V == @inferred Vect[I](gen)
    @test V' == @testinferred Vect[I](gen; dual = true)
    @test V == @testinferred Vect[I](gen...)
    @test V' == @testinferred Vect[I](gen...; dual = true)
    @test V == @testinferred Vect[I](Dict(gen))
    @test V' == @testinferred Vect[I](Dict(gen); dual = true)
    @test V == @testinferred typeof(V)(c => dim(V, c) for c in sectors(V))
    if I isa ZNIrrep
        @test V == @testinferred typeof(V)(V.dims)
        @test V' == @testinferred typeof(V)(V.dims; dual = true)
    end
    @test @testinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test V == GradedSpace(reverse(collect(gen))...)
    @test eval_show(V) == V
    @test eval_show(typeof(V)) == typeof(V)
    # space with no sectors
    @test dim(@testinferred(zerospace(V))) == 0
    # space with unit(s), always test as if multifusion
    W = @testinferred GradedSpace(unit => 1 for unit in allunits(I))
    dict = Dict(unit => 1 for unit in allunits(I))
    @test W == GradedSpace(dict)
    @test W == GradedSpace(push!(dict, randsector(I) => 0))
    @test @testinferred(zerospace(V)) == GradedSpace(unit => 0 for unit in allunits(I))
    randunit = rand(collect(allunits(I)))
    @test_throws ArgumentError("Sector $(randunit) appears multiple times") GradedSpace(randunit => 1, randunit => 3)

    @test isunitspace(W)
    @test @testinferred(unitspace(V)) == W == unitspace(typeof(V))
    if UnitStyle(I) isa SimpleUnit
        @test @testinferred(leftunitspace(V)) == W == @testinferred(rightunitspace(V))
    else
        @test_throws ArgumentError leftunitspace(V)
        @test_throws ArgumentError rightunitspace(V)
    end
    @test eval_show(W) == W
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, GradedSpace)
    @test isa(V, GradedSpace{I})
    @test @testinferred(dual(V)) == @testinferred(conj(V)) == @testinferred(adjoint(V)) != V
    @test @testinferred(field(V)) == ℂ
    @test @testinferred(sectortype(V)) == I
    slist = @testinferred sectors(V)
    @test @testinferred(hassector(V, first(slist)))
    @test @testinferred(dim(V)) == sum(dim(s) * dim(V, s) for s in slist)
    @test @testinferred(reduceddim(V)) == sum(dim(V, s) for s in slist)
    @testinferred dim(V, first(slist))
    if hasfusiontensor(I)
        @test @testinferred(axes(V)) == Base.OneTo(dim(V))
    end
    @test @testinferred(⊕(V, zerospace(V))) == V
    @test @testinferred(⊕(V, V)) == Vect[I](c => 2dim(V, c) for c in sectors(V))
    @test @testinferred(⊕(V, V, V, V)) == Vect[I](c => 4dim(V, c) for c in sectors(V))
    @test @testinferred(⊕(V, unitspace(V))) == Vect[I](c => isunit(c) + dim(V, c) for c in sectors(V))
    @test @testinferred(fuse(V, unitspace(V))) == V
    d = Dict{I, Int}()
    for a in sectors(V), b in sectors(V)
        for c in a ⊗ b
            d[c] = get(d, c, 0) + dim(V, a) * dim(V, b) * Nsymbol(a, b, c)
        end
    end
    @test @testinferred(fuse(V, V)) == GradedSpace(d)
    @test @testinferred(flip(V)) == Vect[I](conj(c) => dim(V, c) for c in sectors(V))'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test @testinferred(⊕(V, V)) == @testinferred supremum(V, ⊕(V, V))
    @test V == @testinferred infimum(V, ⊕(V, V))
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))

    u = first(allunits(I))
    @test infimum(V, GradedSpace(u => 3)) == GradedSpace(u => 2)
    @test_throws SpaceMismatch (⊕(V, V'))
end
