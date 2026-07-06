# GradedSpace
# ===========
# Ported from the sector-parametrized part of test/symmetries/spaces.jl.

"""
    eval_show(x)

Use `show` to generate a string representation of `x`, then parse and evaluate the resulting expression.
"""
function eval_show(x) #TODO: move this function to setup so it doesn't repeat
    str = sprint(show, x; context = (:module => @__MODULE__))
    ex = Meta.parse(str)
    return eval(ex)
end

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
    @test V == @constinferred GradedSpace(gen...)
    @test V' == @constinferred GradedSpace(gen...; dual = true)
    @test V == @constinferred GradedSpace(tuple(gen...))
    @test V' == @constinferred GradedSpace(tuple(gen...); dual = true)
    @test V == @constinferred GradedSpace(Dict(gen))
    @test V' == @constinferred GradedSpace(Dict(gen); dual = true)
    @test V == @inferred Vect[I](gen)
    @test V' == @constinferred Vect[I](gen; dual = true)
    @test V == @constinferred Vect[I](gen...)
    @test V' == @constinferred Vect[I](gen...; dual = true)
    @test V == @constinferred Vect[I](Dict(gen))
    @test V' == @constinferred Vect[I](Dict(gen); dual = true)
    @test V == @constinferred typeof(V)(c => dim(V, c) for c in sectors(V))
    if I isa ZNIrrep
        @test V == @constinferred typeof(V)(V.dims)
        @test V' == @constinferred typeof(V)(V.dims; dual = true)
    end
    @test @constinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test V == GradedSpace(reverse(collect(gen))...)
    @test eval_show(V) == V
    @test eval_show(typeof(V)) == typeof(V)
    # space with no sectors
    @test dim(@constinferred(zerospace(V))) == 0
    # space with unit(s), always test as if multifusion
    W = @constinferred GradedSpace(unit => 1 for unit in allunits(I))
    dict = Dict(unit => 1 for unit in allunits(I))
    @test W == GradedSpace(dict)
    @test W == GradedSpace(push!(dict, randsector(I) => 0))
    @test @constinferred(zerospace(V)) == GradedSpace(unit => 0 for unit in allunits(I))
    randunit = rand(collect(allunits(I)))
    @test_throws ArgumentError("Sector $(randunit) appears multiple times") GradedSpace(randunit => 1, randunit => 3)

    @test isunitspace(W)
    @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
    if UnitStyle(I) isa SimpleUnit
        @test @constinferred(leftunitspace(V)) == W == @constinferred(rightunitspace(V))
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
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V)) != V
    @test @constinferred(field(V)) == ℂ
    @test @constinferred(sectortype(V)) == I
    slist = @constinferred sectors(V)
    @test @constinferred(hassector(V, first(slist)))
    @test @constinferred(dim(V)) == sum(dim(s) * dim(V, s) for s in slist)
    @test @constinferred(reduceddim(V)) == sum(dim(V, s) for s in slist)
    @constinferred dim(V, first(slist))
    if hasfusiontensor(I)
        @test @constinferred(axes(V)) == Base.OneTo(dim(V))
    end
    @test @constinferred(⊕(V, zerospace(V))) == V
    @test @constinferred(⊕(V, V)) == Vect[I](c => 2dim(V, c) for c in sectors(V))
    @test @constinferred(⊕(V, V, V, V)) == Vect[I](c => 4dim(V, c) for c in sectors(V))
    @test @constinferred(⊕(V, unitspace(V))) == Vect[I](c => isunit(c) + dim(V, c) for c in sectors(V))
    @test @constinferred(fuse(V, unitspace(V))) == V
    d = Dict{I, Int}()
    for a in sectors(V), b in sectors(V)
        for c in a ⊗ b
            d[c] = get(d, c, 0) + dim(V, a) * dim(V, b) * Nsymbol(a, b, c)
        end
    end
    @test @constinferred(fuse(V, V)) == GradedSpace(d)
    @test @constinferred(flip(V)) == Vect[I](conj(c) => dim(V, c) for c in sectors(V))'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test @constinferred(⊕(V, V)) == @constinferred supremum(V, ⊕(V, V))
    @test V == @constinferred infimum(V, ⊕(V, V))
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))

    u = first(allunits(I))
    @test infimum(V, GradedSpace(u => 3)) == GradedSpace(u => 2)
    @test_throws SpaceMismatch (⊕(V, V'))
end
