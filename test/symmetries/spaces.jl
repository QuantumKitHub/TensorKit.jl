using Test, TestExtras
using TensorKit
using TensorKit: hassector, type_repr, HomSpace

# TODO: remove this once type_repr works for all included types
using TensorKitSectors

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

"""
    eval_show(x)

Use `show` to generate a string representation of `x`, then parse and evaluate the resulting expression.
"""
function eval_show(x)
    str = sprint(show, x; context = (:module => @__MODULE__))
    ex = Meta.parse(str)
    return eval(ex)
end

@timedtestset "Fields" begin
    @test isa(ℝ, Field)
    @test isa(ℂ, Field)
    @test eval_show(ℝ) == ℝ
    @test eval_show(ℂ) == ℂ
    @test ℝ ⊆ ℝ
    @test ℝ ⊆ ℂ
    @test ℂ ⊆ ℂ
    @test !(ℂ ⊆ ℝ)

    for T in (Int8, Int16, Int32, Int64, BigInt)
        @test one(T) ∈ ℝ
        @test one(Rational{T}) ∈ ℝ
        @test !(one(Complex{T}) ∈ ℝ)
        @test !(one(Complex{Rational{T}}) ∈ ℝ)
        @test one(T) ∈ ℂ
        @test one(Rational{T}) ∈ ℂ
        @test one(Complex{T}) ∈ ℂ
        @test one(Complex{Rational{T}} ∈ ℂ)

        @test T ⊆ ℝ
        @test Rational{T} ⊆ ℝ
        @test !(Complex{T} ⊆ ℝ)
        @test !(Complex{Rational{T}} ⊆ ℝ)
        @test T ⊆ ℂ
        @test Rational{T} ⊆ ℂ
        @test Complex{T} ⊆ ℂ
        @test Complex{Rational{T}} ⊆ ℂ
    end
    for T in (Float32, Float64, BigFloat)
        @test one(T) ∈ ℝ
        @test !(one(Complex{T}) ∈ ℝ)
        @test one(T) ∈ ℂ
        @test one(Complex{T} ∈ ℂ)

        @test T ⊆ ℝ
        @test !(Complex{T} ⊆ ℝ)
        @test T ⊆ ℂ
        @test Complex{T} ⊆ ℂ
    end
end

@timedtestset "ElementarySpace: CartesianSpace" begin
    d = 2
    V = ℝ^d
    @test eval_show(V) == V
    @test eval_show(typeof(V)) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, CartesianSpace)
    @test !isdual(V)
    @test !isdual(V')
    @test V == CartesianSpace(Trivial() => d) == CartesianSpace(Dict(Trivial() => d))
    @test @constinferred(hash(V)) == hash(deepcopy(V))
    @test V == @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V))
    @test field(V) == ℝ
    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(zerospace(V))) == 0
    @test (sectors(zerospace(V))...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    @test ℝ^d == ℝ[](d) == CartesianSpace(d) == typeof(V)(d)
    W = @constinferred ℝ^1
    @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
    @test @constinferred(zerospace(V)) == ℝ^0 == zerospace(typeof(V))
    @test @constinferred(⊕(V, zerospace(V))) == V
    @test @constinferred(⊕(V, V)) == ℝ^(2d)
    @test @constinferred(⊕(V, unitspace(V))) == ℝ^(d + 1)
    @test @constinferred(⊕(V, V, V, V)) == ℝ^(4d)
    @test @constinferred(fuse(V, V)) == ℝ^(d^2)
    @test @constinferred(fuse(V, V', V, V')) == ℝ^(d^4)
    @test @constinferred(flip(V)) == V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))
    @test @constinferred(infimum(V, ℝ^3)) == V
    @test @constinferred(supremum(V', ℝ^3)) == ℝ^3
end

@timedtestset "ElementarySpace: ComplexSpace" begin
    d = 2
    V = ℂ^d
    @test eval_show(V) == V
    @test eval_show(V') == V'
    @test eval_show(typeof(V)) == typeof(V)
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test isa(InnerProductStyle(V), HasInnerProduct)
    @test isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test isa(V, ComplexSpace)
    @test !isdual(V)
    @test isdual(V')
    @test V == ComplexSpace(Trivial() => d) == ComplexSpace(Dict(Trivial() => d))
    @test @constinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test @constinferred(dual(V)) == @constinferred(conj(V)) == @constinferred(adjoint(V)) != V
    @test @constinferred(field(V)) == ℂ
    @test @constinferred(sectortype(V)) == Trivial
    @test @constinferred(sectortype(V)) == Trivial
    @test ((@constinferred sectors(V))...,) == (Trivial(),)
    @test length(sectors(V)) == 1
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test dim(@constinferred(zerospace(V))) == 0
    @test (sectors(zerospace(V))...,) == ()
    @test @constinferred(axes(V)) == Base.OneTo(d)
    @test ℂ^d == Vect[Trivial](d) == Vect[](Trivial() => d) == ℂ[](d) == typeof(V)(d)
    W = @constinferred ℂ^1
    @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
    @test @constinferred(zerospace(V)) == ℂ^0 == zerospace(typeof(V))
    @test @constinferred(⊕(V, zerospace(V))) == V
    @test @constinferred(⊕(V, V)) == ℂ^(2d)
    @test_throws SpaceMismatch (⊕(V, V'))
    # promote_except = ErrorException("promotion of types $(typeof(ℝ^d)) and " *
    #                                 "$(typeof(ℂ^d)) failed to change any arguments")
    # @test_throws promote_except (⊕(ℝ^d, ℂ^d))
    @test_throws ErrorException (⊗(ℝ^d, ℂ^d))
    @test @constinferred(⊕(V, V)) == ℂ^(2d)
    @test @constinferred(⊕(V, unitspace(V))) == ℂ^(d + 1)
    @test @constinferred(⊕(V, V, V, V)) == ℂ^(4d)
    @test @constinferred(fuse(V, V)) == ℂ^(d^2)
    @test @constinferred(fuse(V, V', V, V')) == ℂ^(d^4)
    @test @constinferred(flip(V)) == V'
    @test flip(V) ≅ V
    @test flip(V) ≾ V
    @test flip(V) ≿ V
    @test V ≺ ⊕(V, V)
    @test !(V ≻ ⊕(V, V))
    @test @constinferred(infimum(V, ℂ^3)) == V
    @test @constinferred(supremum(V', (ℂ^3)')) == dual(ℂ^3) == conj(ℂ^3)
end

@timedtestset "ElementarySpace: GeneralSpace" begin
    d = 2
    V = GeneralSpace{ℂ}(d)
    @test eval_show(V) == V
    @test eval_show(dual(V)) == dual(V)
    @test eval_show(conj(V)) == conj(V)
    @test eval_show(typeof(V)) == typeof(V)
    @test !isdual(V)
    @test isdual(V')
    @test !isdual(conj(V))
    @test isdual(conj(V'))
    @test !TensorKit.isconj(V)
    @test !TensorKit.isconj(V')
    @test TensorKit.isconj(conj(V))
    @test TensorKit.isconj(conj(V'))
    @test isa(V, VectorSpace)
    @test isa(V, ElementarySpace)
    @test !isa(InnerProductStyle(V), HasInnerProduct)
    @test !isa(InnerProductStyle(V), EuclideanInnerProduct)
    @test @constinferred(hash(V)) == hash(deepcopy(V)) != hash(V')
    @test @constinferred(dual(V)) != @constinferred(conj(V)) != V
    @test @constinferred(field(V)) == ℂ
    @test @constinferred(sectortype(V)) == Trivial
    @test @constinferred(hassector(V, Trivial()))
    @test @constinferred(dim(V)) == d == @constinferred(dim(V, Trivial()))
    @test @constinferred(axes(V)) == Base.OneTo(d)
end

@timedtestset "ElementarySpace: $(type_repr(Vect[I]))" for I in sectorlist
    if Base.IteratorSize(values(I)) === Base.IsInfinite()
        set = unique(vcat(unit(I), [randsector(I) for k in 1:10]))
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
    # space with a single sector
    W = @constinferred GradedSpace(unit(I) => 1)
    @test W == GradedSpace(unit(I) => 1, randsector(I) => 0)
    @test @constinferred(unitspace(V)) == W == unitspace(typeof(V))
    @test @constinferred(zerospace(V)) == GradedSpace(unit(I) => 0)
    # randsector never returns trivial sector, so this cannot error
    @test_throws ArgumentError GradedSpace(unit(I) => 1, randsector(I) => 0, unit(I) => 3)
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
    @test infimum(V, GradedSpace(unit(I) => 3)) == GradedSpace(unit(I) => 2)
    @test_throws SpaceMismatch (⊕(V, V'))
end

@timedtestset "ProductSpace{ℂ}" begin
    V1, V2, V3, V4 = ℂ^1, ℂ^2, ℂ^3, ℂ^4
    P = @constinferred ProductSpace(V1, V2, V3, V4)
    @test eval_show(P) == P
    @test eval_show(typeof(P)) == typeof(P)
    @test isa(P, VectorSpace)
    @test isa(P, CompositeSpace)
    @test spacetype(P) == ComplexSpace
    @test sectortype(P) == Trivial
    @test @constinferred(hash(P)) == hash(deepcopy(P)) != hash(P')
    @test P == deepcopy(P)
    @test P == typeof(P)(P...)
    @constinferred (x -> tuple(x...))(P)
    @test @constinferred(dual(P)) == P'
    @test @constinferred(field(P)) == ℂ
    @test @constinferred(*(V1, V2, V3, V4)) == P
    @test @constinferred(⊗(V1, V2, V3, V4)) == P
    @test @constinferred(⊗(V1, V2 ⊗ V3 ⊗ V4)) == P
    @test @constinferred(⊗(V1 ⊗ V2, V3 ⊗ V4)) == P
    @test @constinferred(⊗(V1, V2, V3 ⊗ V4)) == P
    @test @constinferred(⊗(V1, V2 ⊗ V3, V4)) == P
    @test V1 * V2 * unitspace(V1) * V3 * V4 ==
        @constinferred(insertleftunit(P, 3)) ==
        @constinferred(insertrightunit(P, 2))
    @test @constinferred(removeunit(V1 * V2 * unitspace(V1)' * V3 * V4, 3)) == P
    @test fuse(V1, V2', V3) ≅ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≾ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≿ V1 ⊗ V2' ⊗ V3
    @test fuse(flip(V1), V2, flip(V3)) ≅ V1 ⊗ V2 ⊗ V3
    @test @constinferred(⊗(P)) == P
    @test @constinferred(⊗(V1)) == ProductSpace(V1)
    @test eval_show(⊗(V1)) == ⊗(V1)
    @test @constinferred(one(V1)) == @constinferred(one(typeof(V1))) ==
        @constinferred(one(P)) == @constinferred(one(typeof(P))) ==
        ProductSpace{ComplexSpace}(())
    @test eval_show(one(P)) == one(P)
    @test @constinferred(⊗(one(P), P)) == P
    @test @constinferred(⊗(P, one(P))) == P
    @test @constinferred(⊗(one(P), one(P))) == one(P)
    @test @constinferred(adjoint(P)) == dual(P) == V4' ⊗ V3' ⊗ V2' ⊗ V1'
    @test @constinferred(dims(P)) == map(dim, (V1, V2, V3, V4))
    @test @constinferred(dim(P)) == prod(dim, (V1, V2, V3, V4))
    @test @constinferred(dim(P, 2)) == dim(V2)
    @test @constinferred(dim(one(P))) == 1
    @test first(@constinferred(sectors(P))) == (Trivial(), Trivial(), Trivial(), Trivial())
    @test first(@constinferred(sectors(one(P)))) == ()
    cube(x) = x^3
    @test @constinferred(cube(V1)) == V1 ⊗ V1 ⊗ V1
    N = 3
    @test V1^N == V1 ⊗ V1 ⊗ V1
    @test P^2 == P ⊗ P
    @test @constinferred(dims(P, first(sectors(P)))) == dims(P)
    @test ((@constinferred blocksectors(P))...,) == (Trivial(),)
    @test isempty(blocksectors(P ⊗ ℂ^0))
    @test isempty(@constinferred(sectors(P ⊗ ℂ^0)))
    @test @constinferred(blockdim(P, first(blocksectors(P)))) == dim(P)
    @test @constinferred(blockdim(P, Trivial())) == dim(P)
    @test @constinferred(blockdim(one(P), Trivial())) == 1
    @test Base.IteratorEltype(P) == Base.IteratorEltype(typeof(P)) == Base.IteratorEltype(P.spaces)
    @test Base.IteratorSize(P) == Base.IteratorSize(typeof(P)) == Base.IteratorSize(P.spaces)
    @test Base.eltype(P) == Base.eltype(typeof(P)) == typeof(V1)
    @test eltype(collect(P)) == typeof(V1)
    @test collect(P) == [V1, V2, V3, V4]
    @test_throws MethodError P ⊕ P
end

@timedtestset "ProductSpace{SU₂Space}" begin
    V1, V2, V3 = SU₂Space(0 => 3, 1 // 2 => 1), SU₂Space(0 => 2, 1 => 1),
        SU₂Space(1 // 2 => 1, 1 => 1)'
    P = @constinferred ProductSpace(V1, V2, V3)
    @test eval_show(P) == P
    @test eval_show(typeof(P)) == typeof(P)
    @test isa(P, VectorSpace)
    @test isa(P, CompositeSpace)
    @test spacetype(P) == SU₂Space
    @test sectortype(P) == Irrep[SU₂] == SU2Irrep
    @test @constinferred(hash(P)) == hash(deepcopy(P)) != hash(P')
    @test @constinferred(dual(P)) == P'
    @test @constinferred(field(P)) == ℂ
    @test @constinferred(*(V1, V2, V3)) == P
    @test @constinferred(⊗(V1, V2, V3)) == P
    @test @constinferred(adjoint(P)) == dual(P) == V3' ⊗ V2' ⊗ V1'
    @test V1 * V2 * unitspace(V1)' * V3 ==
        @constinferred(insertleftunit(P, 3; conj = true)) ==
        @constinferred(insertrightunit(P, 2; conj = true))
    @test P == @constinferred(removeunit(insertleftunit(P, 3), 3))
    @test fuse(V1, V2', V3) ≅ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≾ V1 ⊗ V2' ⊗ V3 ≾ fuse(V1 ⊗ V2' ⊗ V3)
    @test fuse(V1, V2') ⊗ V3 ≾ V1 ⊗ V2' ⊗ V3
    @test fuse(V1, V2', V3) ≿ V1 ⊗ V2' ⊗ V3 ≿ fuse(V1 ⊗ V2' ⊗ V3)
    @test V1 ⊗ fuse(V2', V3) ≿ V1 ⊗ V2' ⊗ V3
    @test fuse(flip(V1) ⊗ V2) ⊗ flip(V3) ≅ V1 ⊗ V2 ⊗ V3
    @test @constinferred(⊗(V1)) == ProductSpace(V1)
    @test @constinferred(one(V1)) == @constinferred(one(typeof(V1))) ==
        @constinferred(one(P)) == @constinferred(one(typeof(P)))
    @test @constinferred(dims(P)) == map(dim, (V1, V2, V3))
    @test @constinferred(dim(P)) == prod(dim, (V1, V2, V3))
    @test @constinferred(dim(one(P))) == 1
    @test first(@constinferred(sectors(one(P)))) == ()
    @test @constinferred(blockdim(one(P), Irrep[SU₂](0))) == 1
    for s in @constinferred(sectors(P))
        @test hassector(P, s)
        @test @constinferred(dims(P, s)) == dim.((V1, V2, V3), s)
    end
    @test sum(dim(c) * blockdim(P, c) for c in @constinferred(blocksectors(P))) == dim(P)
end

@timedtestset "Deligne tensor product of spaces" begin
    V1 = SU₂Space(0 => 3, 1 // 2 => 1)
    V2 = SU₂Space(0 => 2, 1 => 1)'
    V3 = ℤ₃Space(0 => 3, 1 => 2, 2 => 1)
    V4 = ℂ^3

    for W1 in (V1, V2, V3, V4)
        for W2 in (V1, V2, V3, V4)
            for W3 in (V1, V2, V3, V4)
                for W4 in (V1, V2, V3, V4)
                    Ws = @constinferred(W1 ⊠ W2 ⊠ W3 ⊠ W4)
                    @test Ws == @constinferred((W1 ⊠ W2) ⊠ (W3 ⊠ W4)) ==
                        @constinferred(((W1 ⊠ W2) ⊠ W3) ⊠ W4) ==
                        @constinferred((W1 ⊠ (W2 ⊠ W3)) ⊠ W4) ==
                        @constinferred(W1 ⊠ ((W2 ⊠ W3)) ⊠ W4) ==
                        @constinferred(W1 ⊠ (W2 ⊠ (W3 ⊠ W4)))
                    I1, I2, I3, I4 = map(sectortype, (W1, W2, W3, W4))
                    I = sectortype(Ws)
                    @test I == @constinferred((I1 ⊠ I2) ⊠ (I3 ⊠ I4)) ==
                        @constinferred(((I1 ⊠ I2) ⊠ I3) ⊠ I4) ==
                        @constinferred((I1 ⊠ (I2 ⊠ I3)) ⊠ I4) ==
                        @constinferred(I1 ⊠ ((I2 ⊠ I3)) ⊠ I4) ==
                        @constinferred(I1 ⊠ (I2 ⊠ (I3 ⊠ I4)))
                    @test dim(Ws) == dim(W1) * dim(W2) * dim(W3) * dim(W4)
                end
            end
        end
    end
    @test sectortype(@constinferred((V1 ⊗ V2) ⊠ V3)) == @constinferred(Irrep[SU₂ × ℤ₃])
    @test dim((V1 ⊗ V2) ⊠ V3) == dim(V1) * dim(V2) * dim(V3)
    @test sectortype((V1 ⊗ V2) ⊠ V3 ⊠ V4) == Irrep[SU₂ × ℤ₃]
    @test dim((V1 ⊗ V2) ⊠ V3 ⊠ V4) == dim(V1) * dim(V2) * dim(V3) * dim(V4)
    @test fuse(V2 ⊠ V4) == fuse(V4 ⊠ V2) == SU₂Space(0 => 6, 1 => 3)
    @test fuse(V3 ⊠ V4) == fuse(V4 ⊠ V3) == ℤ₃Space(0 => 9, 1 => 6, 2 => 3)
end

@timedtestset "HomSpace" begin
    for (V1, V2, V3, V4, V5) in (Vtr, Vℤ₃, VSU₂)
        W = HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
        @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
        @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
        @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
        @test eval_show(W) == W
        @test eval_show(typeof(W)) == typeof(W)
        @test spacetype(W) == typeof(V1)
        @test sectortype(W) == sectortype(V1)
        @test W[1] == V1
        @test W[2] == V2
        @test W[3] == V3'
        @test W[4] == V4'
        @test W[5] == V5'
        @test @constinferred(hash(W)) == hash(deepcopy(W)) != hash(W')
        @test W == deepcopy(W)
        @test W == @constinferred permute(W, ((1, 2), (3, 4, 5)))
        @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
        @test (V1 ⊗ V2 ← V1 ⊗ V2) == @constinferred TensorKit.compose(W, W')
        @test (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ unitspace(V5)) ==
            @constinferred(insertleftunit(W)) ==
            @constinferred(insertrightunit(W))
        @test @constinferred(removeunit(insertleftunit(W), $(numind(W) + 1))) == W
        @test (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5 ⊗ unitspace(V5)') ==
            @constinferred(insertleftunit(W; conj = true)) ==
            @constinferred(insertrightunit(W; conj = true))
        @test (unitspace(V1) ⊗ V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5) ==
            @constinferred(insertleftunit(W, 1)) ==
            @constinferred(insertrightunit(W, 0))
        @test (V1 ⊗ V2 ⊗ unitspace(V1) ← V3 ⊗ V4 ⊗ V5) ==
            @constinferred(insertrightunit(W, 2))
        @test (V1 ⊗ V2 ← unitspace(V1) ⊗ V3 ⊗ V4 ⊗ V5) ==
            @constinferred(insertleftunit(W, 3))
        @test @constinferred(removeunit(insertleftunit(W, 3), 3)) == W
        @test @constinferred(insertrightunit(one(V1) ← V1, 0)) == (unitspace(V1) ← V1)
        @test_throws BoundsError insertleftunit(one(V1) ← V1, 0)
    end
end

@timedtestset "show and friends" begin
    V = U1Space(i => 1 for i in 1:3)
    @test string(V) == "$(type_repr(typeof(V)))(1 => 1, 2 => 1, 3 => 1)"
    @test string(V') == "$(type_repr(typeof(V)))(1 => 1, 2 => 1, 3 => 1)'"
    @test sprint((x, y) -> show(x, MIME"text/plain"(), y), V) == "$(type_repr(typeof(V)))(…) of dim 3:\n 1 => 1\n 2 => 1\n 3 => 1"
    @test sprint((x, y) -> show(x, MIME"text/plain"(), y), V') == "$(type_repr(typeof(V)))(…)' of dim 3:\n 1 => 1\n 2 => 1\n 3 => 1"
end

TensorKit.empty_globalcaches!()
