# Product sectors
# ---------------
# The fusion trees of a `ProductSector` are pairs of fusion trees of two parts of the product (head
# and tail), such that the manipulations of a block are the Kronecker products of the manipulations
# of the head and tail blocks. A product with both `UniqueFusion` and other factors is split into
# its other factors and its `UniqueFusion` factors, which have a single tree pair; any other product
# is split into its first and its remaining factors.

# blocks of products of at least two sectors
const ProductSectorBlock = FusionTreeBlock{<:ProductSector{<:Tuple{Sector, Sector, Vararg{Sector}}}}

firstfactor(s::ProductSector) = first(s.sectors)
otherfactors(s::ProductSector{<:Tuple{Sector, Sector}}) = last(s.sectors)
otherfactors(s::ProductSector) = ProductSector(Base.tail(s.sectors))
firstfactortype(::Type{ProductSector{T}}) where {T} = fieldtype(T, 1)
otherfactorstype(::Type{ProductSector{Tuple{H, T}}}) where {H <: Sector, T <: Sector} = T
otherfactorstype(::Type{ProductSector{T}}) where {T} = ProductSector{Base.tuple_type_tail(T)}

Base.@assume_effects :foldable function ismixedproduct(::Type{I}) where {I <: ProductSector}
    isunique = map(S -> FusionStyle(S) isa UniqueFusion, TensorKitSectors._sectors(I))
    return any(isunique) && !all(isunique)
end

# the non-`UniqueFusion` and the `UniqueFusion` factors, with `Trivial` for no factors
partitionfactors(s::Sector) = FusionStyle(s) isa UniqueFusion ? (Trivial(), s) : (s, Trivial())
function partitionfactors(s::ProductSector)
    g, a = partitionfactors(otherfactors(s))
    f = firstfactor(s)
    return FusionStyle(f) isa UniqueFusion ? (g, f ⊠ a) : (f ⊠ g, a)
end
partitionfactortypes(::Type{I}) where {I <: Sector} =
    FusionStyle(I) isa UniqueFusion ? (Trivial, I) : (I, Trivial)
function partitionfactortypes(::Type{I}) where {I <: ProductSector}
    G, A = partitionfactortypes(otherfactorstype(I))
    F = firstfactortype(I)
    return FusionStyle(F) isa UniqueFusion ? (G, F ⊠ A) : (F ⊠ G, A)
end
# the inverse of `partitionfactors` for the factors of `I`
factortuple(s::ProductSector) = s.sectors
factortuple(::Trivial) = ()
factortuple(s::Sector) = (s,)
interleave(::Type{Tuple{}}, gs, as) = ()
function interleave(::Type{T}, gs, as) where {T <: Tuple}
    return if FusionStyle(fieldtype(T, 1)) isa UniqueFusion
        (first(as), interleave(Base.tuple_type_tail(T), gs, Base.tail(as))...)
    else
        (first(gs), interleave(Base.tuple_type_tail(T), Base.tail(gs), as)...)
    end
end
unpartition(::Type{ProductSector{T}}, g, a) where {T} =
    ProductSector{T}(interleave(T, factortuple(g), factortuple(a)))

headsector(s::ProductSector) =
    ismixedproduct(typeof(s)) ? first(partitionfactors(s)) : firstfactor(s)
tailsector(s::ProductSector) =
    ismixedproduct(typeof(s)) ? last(partitionfactors(s)) : otherfactors(s)
headsectortype(::Type{I}) where {I <: ProductSector} =
    ismixedproduct(I) ? first(partitionfactortypes(I)) : firstfactortype(I)
tailsectortype(::Type{I}) where {I <: ProductSector} =
    ismixedproduct(I) ? last(partitionfactortypes(I)) : otherfactorstype(I)
joinsectors(::Type{I}, h, t) where {I <: ProductSector} =
    ismixedproduct(I) ? unpartition(I, h, t) : h ⊠ t

# the vertex labels of a product are the linear indices of the head and tail labels, tail first
vertexlines(f::FusionTree) = (f.uncoupled[1], f.innerlines..., f.coupled)
vertexmultiplicities(f::FusionTree) =
    ntuple(i -> Nsymbol(vertexlines(f)[i], f.uncoupled[i + 1], vertexlines(f)[i + 1]), length(f) - 1)

maptree(proj, f::FusionTree, vertices) = FusionTree(
    map(proj, f.uncoupled), proj(f.coupled), f.isdual, map(proj, f.innerlines), vertices
)
function headtail(f::FusionTree)
    vertices_h = vertices_t = f.vertices
    if FusionStyle(f) isa GenericFusion && length(f) > 1
        ns = vertexmultiplicities(maptree(tailsector, f, map(Returns(1), f.vertices)))
        vertices_h = map((μ, n) -> div(μ - 1, n) + 1, f.vertices, ns)
        vertices_t = map((μ, n) -> mod(μ - 1, n) + 1, f.vertices, ns)
    end
    return maptree(headsector, f, vertices_h), maptree(tailsector, f, vertices_t)
end
function jointree(::Type{I}, h::FusionTree, t::FusionTree) where {I}
    vertices = h.vertices
    if FusionStyle(I) isa GenericFusion && length(h) > 1
        vertices = map((μ, ν, n) -> ν + (μ - 1) * n, h.vertices, t.vertices, vertexmultiplicities(t))
    end
    lift(a, b) = joinsectors(I, a, b)
    return fusiontreetype(I, length(h))(
        map(lift, h.uncoupled, t.uncoupled), lift(h.coupled, t.coupled), h.isdual,
        map(lift, h.innerlines, t.innerlines), vertices
    )
end

# the canonical order of a product block is that of the Kronecker product of the head and tail
# blocks, with the head running fastest
productblock(::Type{I}, head::FusionTreeBlock, tail::FusionTreeBlock) where {I} = FusionTreeBlock(
    vec(
        [
            (jointree(I, h₁, t₁), jointree(I, h₂, t₂))
                for (h₁, h₂) in fusiontrees(head), (t₁, t₂) in fusiontrees(tail)
        ]
    )
)
function FusionTreeBlock{I}(
        uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}},
        isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}};
        sizehint::Int = 0
    ) where {I <: ProductSector{<:Tuple{Sector, Sector, Vararg{Sector}}}, N₁, N₂}
    # a block of `UniqueFusion` sectors has a single tree pair, in any order
    FusionStyle(I) isa UniqueFusion && return fusiontreeblock(I, uncoupled, isdual; sizehint)
    # the tail holds the `UniqueFusion` factors of a mixed product, which are cheaper to construct
    tail = FusionTreeBlock{tailsectortype(I)}(map.(tailsector, uncoupled), isdual)
    isempty(tail) && return FusionTreeBlock(Vector{fusiontreetype(I, N₁, N₂)}())
    head = FusionTreeBlock{headsectortype(I)}(map.(headsector, uncoupled), isdual)
    return productblock(I, head, tail)
end

# the distinct tree pairs of one factor, and the position of every tree pair among them
function distincttrees(trees, othertrees)
    FusionStyle(first(trees[1])) isa UniqueFusion && return trees[1:1], fill(1, length(trees))
    FusionStyle(first(othertrees[1])) isa UniqueFusion && return trees, eachindex(trees)
    distinct = unique(trees)
    return distinct, indexin(trees, distinct)
end

# apply `transform` to the block `src` through its head and tail, with the recoupling
# matrix with scalar type `T`
function product_transform(transform::F, ::Type{T}, src::FusionTreeBlock{I}) where {F, T, I}
    trees = map(fusiontrees(src)) do (f₁, f₂)
        h₁, t₁ = headtail(f₁)
        h₂, t₂ = headtail(f₂)
        return (h₁, h₂), (t₁, t₂)
    end
    trees_h, trees_t = map(first, trees), map(last, trees)
    src_h, cols_h = distincttrees(trees_h, trees_t)
    src_t, cols_t = distincttrees(trees_t, trees_h)
    dst_h, U_h = transform(FusionTreeBlock(src_h))
    dst_t, U_t = transform(FusionTreeBlock(src_t))

    dst = productblock(I, dst_h, dst_t)
    U = Matrix{T}(undef, length(dst), length(src))
    for (col, i, j) in zip(axes(U, 2), cols_h, cols_t)
        row = 0
        for jt in axes(U_t, 1), jh in axes(U_h, 1)
            U[row += 1, col] = U_h[jh, i] * U_t[jt, j]
        end
    end
    return dst => U
end
