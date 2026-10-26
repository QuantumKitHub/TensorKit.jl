# Product sectors
# ---------------
# FusionTree manipulations for `ProductSector` are done factor-by-factor: transformation matrices are
# computed by kronecker product of the individual ones.
# For efficiency we separate all `UniqueFusion` factors and handle these together, while handling the
# rest one-by-one.

# products of at least two sectors, also with named factors
const MultiSectorTuple = Tuple{Sector, Sector, Vararg{Sector}}
const ProductSectorType = Union{
    ProductSector{<:MultiSectorTuple}, NamedSector{<:NamedTuple{<:Any, <:MultiSectorTuple}},
}
const ProductSectorBlock = FusionTreeBlock{<:ProductSectorType}

firstfactor(s::ProductSector) = first(s.sectors)
otherfactors(s::ProductSector{<:Tuple{Sector, Sector}}) = last(s.sectors)
otherfactors(s::ProductSector) = ProductSector(Base.tail(s.sectors))
firstfactortype(::Type{ProductSector{T}}) where {T} = fieldtype(T, 1)
otherfactorstype(::Type{ProductSector{Tuple{H, T}}}) where {H <: Sector, T <: Sector} = T
otherfactorstype(::Type{ProductSector{T}}) where {T} = ProductSector{Base.tuple_type_tail(T)}

Base.@assume_effects :foldable function ismixedproduct(::Type{ProductSector{T}}) where {T}
    isunique = map(S -> FusionStyle(S) isa UniqueFusion, fieldtypes(T))
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

headsector(s::ProductSector) = ismixedproduct(typeof(s)) ? first(partitionfactors(s)) : firstfactor(s)
tailsector(s::ProductSector) = ismixedproduct(typeof(s)) ? last(partitionfactors(s)) : otherfactors(s)
headsectortype(::Type{I}) where {I <: ProductSector} = ismixedproduct(I) ? first(partitionfactortypes(I)) : firstfactortype(I)
tailsectortype(::Type{I}) where {I <: ProductSector} = ismixedproduct(I) ? last(partitionfactortypes(I)) : otherfactorstype(I)
joinsectors(::Type{I}, h, t) where {I <: ProductSector} = ismixedproduct(I) ? unpartition(I, h, t) : h ⊠ t

# named products are split and joined as the corresponding `ProductSector`
productsectortype(::Type{NamedSector{NamedTuple{K, T}}}) where {K, T} = ProductSector{T}
headsector(s::NamedSector) = headsector(ProductSector(s))
tailsector(s::NamedSector) = tailsector(ProductSector(s))
headsectortype(::Type{I}) where {I <: NamedSector} = headsectortype(productsectortype(I))
tailsectortype(::Type{I}) where {I <: NamedSector} = tailsectortype(productsectortype(I))
joinsectors(::Type{I}, h, t) where {I <: NamedSector} = I(joinsectors(productsectortype(I), h, t))

# the vertex labels of a product are the linear indices of the head and tail labels, with the tail
# running fastest, such that a `UniqueFusion` tail leaves the labels of the head unchanged
vertexlines(f::FusionTree) = (f.uncoupled[1], f.innerlines..., f.coupled)
vertexmultiplicities(f::FusionTree) =
    ntuple(i -> Nsymbol(vertexlines(f)[i], f.uncoupled[i + 1], vertexlines(f)[i + 1]), length(f) - 1)

maptree(proj, f::FusionTree, vertices) = FusionTree(
    map(proj, f.uncoupled), proj(f.coupled), f.isdual, map(proj, f.innerlines), vertices
)
function headtail(f::FusionTree)
    vertices_h = vertices_t = f.vertices
    if FusionStyle(tailsectortype(sectortype(f))) isa UniqueFusion
        vertices_t = map(Returns(1), f.vertices)
    elseif FusionStyle(f) isa GenericFusion && length(f) > 1
        ns = vertexmultiplicities(maptree(tailsector, f, map(Returns(1), f.vertices)))
        vertices_h = map((μ, n) -> div(μ - 1, n) + 1, f.vertices, ns)
        vertices_t = map((μ, n) -> mod(μ - 1, n) + 1, f.vertices, ns)
    end
    return maptree(headsector, f, vertices_h), maptree(tailsector, f, vertices_t)
end
function jointree(::Type{I}, h::FusionTree, t::FusionTree) where {I}
    vertices = if FusionStyle(I) isa GenericFusion && length(h) > 1 && !(FusionStyle(t) isa UniqueFusion)
        map((μ, ν, n) -> ν + (μ - 1) * n, h.vertices, t.vertices, vertexmultiplicities(t))
    else
        h.vertices
    end
    lift(a, b) = joinsectors(I, a, b)
    return fusiontreetype(I, length(h))(
        map(lift, h.uncoupled, t.uncoupled), lift(h.coupled, t.coupled), h.isdual,
        map(lift, h.innerlines, t.innerlines), vertices
    )
end

# the canonical order of a product block is that of the Kronecker product of the head and tail
# blocks, with the head running fastest
function productblock(::Type{I}, head::FusionTreeBlock, tail::FusionTreeBlock) where {I}
    trees = [(jointree(I, h₁, t₁), jointree(I, h₂, t₂)) for (h₁, h₂) in fusiontrees(head), (t₁, t₂) in fusiontrees(tail)]
    return FusionTreeBlock(vec(trees))
end

function FusionTreeBlock{I}(
        uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}},
        isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}};
        sizehint::Int = 0
    ) where {I <: ProductSectorType, N₁, N₂}
    # blocks of `UniqueFusion` sectors are enumerated directly
    FusionStyle(I) isa UniqueFusion && return fusiontreeblock(I, uncoupled, isdual; sizehint)

    # the tail holds the `UniqueFusion` factors of a mixed product, which are cheaper to construct
    tail = FusionTreeBlock{tailsectortype(I)}(map.(tailsector, uncoupled), isdual)
    isempty(tail) && return FusionTreeBlock(Vector{fusiontreetype(I, N₁, N₂)}())

    head = FusionTreeBlock{headsectortype(I)}(map.(headsector, uncoupled), isdual)
    return productblock(I, head, tail)
end

# the distinct tree pairs, and the position of every tree pair among them
function distincttrees(trees)
    distinct = unique(trees)
    return distinct, indexin(trees, distinct)
end
headtrees((f₁, f₂)) = (first(headtail(f₁)), first(headtail(f₂)))
tailtrees((f₁, f₂)) = (last(headtail(f₁)), last(headtail(f₂)))

# apply `transform` to the block `src`, where every tree pair of a `UniqueFusion` block maps to a
# single tree pair
function factor_transform(transform::F, ::Type{T}, src::FusionTreeBlock) where {F, T}
    FusionStyle(src) isa UniqueFusion || return transform(src)
    dst = map(transform, fusiontrees(src))
    return FusionTreeBlock(map(first, dst)) => LinearAlgebra.diagm(T[last(d) for d in dst])
end

# apply `transform` to the block `src` through its head and tail, with the recoupling matrix with scalar type `T`
function product_transform(transform::F, ::Type{T}, src::FusionTreeBlock{I}) where {F, T, I}
    if FusionStyle(I) isa UniqueFusion
        return factor_transform(transform, T, src)
    elseif FusionStyle(tailsectortype(I)) isa UniqueFusion && numind(src) > 0
        # the tail holds the `UniqueFusion` factors of a mixed product, with a single tree pair
        (t₁, t₂), coeff = transform(tailtrees(first(fusiontrees(src))))
        dst_h, U_h = transform(FusionTreeBlock(map(headtrees, fusiontrees(src))))
        dst = FusionTreeBlock(
            map(((h₁, h₂),) -> (jointree(I, h₁, t₁), jointree(I, h₂, t₂)), fusiontrees(dst_h))
        )
        U = Matrix{T}(undef, size(U_h))
        U .= coeff .* U_h
        return dst => U
    end
    src_h, cols_h = distincttrees(map(headtrees, fusiontrees(src)))
    src_t, cols_t = distincttrees(map(tailtrees, fusiontrees(src)))
    dst_h, U_h = factor_transform(transform, T, FusionTreeBlock(src_h))
    dst_t, U_t = factor_transform(transform, T, FusionTreeBlock(src_t))

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
