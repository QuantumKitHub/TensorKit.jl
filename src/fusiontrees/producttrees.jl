# Product sectors
# ---------------
# Fusion tree manipulations of a product sector are computed from those of two parts of the product,
# as the Kronecker product of their recoupling matrices.

# products of at least two sectors, also with names
const _MultiSectorTuple = Tuple{Sector, Sector, Vararg{Sector}}
const ProductSectorType = Union{
    ProductSector{<:_MultiSectorTuple}, NamedSector{<:NamedTuple{<:Any, <:_MultiSectorTuple}},
}
const ProductSectorBlock = FusionTreeBlock{<:ProductSectorType}

# The two parts of a product are its non-`UniqueFusion` and its `UniqueFusion` sectors if it has both,
# and otherwise its first and its remaining sectors, on which we recurse.
Base.@assume_effects :foldable function _partpositions(::Type{ProductSector{T}}) where {T}
    positions = ntuple(identity, fieldcount(T))
    isunique = map(i -> FusionStyle(fieldtype(T, i)) isa UniqueFusion, positions)
    any(isunique) && !all(isunique) &&
        return Base.filter(i -> !isunique[i], positions), Base.filter(i -> isunique[i], positions)
    return (first(positions),), Base.tail(positions)
end
# parts with a single sector are not wrapped in a `ProductSector`, to share the caches of that sector
_sectorproduct(sectors::Tuple{Sector}) = only(sectors)
_sectorproduct(sectors::Tuple) = ProductSector(sectors)
_sectortuple(s::ProductSector) = s.sectors
_sectortuple(s::Sector) = (s,)
# select the entries at compile-time positions, which keeps heterogeneous tuples inferable
@generated _getentries(t::Tuple, ::Val{P}) where {P} = Expr(:tuple, (:(t[$i]) for i in P)...)

function deconstructsector(s::ProductSector)
    positions₁, positions₂ = _partpositions(typeof(s))
    return _sectorproduct(_getentries(s.sectors, Val(positions₁))),
        _sectorproduct(_getentries(s.sectors, Val(positions₂)))
end
Base.@assume_effects :foldable function deconstructsectortype(::Type{ProductSector{T}}) where {T}
    return map(_partpositions(ProductSector{T})) do positions
        return length(positions) == 1 ? fieldtype(T, only(positions)) :
            ProductSector{Tuple{map(i -> fieldtype(T, i), positions)...}}
    end
end
function reconstructsector(::Type{ProductSector{T}}, s₁::Sector, s₂::Sector) where {T}
    sectors = (_sectortuple(s₁)..., _sectortuple(s₂)...)
    positions = TupleTools.invperm(TupleTools.vcat(_partpositions(ProductSector{T})...))
    return ProductSector{T}(_getentries(sectors, Val(positions)))
end

# named products are split and joined as the corresponding `ProductSector`
_productsectortype(::Type{NamedSector{NamedTuple{K, T}}}) where {K, T} = ProductSector{T}
deconstructsector(s::NamedSector) = deconstructsector(ProductSector(s))
deconstructsectortype(::Type{I}) where {I <: NamedSector} = deconstructsectortype(_productsectortype(I))
reconstructsector(::Type{I}, s₁::Sector, s₂::Sector) where {I <: NamedSector} =
    I(reconstructsector(_productsectortype(I), s₁, s₂))

# the vertex labels of a product are the linear indices of the labels of its parts, with the second
# part running fastest, such that a `UniqueFusion` second part leaves the labels of the first unchanged
_vertexlines(f::FusionTree) = (f.uncoupled[1], f.innerlines..., f.coupled)
_vertexmultiplicities(f::FusionTree) =
    ntuple(i -> Nsymbol(_vertexlines(f)[i], f.uncoupled[i + 1], _vertexlines(f)[i + 1]), length(f) - 1)

_maptree(proj, f::FusionTree, vertices) = FusionTree(
    map(proj, f.uncoupled), proj(f.coupled), f.isdual, map(proj, f.innerlines), vertices
)
function deconstructtree(f::FusionTree)
    vertices₁ = vertices₂ = f.vertices
    if FusionStyle(last(deconstructsectortype(sectortype(f)))) isa UniqueFusion
        vertices₂ = map(Returns(1), f.vertices)
    elseif FusionStyle(f) isa GenericFusion && length(f) > 1
        ns = _vertexmultiplicities(_maptree(last ∘ deconstructsector, f, map(Returns(1), f.vertices)))
        vertices₁ = map((μ, n) -> div(μ - 1, n) + 1, f.vertices, ns)
        vertices₂ = map((μ, n) -> mod(μ - 1, n) + 1, f.vertices, ns)
    end
    return _maptree(first ∘ deconstructsector, f, vertices₁),
        _maptree(last ∘ deconstructsector, f, vertices₂)
end
function reconstructtree(::Type{I}, f₁::FusionTree, f₂::FusionTree) where {I}
    vertices = if FusionStyle(I) isa GenericFusion && length(f₁) > 1 && !(FusionStyle(f₂) isa UniqueFusion)
        map((μ, ν, n) -> ν + (μ - 1) * n, f₁.vertices, f₂.vertices, _vertexmultiplicities(f₂))
    else
        f₁.vertices
    end
    join(a, b) = reconstructsector(I, a, b)
    return fusiontreetype(I, length(f₁))(
        map(join, f₁.uncoupled, f₂.uncoupled), join(f₁.coupled, f₂.coupled), f₁.isdual,
        map(join, f₁.innerlines, f₂.innerlines), vertices
    )
end

# the canonical order of a product block is that of the Kronecker product of the blocks of its
# parts, with the first part running fastest
function productblock(::Type{I}, block₁::FusionTreeBlock, block₂::FusionTreeBlock) where {I}
    trees = [
        (reconstructtree(I, f₁, g₁), reconstructtree(I, f₂, g₂))
            for (f₁, f₂) in fusiontrees(block₁), (g₁, g₂) in fusiontrees(block₂)
    ]
    return FusionTreeBlock(vec(trees))
end

function FusionTreeBlock{I}(
        uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}},
        isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}};
        sizehint::Int = 0
    ) where {I <: ProductSectorType, N₁, N₂}
    # blocks of `UniqueFusion` sectors are enumerated directly
    FusionStyle(I) isa UniqueFusion && return fusiontreeblock(I, uncoupled, isdual; sizehint)

    # the second part holds the `UniqueFusion` sectors of a mixed product, which are cheaper to construct
    I₁, I₂ = deconstructsectortype(I)
    block₂ = FusionTreeBlock{I₂}(map.(last ∘ deconstructsector, uncoupled), isdual)
    isempty(block₂) && return FusionTreeBlock(Vector{fusiontreetype(I, N₁, N₂)}())

    block₁ = FusionTreeBlock{I₁}(map.(first ∘ deconstructsector, uncoupled), isdual)
    return productblock(I, block₁, block₂)
end

# the distinct tree pairs, and the position of every tree pair among them
function _distincttrees(trees)
    distinct = unique(trees)
    return distinct, indexin(trees, distinct)
end
# the tree pairs of the two parts of a tree pair
_deconstructtrees((f₁, f₂)) = map(tuple, deconstructtree(f₁), deconstructtree(f₂))

# apply `transform` to the block `src`, where every tree pair of a `UniqueFusion` block maps to a
# single tree pair
function _unique_transform(transform::F, ::Type{T}, src::FusionTreeBlock) where {F, T}
    FusionStyle(src) isa UniqueFusion || return transform(src)
    dst = map(transform, fusiontrees(src))
    return FusionTreeBlock(map(first, dst)) => LinearAlgebra.diagm(T[last(d) for d in dst])
end

# apply `transform` to the block `src` through its parts, with the recoupling matrix with scalar type `T`
function product_transform(transform::F, ::Type{T}, src::FusionTreeBlock{I}) where {F, T, I}
    if FusionStyle(I) isa UniqueFusion
        return _unique_transform(transform, T, src)
    elseif FusionStyle(last(deconstructsectortype(I))) isa UniqueFusion && numind(src) > 0
        # the second part holds the `UniqueFusion` sectors of a mixed product, with a single tree pair
        (g₁, g₂), coeff = transform(last(_deconstructtrees(first(fusiontrees(src)))))
        dst₁, U₁ = transform(FusionTreeBlock(map(first ∘ _deconstructtrees, fusiontrees(src))))
        dst = FusionTreeBlock(
            map(((f₁, f₂),) -> (reconstructtree(I, f₁, g₁), reconstructtree(I, f₂, g₂)), fusiontrees(dst₁))
        )
        U = Matrix{T}(undef, size(U₁))
        U .= coeff .* U₁
        return dst => U
    end
    src₁, cols₁ = _distincttrees(map(first ∘ _deconstructtrees, fusiontrees(src)))
    src₂, cols₂ = _distincttrees(map(last ∘ _deconstructtrees, fusiontrees(src)))
    dst₁, U₁ = _unique_transform(transform, T, FusionTreeBlock(src₁))
    dst₂, U₂ = _unique_transform(transform, T, FusionTreeBlock(src₂))

    dst = productblock(I, dst₁, dst₂)
    U = Matrix{T}(undef, length(dst), length(src))
    for (col, i, j) in zip(axes(U, 2), cols₁, cols₂)
        row = 0
        for j₂ in axes(U₂, 1), j₁ in axes(U₁, 1)
            U[row += 1, col] = U₁[j₁, i] * U₂[j₂, j]
        end
    end
    return dst => U
end
