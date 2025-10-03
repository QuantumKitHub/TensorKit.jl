"""
    ⊠(V₁::VectorSpace, V₂::VectorSpace)

Given two vector spaces `V₁` and `V₂` (`ElementarySpace` or `ProductSpace`), or thus,
objects of corresponding fusion categories ``C₁`` and ``C₂``, ``V₁ ⊠ V₂`` constructs the
Deligne tensor product, an object in ``C₁ ⊠ C₂`` which is the natural tensor product of
those categories. In particular, the corresponding type of sectors (simple objects) is
given by `sectortype(V₁ ⊠ V₂) == sectortype(V₁) ⊠ sectortype(V₂)` and can be thought of as
a tuple of the individual sectors.

The Deligne tensor product also works in the type domain and for sectors and tensors. For
group representations, we have `Rep[G₁] ⊠ Rep[G₂] == Rep[G₁ × G₂]`, i.e. these are the
natural representation spaces of the direct product of two groups.
"""
function ⊠(V₁::VectorSpace, V₂::VectorSpace)
    field(V₁) == field(V₂) || throw_incompatible_fields(V₁, V₂)
    return (V₁ ⊠ one(V₂)) ⊗ (one(V₁) ⊠ V₂)
end

# define deligne products with empty tensor product: just add a trivial sector of the type
# of the empty space to each of the sectors in the non-empty space
function ⊠(V::GradedSpace, P₀::ProductSpace{<:ElementarySpace, 0})
    field(V) == field(P₀) || throw_incompatible_fields(V, P₀)
    I₁ = sectortype(V)
    I₂ = sectortype(P₀)
    return Vect[I₁ ⊠ I₂](
        ifelse(isdual(V), dual(c), c) ⊠ one(I₂) => dim(V, c)
            for c in sectors(V); dual = isdual(V)
    )
end

function ⊠(P₀::ProductSpace{<:ElementarySpace, 0}, V::GradedSpace)
    field(P₀) == field(V) || throw_incompatible_fields(P₀, V)
    I₁ = sectortype(P₀)
    I₂ = sectortype(V)
    return Vect[I₁ ⊠ I₂](
        one(I₁) ⊠ ifelse(isdual(V), dual(c), c) => dim(V, c)
            for c in sectors(V); dual = isdual(V)
    )
end

function ⊠(V::ComplexSpace, P₀::ProductSpace{<:ElementarySpace, 0})
    field(V) == field(P₀) || throw_incompatible_fields(V, P₀)
    I₂ = sectortype(P₀)
    return Vect[I₂](one(I₂) => dim(V); dual = isdual(V))
end

function ⊠(P₀::ProductSpace{<:ElementarySpace, 0}, V::ComplexSpace)
    field(P₀) == field(V) || throw_incompatible_fields(P₀, V)
    I₁ = sectortype(P₀)
    return Vect[I₁](one(I₁) => dim(V); dual = isdual(V))
end

function ⊠(P::ProductSpace{<:ElementarySpace, 0}, P₀::ProductSpace{<:ElementarySpace, 0})
    field(P) == field(P₀) || throw_incompatible_fields(P, P₀)
    I₁ = sectortype(P)
    I₂ = sectortype(P₀)
    return one(Vect[I₁ ⊠ I₂])
end

function ⊠(P::ProductSpace{<:ElementarySpace}, P₀::ProductSpace{<:ElementarySpace, 0})
    field(P) == field(P₀) || throw_incompatible_fields(P, P₀)
    I₁ = sectortype(P)
    I₂ = sectortype(P₀)
    S = Vect[I₁ ⊠ I₂]
    N = length(P)
    return ProductSpace{S, N}(map(V -> V ⊠ P₀, tuple(P...)))
end

function ⊠(P₀::ProductSpace{<:ElementarySpace, 0}, P::ProductSpace{<:ElementarySpace})
    field(P₀) == field(P) || throw_incompatible_fields(P₀, P)
    I₁ = sectortype(P₀)
    I₂ = sectortype(P)
    S = Vect[I₁ ⊠ I₂]
    N = length(P)
    return ProductSpace{S, N}(map(V -> P₀ ⊠ V, tuple(P...)))
end

@noinline function throw_incompatible_fields(P₁, P₂)
    throw(ArgumentError("Deligne products require spaces over the same field: $(field(P₁)) ≠ $(field(P₂))"))
end
