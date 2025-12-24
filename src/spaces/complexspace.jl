"""
    struct ComplexSpace <: ElementarySpace
    ComplexSpace(d::Integer = 0; dual = false)
    ℂ^d

A standard complex vector space ``ℂ^d`` with Euclidean inner product and no additional
structure. It is completely characterised by its dimension and whether its the normal space
or its dual (which is canonically isomorphic to the conjugate space).
"""
struct ComplexSpace <: ElementarySpace
    d::Int
    dual::Bool
end

ComplexSpace(d::Integer = 0; dual::Bool = false) = ComplexSpace(Int(d), dual)
ComplexSpace(dim::Pair; kwargs...) = ComplexSpace((dim,); kwargs...)
function ComplexSpace(dims; dual::Bool = false)
    # using manual iteration here to avoid depending on `length` while still checking it is
    # 0 ≤ length ≤ 1
    next = Base.iterate(dims)
    isnothing(next) && return ComplexSpace(0, dual)

    (c, d), state = next
    convert(Trivial, c) === Trivial() ||
        throw(SectorMismatch(lazy"$c is not a valid charge for ComplexSpace"))

    V = ComplexSpace(d, dual)

    next = Base.iterate(dims, state)
    isnothing(next) ||
        throw(SectorMismatch(lazy"$dims is not a valid dimension iterable for ComplexSpace"))

    return V
end

# convenience constructor
Base.getindex(::ComplexNumbers) = ComplexSpace
Base.:^(::ComplexNumbers, d::Int) = ComplexSpace(d)

# Corresponding methods:
#------------------------
field(::Type{ComplexSpace}) = ℂ
InnerProductStyle(::Type{ComplexSpace}) = EuclideanInnerProduct()

dim(V::ComplexSpace, s::Trivial = Trivial()) = V.d
Base.axes(V::ComplexSpace, ::Trivial = Trivial()) = Base.OneTo(dim(V))

dual(V::ComplexSpace) = ComplexSpace(dim(V), !isdual(V))
Base.conj(V::ComplexSpace) = dual(V)
isdual(V::ComplexSpace) = V.dual
isconj(V::ComplexSpace) = true
flip(V::ComplexSpace) = dual(V)

unitspace(::Type{ComplexSpace}) = ComplexSpace(1)
zerospace(::Type{ComplexSpace}) = ComplexSpace(0)
function ⊕(V₁::ComplexSpace, V₂::ComplexSpace)
    return isdual(V₁) == isdual(V₂) ?
        ComplexSpace(dim(V₁) + dim(V₂), isdual(V₁)) :
        throw(SpaceMismatch("Direct sum of a vector space and its dual does not exist"))
end
function ⊖(V::ComplexSpace, W::ComplexSpace)
    (V ≿ W && isdual(V) == isdual(W)) ||
        throw(ArgumentError("$(W) is not a subspace of $(V)"))
    return ComplexSpace(dim(V) - dim(W), isdual(V))
end

fuse(V₁::ComplexSpace, V₂::ComplexSpace) = ComplexSpace(V₁.d * V₂.d)

function infimum(V₁::ComplexSpace, V₂::ComplexSpace)
    return isdual(V₁) == isdual(V₂) ?
        ComplexSpace(min(dim(V₁), dim(V₂)), isdual(V₁)) :
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
end
function supremum(V₁::ComplexSpace, V₂::ComplexSpace)
    return isdual(V₁) == isdual(V₂) ?
        ComplexSpace(max(dim(V₁), dim(V₂)), isdual(V₁)) :
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
end

hassector(V::ComplexSpace, ::Trivial) = dim(V) != 0
sectors(V::ComplexSpace) = OneOrNoneIterator(dim(V) != 0, Trivial())
sectortype(::Type{ComplexSpace}) = Trivial

Base.show(io::IO, V::ComplexSpace) = print(io, isdual(V) ? "(ℂ^$(V.d))'" : "ℂ^$(V.d)")
