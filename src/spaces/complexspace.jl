"""
    struct ComplexSpace <: ElementarySpace

A standard complex vector space ℂ^d with Euclidean inner product and no additional
structure. It is completely characterised by its dimension and whether its the normal space
or its dual (which is canonically isomorphic to the conjugate space).
"""
struct ComplexSpace <: ElementarySpace
    d::Int
    dual::Bool
end
ComplexSpace(d::Integer = 0; dual = false) = ComplexSpace(Int(d), dual)
function ComplexSpace(dim::Pair; dual = false)
    if dim.first === Trivial()
        return ComplexSpace(dim.second; dual = dual)
    else
        msg = "$(dim) is not a valid dimension for ComplexSpace"
        throw(SectorMismatch(msg))
    end
end
function ComplexSpace(dims::AbstractDict; kwargs...)
    if length(dims) == 0
        return ComplexSpace(0; kwargs...)
    elseif length(dims) == 1
        return ComplexSpace(first(dims); kwargs...)
    else
        msg = "$(dims) is not a valid dimension dictionary for ComplexSpace"
        throw(SectorMismatch(msg))
    end
end
ComplexSpace(g::Base.Generator; kwargs...) = ComplexSpace(g...; kwargs...)

field(::Type{ComplexSpace}) = ℂ
InnerProductStyle(::Type{ComplexSpace}) = EuclideanInnerProduct()

# convenience constructor
Base.getindex(::ComplexNumbers) = ComplexSpace
Base.:^(::ComplexNumbers, d::Int) = ComplexSpace(d)

# Corresponding methods:
#------------------------
dim(V::ComplexSpace, s::Trivial = Trivial()) = V.d
isdual(V::ComplexSpace) = V.dual
Base.axes(V::ComplexSpace, ::Trivial = Trivial()) = Base.OneTo(dim(V))
hassector(V::ComplexSpace, ::Trivial) = dim(V) != 0
sectors(V::ComplexSpace) = OneOrNoneIterator(dim(V) != 0, Trivial())
sectortype(::Type{ComplexSpace}) = Trivial

Base.conj(V::ComplexSpace) = ComplexSpace(dim(V), !isdual(V))

unitspace(::Type{ComplexSpace}) = ComplexSpace(1)
Base.zero(::Type{ComplexSpace}) = ComplexSpace(0)

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
flip(V::ComplexSpace) = dual(V)

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

Base.show(io::IO, V::ComplexSpace) = print(io, isdual(V) ? "(ℂ^$(V.d))'" : "ℂ^$(V.d)")
