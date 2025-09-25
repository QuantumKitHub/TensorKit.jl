"""
    struct CartesianSpace <: ElementarySpace

A real Euclidean space `ℝ^d`, which is therefore self-dual. `CartesianSpace` has no
additonal structure and is completely characterised by its dimension `d`. This is the
vector space that is implicitly assumed in most of matrix algebra.
"""
struct CartesianSpace <: ElementarySpace
    d::Int
end
CartesianSpace(d::Integer = 0; dual = false) = CartesianSpace(Int(d))
function CartesianSpace(dim::Pair; dual = false)
    if dim.first === Trivial()
        return CartesianSpace(dim.second; dual = dual)
    else
        msg = "$(dim) is not a valid dimension for CartesianSpace"
        throw(SectorMismatch(msg))
    end
end
function CartesianSpace(dims::AbstractDict; kwargs...)
    if length(dims) == 0
        return CartesianSpace(0; kwargs...)
    elseif length(dims) == 1
        return CartesianSpace(first(dims); kwargs...)
    else
        msg = "$(dims) is not a valid dimension dictionary for CartesianSpace"
        throw(SectorMismatch(msg))
    end
end
CartesianSpace(g::Base.Generator; kwargs...) = CartesianSpace(g...; kwargs...)

field(::Type{CartesianSpace}) = ℝ
InnerProductStyle(::Type{CartesianSpace}) = EuclideanInnerProduct()

Base.conj(V::CartesianSpace) = V
isdual(V::CartesianSpace) = false

# convenience constructor
Base.getindex(::RealNumbers) = CartesianSpace
Base.:^(::RealNumbers, d::Int) = CartesianSpace(d)

# Corresponding methods:
#------------------------
dim(V::CartesianSpace, ::Trivial = Trivial()) = V.d
Base.axes(V::CartesianSpace, ::Trivial = Trivial()) = Base.OneTo(dim(V))
hassector(V::CartesianSpace, ::Trivial) = dim(V) != 0
sectors(V::CartesianSpace) = OneOrNoneIterator(dim(V) != 0, Trivial())
sectortype(::Type{CartesianSpace}) = Trivial

unitspace(::Type{CartesianSpace}) = CartesianSpace(1)
zerospace(::Type{CartesianSpace}) = CartesianSpace(0)
⊕(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(V₁.d + V₂.d)
function ⊖(V::CartesianSpace, W::CartesianSpace)
    V ≿ W || throw(ArgumentError("$(W) is not a subspace of $(V)"))
    return CartesianSpace(dim(V) - dim(W))
end

fuse(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(V₁.d * V₂.d)
flip(V::CartesianSpace) = V

infimum(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(min(V₁.d, V₂.d))
supremum(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(max(V₁.d, V₂.d))

Base.show(io::IO, V::CartesianSpace) = print(io, "ℝ^$(V.d)")
