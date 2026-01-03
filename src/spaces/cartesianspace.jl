"""
    struct CartesianSpace <: ElementarySpace
    CartesianSpace(d::Integer = 0; dual = false)
    ℝ^d

A real Euclidean space ``ℝ^d``. `CartesianSpace` has no additonal structure and
is completely characterised by its dimension `d`. A `dual` keyword argument is
accepted for compatibility with other space constructors, but is ignored
since the dual of a Cartesian space is isomorphic to itself. This is the
vector space that is implicitly assumed in most of matrix algebra.
"""
struct CartesianSpace <: ElementarySpace
    d::Int

    # required to avoid CartesianSpace(::Any) default constructor:
    CartesianSpace(d::Int) = new(d)
end

CartesianSpace(d::Integer = 0; dual = false) = CartesianSpace(Int(d))
CartesianSpace(dim::Pair; kwargs...) = CartesianSpace((dim,); kwargs...)
function CartesianSpace(dims; dual::Bool = false)
    # using manual iteration here to avoid depending on `length` while still checking it is
    # 0 ≤ length ≤ 1
    next = Base.iterate(dims)
    isnothing(next) && return CartesianSpace(0)

    (c, d), state = next
    convert(Trivial, c) === Trivial() ||
        throw(SectorMismatch(lazy"$c is not a valid charge for CartesianSpace"))

    V = CartesianSpace(d)

    next = Base.iterate(dims, state)
    isnothing(next) ||
        throw(SectorMismatch(lazy"$dims is not a valid dimension iterable for CartesianSpace"))

    return V
end

# convenience constructor
Base.getindex(::RealNumbers) = CartesianSpace
Base.:^(::RealNumbers, d::Int) = CartesianSpace(d)

# Corresponding methods:
#------------------------
field(::Type{CartesianSpace}) = ℝ
InnerProductStyle(::Type{CartesianSpace}) = EuclideanInnerProduct()

dim(V::CartesianSpace, ::Trivial = Trivial()) = V.d
Base.axes(V::CartesianSpace, ::Trivial = Trivial()) = Base.OneTo(dim(V))

dual(V::CartesianSpace) = V
Base.conj(V::CartesianSpace) = dual(V)
isdual(V::CartesianSpace) = false
isconj(V::CartesianSpace) = false
flip(V::CartesianSpace) = V

unitspace(::Type{CartesianSpace}) = CartesianSpace(1)
zerospace(::Type{CartesianSpace}) = CartesianSpace(0)
⊕(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(V₁.d + V₂.d)
function ⊖(V::CartesianSpace, W::CartesianSpace)
    V ≿ W || throw(ArgumentError("$(W) is not a subspace of $(V)"))
    return CartesianSpace(dim(V) - dim(W))
end

fuse(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(V₁.d * V₂.d)

infimum(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(min(V₁.d, V₂.d))
supremum(V₁::CartesianSpace, V₂::CartesianSpace) = CartesianSpace(max(V₁.d, V₂.d))

hassector(V::CartesianSpace, ::Trivial) = dim(V) != 0
sectors(V::CartesianSpace) = OneOrNoneIterator(dim(V) != 0, Trivial())
sectortype(::Type{CartesianSpace}) = Trivial

Base.show(io::IO, V::CartesianSpace) = print(io, "ℝ^$(V.d)")
