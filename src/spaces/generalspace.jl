"""
    struct GeneralSpace{𝔽} <: ElementarySpace

A finite-dimensional space over an arbitrary field `𝔽` without additional structure.
It is thus characterized by its dimension, and whether or not it is the dual and/or
conjugate space. For a real field `𝔽`, the space and its conjugate are the same.
"""
struct GeneralSpace{𝔽} <: ElementarySpace
    d::Int
    dual::Bool
    conj::Bool
    function GeneralSpace{𝔽}(d::Int, dual::Bool, conj::Bool) where {𝔽}
        d >= 0 ||
            throw(ArgumentError("Dimension of a vector space should be bigger than zero"))
        if 𝔽 isa Field
            new{𝔽}(Int(d), dual, (𝔽 ⊆ ℝ) ? false : conj)
        else
            throw(ArgumentError("Unrecognised scalar field: $𝔽"))
        end
    end
end
function GeneralSpace{𝔽}(d::Int=0; dual::Bool=false, conj::Bool=false) where {𝔽}
    return GeneralSpace{𝔽}(d, dual, conj)
end

dim(V::GeneralSpace, s::Trivial=Trivial()) = V.d
isdual(V::GeneralSpace) = V.dual
isconj(V::GeneralSpace) = V.conj

Base.axes(V::GeneralSpace, ::Trivial=Trivial()) = Base.OneTo(dim(V))
hassector(V::GeneralSpace, ::Trivial) = dim(V) != 0
sectors(V::GeneralSpace) = OneOrNoneIterator(dim(V) != 0, Trivial())
sectortype(::Type{<:GeneralSpace}) = Trivial

field(::Type{GeneralSpace{𝔽}}) where {𝔽} = 𝔽
InnerProductStyle(::Type{<:GeneralSpace}) = NoInnerProduct()

unitspace(::Type{GeneralSpace{𝔽}}) where {𝔽} = GeneralSpace{𝔽}(1, false, false)
Base.zero(::Type{GeneralSpace{𝔽}}) where {𝔽} = GeneralSpace{𝔽}(0, false, false)

dual(V::GeneralSpace{𝔽}) where {𝔽} = GeneralSpace{𝔽}(dim(V), !isdual(V), isconj(V))
Base.conj(V::GeneralSpace{𝔽}) where {𝔽} = GeneralSpace{𝔽}(dim(V), isdual(V), !isconj(V))

function Base.show(io::IO, V::GeneralSpace{𝔽}) where {𝔽}
    if isconj(V)
        print(io, "conj(")
    end
    print(io, "GeneralSpace{", 𝔽, "}(", dim(V), ")")
    if isdual(V)
        print(io, "'")
    end
    if isconj(V)
        print(io, ")")
    end
end
