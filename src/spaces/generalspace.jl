"""
    struct GeneralSpace{𝔽} <: ElementarySpace
    GeneralSpace{𝔽}(d::Integer = 0; dual::Bool = false, conj::Bool = false)

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
        return if 𝔽 isa Field
            new{𝔽}(Int(d), dual, (𝔽 ⊆ ℝ) ? false : conj)
        else
            throw(ArgumentError(lazy"Unrecognised scalar field: $𝔽"))
        end
    end
end
function GeneralSpace{𝔽}(d::Int = 0; dual::Bool = false, conj::Bool = false) where {𝔽}
    return GeneralSpace{𝔽}(d, dual, conj)
end

# Corresponding methods:
#------------------------
field(::Type{GeneralSpace{𝔽}}) where {𝔽} = 𝔽
InnerProductStyle(::Type{<:GeneralSpace}) = NoInnerProduct()

dim(V::GeneralSpace, s::Trivial = Trivial()) = V.d
Base.axes(V::GeneralSpace, ::Trivial = Trivial()) = Base.OneTo(dim(V))

dual(V::GeneralSpace{𝔽}) where {𝔽} = GeneralSpace{𝔽}(dim(V), !isdual(V), isconj(V))
Base.conj(V::GeneralSpace{𝔽}) where {𝔽} = 𝔽 == ℝ ? V : GeneralSpace{𝔽}(dim(V), isdual(V), !isconj(V))
isdual(V::GeneralSpace) = V.dual
isconj(V::GeneralSpace{𝔽}) where {𝔽} = 𝔽 == ℝ ? false : V.conj

unitspace(::Type{GeneralSpace{𝔽}}) where {𝔽} = GeneralSpace{𝔽}(1, false, false)
zerospace(::Type{GeneralSpace{𝔽}}) where {𝔽} = GeneralSpace{𝔽}(0, false, false)

hassector(V::GeneralSpace, ::Trivial) = dim(V) != 0
sectors(V::GeneralSpace) = OneOrNoneIterator(dim(V) != 0, Trivial())
sectortype(::Type{<:GeneralSpace}) = Trivial

function Base.show(io::IO, V::GeneralSpace{𝔽}) where {𝔽}
    isconj(V) && print(io, "conj(")
    print(io, "GeneralSpace{", 𝔽, "}(", dim(V), ")")
    isdual(V) && print(io, "'")
    isconj(V) && print(io, ")")
    return nothing
end
