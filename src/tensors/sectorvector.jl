"""
    struct SectorVector{T, I, A <: AbstractVector{T}} <: AbstractVector{T}

A representation of a vector with values of type `T`, with certain regions labeled by keys of type `I`.
These objects behave as their underlying parent vectors of type `A`, but additionally can be indexed through
keys of type `I` to produce the appropriate views.
Intuitively, these objects can be thought of as the combination of an `AbstractVector` and an `AbstractDict`.
"""
struct SectorVector{T, I, A <: AbstractVector{T}} <: AbstractVector{T}
    data::A
    structure::SectorDict{I, UnitRange{Int}}
end

function SectorVector{T}(::UndefInitializer, V::ElementarySpace) where {T}
    data = Vector{T}(undef, reduceddim(V))
    structure = diagonalblockstructure(V ← V)
    return SectorVector(data, structure)
end
function SectorVector{T, I, A}(::UndefInitializer, V::ElementarySpace) where {T, I, A <: AbstractVector{T}}
    data = A(undef, reduceddim(V))
    structure = diagonalblockstructure(V ← V)
    return SectorVector{T, I, A}(data, structure)
end

Base.parent(v::SectorVector) = v.data

# AbstractVector interface
# ------------------------
Base.eltype(::Type{SectorVector{T, I, A}}) where {T, I, A} = T
Base.IndexStyle(::Type{SectorVector{T, I, A}}) where {T, I, A} = Base.IndexLinear()

@inline Base.getindex(v::SectorVector, i::Int) = getindex(parent(v), i)
@inline Base.setindex!(v::SectorVector, val, i::Int) = setindex!(parent(v), val, i)

Base.size(v::SectorVector, args...) = size(parent(v), args...)

Base.similar(v::SectorVector) = SectorVector(similar(v.data), v.structure)
Base.similar(v::SectorVector, ::Type{T}) where {T} = SectorVector(similar(v.data, T), v.structure)

Base.copy(v::SectorVector) = SectorVector(copy(v.data), v.structure)

# AbstractDict interface
# ----------------------
Base.keytype(v::SectorVector) = keytype(typeof(v))
Base.keytype(::Type{SectorVector{T, I, A}}) where {T, I, A} = I
Base.valtype(v::SectorVector) = valtype(typeof(v))
Base.valtype(::Type{SectorVector{T, I, A}}) where {T, I, A} = SubArray{T, 1, A, Tuple{UnitRange{Int}}, true}

@inline Base.getindex(v::SectorVector{<:Any, I}, key::I) where {I} = view(v.data, v.structure[key])
@inline Base.setindex!(v::SectorVector{<:Any, I}, val, key::I) where {I} = copy!(view(v.data, v.structure[key]), val)

Base.keys(v::SectorVector) = keys(v.structure)
Base.values(v::SectorVector) = (v[c] for c in keys(v))
Base.pairs(v::SectorVector) = SectorDict(c => v[c] for c in keys(v))

# TensorKit interface
# -------------------
sectortype(::Type{T}) where {T <: SectorVector} = keytype(T)

Base.similar(v::SectorVector, V::ElementarySpace) = SectorVector(undef, V)

blocksectors(v::SectorVector) = keys(v)
blocks(v::SectorVector) = pairs(v)
block(v::SectorVector{T, I, A}, c::I) where {T, I, A} = Base.getindex(v, c)

# VectorInterface and LinearAlgebra interface
# ----------------------------------------------
VectorInterface.zerovector(v::SectorVector, ::Type{S}) where {S <: Number} =
    SectorVector(zerovector(parent(v), S), v.structure)
VectorInterface.zerovector!(v::SectorVector) = (zerovector!(parent(v)); return v)
VectorInterface.zerovector!!(v::SectorVector) = (zerovector!!(parent(v)); return v)

VectorInterface.scale(v::SectorVector, α::Number) = SectorVector(scale(parent(v), α), v.structure)
VectorInterface.scale!(v::SectorVector, α::Number) = (scale!(parent(v), α); return v)
VectorInterface.scale!!(v::SectorVector, α::Number) = (scale!!(parent(v), α); return v)

function VectorInterface.add(v1::SectorVector, v2::SectorVector, α::Number, β::Number)
    return SectorVector(add(parent(v1), parent(v2), α::Number, β::Number), v1.structure)
end
function VectorInterface.add!(v1::SectorVector, v2::SectorVector, α::Number, β::Number)
    add!(parent(v1), parent(v2), α, β)
    return v1
end
function VectorInterface.add!!(v1::SectorVector, v2::SectorVector, α::Number, β::Number)
    add!!(parent(v1), parent(v2), α, β)
    return v1
end

function VectorInterface.inner(v1::SectorVector, v2::SectorVector)
    v1.structure == v2.structure || throw(SpaceMismatch("Sector structures do not match"))
    I = sectortype(v1)
    if FusionStyle(I) isa UniqueFusion # all quantum dimensions are one
        return inner(parent(v1), parent(v2))
    else
        T = VectorInterface.promote_inner(v1, v2)
        s = zero(T)
        for c in blocksectors(v1)
            b1 = block(v1, c)
            b2 = block(v2, c)
            s += convert(T, dim(c)) * inner(b1, b2)
        end
    end
    return s
end

LinearAlgebra.dot(v1::SectorVector, v2::SectorVector) = inner(v1, v2)

function LinearAlgebra.norm(v::SectorVector, p::Real = 2)
    return _norm(blocks(v), p, float(zero(real(scalartype(v)))))
end

# Common functionality
# --------------------
# specific overloads for performance and/or GPU
Base.minimum(x::SectorVector) = minimum(parent(x))
Base.minimum(f, x::SectorVector) = minimum(f, parent(x))
Base.maximum(x::SectorVector) = maximum(parent(x))
Base.maximum(f, x::SectorVector) = maximum(f, parent(x))
