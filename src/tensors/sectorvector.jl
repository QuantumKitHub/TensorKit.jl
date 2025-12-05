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
