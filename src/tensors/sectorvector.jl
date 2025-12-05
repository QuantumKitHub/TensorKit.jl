struct SectorVector{T <: Number, I <: Sector, A <: DenseVector{T}} <: AbstractVector{T}
    data::A
    structure::SectorDict{I, UnitRange{Int}}
end

function SectorVector{T}(::UndefInitializer, V::ElementarySpace) where {T}
    data = Vector{T}(undef, reduceddim(V))
    structure = diagonalblockstructure(V ← V)
    return SectorVector(data, structure)
end

Base.keytype(::Type{SectorVector{T, I, A}}) where {T, I, A} = I
Base.eltype(::Type{SectorVector{T, I, A}}) where {T, I, A} = T
Base.valtype(v::SectorVector) = valtype(typeof(v))
Base.valtype(::Type{SectorVector{T, I, A}}) where {T, I, A} = SubArray{T, 1, A, Tuple{UnitRange{Int}}, true}

@inline Base.getindex(v::SectorVector{<:Any, I}, key::I) where {I} = view(v.data, v.structure[key])
@inline Base.setindex!(v::SectorVector{<:Any, I}, val, key::I) where {I} = copy!(view(v.data, v.structure[key]), val)

Base.keys(v::SectorVector) = keys(v.structure)
Base.values(v::SectorVector) = (v[c] for c in keys(v))
Base.pairs(v::SectorVector) = SectorDict(c => v[c] for c in keys(v))

Base.isempty(v::SectorVector) = isempty(v.data)
Base.copy(v::SectorVector) = SectorVector(copy(v.data), v.structure)

Base.similar(v::SectorVector) = SectorVector(similar(v.data), v.structure)
Base.similar(v::SectorVector, ::Type{T}) where {T} = SectorVector(similar(v.data, T), v.structure)
Base.similar(v::SectorVector, V::ElementarySpace) = SectorVector(undef, V)

blocksectors(v::SectorVector) = keys(v)
blocks(v::SectorVector) = SectorDict(c => v[c] for c in keys(v))
block(v::SectorVector{T, I, A}, c::I) where {T, I, A} = Base.getindex(v, c)
