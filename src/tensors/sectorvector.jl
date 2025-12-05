struct SectorVector{T <: Number, I <: Sector, A <: DenseVector{T}}
    data::A
    structure::SectorDict{I, UnitRange{Int}}
end

Base.keytype(::Type{SectorVector{T, I, A}}) where {T, I, A} = I
Base.eltype(::Type{SectorVector{T, I, A}}) where {T, I, A} = T
Base.valtype(::Type{SectorVector{T, I, A}}) where {T, I, A} = SubArray{T, 1, A, Tuple{UnitRange{Int}}, true}

@inline Base.getindex(v::SectorVector{T, I}, key::I) where {I} = view(v.data, v.structure[key])
@inline Base.setindex!(v::SectorVector{T, I}, v, key::I) where {I} = copy!(view(v.data, v.structure[key]), v)

Base.keys(v::SectorVector) = keys(v.structure)
Base.values(v::SectorVector) = SectorDict(c => v[c] for c in keys(v))

function Base.iterate(v::SectorVector, args...)
    next = iterate(keys(v), args...)
    isnothing(next) && return next
    item, state = next
    return v[item], state
end
