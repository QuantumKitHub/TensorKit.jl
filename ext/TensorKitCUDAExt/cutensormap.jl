const CuTensorMap{T,S,N₁,N₂,A<:CuVector{T}} = TensorMap{T,S,N₁,N₂,A}
const CuTensor{T, S, N, A<:CuVector{T}} = CuTensorMap{T, S, N, 0, A}

function TensorKit.tensormaptype(S::Type{<:IndexSpace}, N₁, N₂, TorA::Type{<:StridedCuArray})
    if TorA <: CuArray
        return TensorMap{eltype(TorA),S,N₁,N₂,TorA}
    else
        throw(ArgumentError("argument $TorA should specify a scalar type (`<:Number`) or a storage type `<:CuVector{<:Number}`"))
    end
end

function CuTensorMap{T}(::UndefInitializer, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    return CuTensorMap{T,S,N₁,N₂,CuVector{T}}(undef, V)
end

function CuTensorMap{T}(::UndefInitializer, codomain::TensorSpace{S},
                        domain::TensorSpace{S}) where {T,S}
    return CuTensorMap{T}(undef, codomain ← domain)
end
function CuTensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T,S}
    return CuTensorMap{T}(undef, V ← one(V))
end
# constructor starting from block data
"""
    CuTensorMap(data::AbstractDict{<:Sector,<:CuMatrix}, codomain::ProductSpace{S,N₁},
                domain::ProductSpace{S,N₂}) where {S<:ElementarySpace,N₁,N₂}
    CuTensorMap(data, codomain ← domain)
    CuTensorMap(data, domain → codomain)

Construct a `CuTensorMap` by explicitly specifying its block data.

## Arguments
- `data::AbstractDict{<:Sector,<:CuMatrix}`: dictionary containing the block data for
  each coupled sector `c` as a matrix of size `(blockdim(codomain, c), blockdim(domain, c))`.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.

Alternatively, the domain and codomain can be specified by passing a [`HomSpace`](@ref)
using the syntax `codomain ← domain` or `domain → codomain`.
"""
function CuTensorMap(data::AbstractDict{<:Sector,<:CuArray},
                     V::TensorMapSpace{S,N₁,N₂}) where {S,N₁,N₂}
    T = eltype(valtype(data))
    t = CuTensorMap{T}(undef, V)
    for (c, b) in blocks(t)
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
        datac = data[c]
        size(datac) == size(b) ||
            throw(DimensionMismatch("wrong size of block for sector $c"))
        copy!(b, datac)
    end
    for (c, b) in data
        c ∈ blocksectors(t) || isempty(b) ||
            throw(SectorMismatch("data for block sector $c not expected"))
    end
    return t
end
function CuTensorMap{T}(data::DenseVector{T}, codomain::TensorSpace{S},
                        domain::TensorSpace{S}) where {T,S}
    return CuTensorMap(data, codomain ← domain)
end
function CuTensorMap(data::AbstractDict{<:Sector,<:CuMatrix}, codom::TensorSpace{S},
                   dom::TensorSpace{S}) where {S}
    return CuTensorMap(data, codom ← dom)
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function CUDA.$fname(codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {S<:IndexSpace}
            return CUDA.$fname(codomain ← domain)
        end
        function CUDA.$fname(::Type{T}, codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {T,S<:IndexSpace}
            return CUDA.$fname(T, codomain ← domain)
        end
        CUDA.$fname(V::TensorMapSpace) = CUDA.$fname(Float64, V)
        function CUDA.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = CuTensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:rand, :randn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function CUDA.$randfun(codomain::TensorSpace{S},
                               domain::TensorSpace{S}) where {S<:IndexSpace}
            return CUDA.$randfun(codomain ← domain)
        end
        function CUDA.$randfun(::Type{T}, codomain::TensorSpace{S},
                               domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return CUDA.$randfun(T, codomain ← domain)
        end
        function CUDA.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               codomain::TensorSpace{S},
                               domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return CUDA.$randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        CUDA.$randfun(codomain::TensorSpace) = CUDA.$randfun(codomain ← one(codomain))
        function CUDA.$randfun(::Type{T}, codomain::TensorSpace) where {T}
            return CUDA.$randfun(T, codomain ← one(codomain))
        end
        function CUDA.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               codomain::TensorSpace) where {T}
            return CUDA.$randfun(rng, T, codomain ← one(domain))
        end

        # filling in default eltype
        CUDA.$randfun(V::TensorMapSpace) = CUDA.$randfun(Float64, V)
        function CUDA.$randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return CUDA.$randfun(rng, Float64, V)
        end

        # filling in default rng
        function CUDA.$randfun(::Type{T}, V::TensorMapSpace) where {T}
            return CUDA.$randfun(Random.default_rng(), T, V)
        end

        # implementation
        function CUDA.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               V::TensorMapSpace) where {T}
            t = CuTensorMap{T}(undef, V)
            CUDA.$randfun!(rng, t)
            return t
        end
    end
end

# converters
# ----------
function Base.convert(::Type{CuTensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => CuArray(b) for (c, b) in d[:data])
        return CuTensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => CuArray(b)
                          for (c, b) in d[:data])
        return CuTensorMap(data, codomain, domain)
    end
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::CuTensorMap)
    
    # TODO: should scalar only work if N₁ == N₂ == 0?
    return @allowscalar dim(codomain(t)) == dim(domain(t)) == 1 ?
           first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end

function TensorKit.similarstoragetype(TT::Type{<:CuTensorMap}, ::Type{T}) where {T}
    return CuVector{T}
end
