const _ROCMatOrDict{I,T} = Union{ROCMatrix{T},SectorDict{I,ROCMatrix{T}}}
const ROCTensorMap{T,S,N₁,N₂,I,A<:_ROCMatOrDict{I,T}} = TensorMap{T,S,N₁,N₂,A}
const ROCTensor{T, S, N, I, A <: _ROCMatOrDict{I, T}} = ROCTensorMap{T, S, N, 0, I, A}

function ROCTensorMap{T}(::UndefInitializer, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    A = ROCMatrix{T, AMDGPU.default_memory}
    TT = tensormaptype{S, N₁, N₂, A}
    return TT(undef, codomain(V), domain(V))
end

function ROCTensorMap{T}(::UndefInitializer, codomain::TensorSpace{S},
                         domain::TensorSpace{S}) where {T,S}
    return ROCTensorMap{T}(undef, codomain ← domain)
end
function ROCTensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T,S}
    return ROCTensorMap{T}(undef, V ← one(V))
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function AMDGPU.$fname(codomain::TensorSpace{S},
                               domain::TensorSpace{S}=one(codomain)) where {S<:IndexSpace}
            return AMDGPU.$fname(codomain ← domain)
        end
        function AMDGPU.$fname(::Type{T}, codomain::TensorSpace{S},
                               domain::TensorSpace{S}=one(codomain)) where {T,S<:IndexSpace}
            return AMDGPU.$fname(T, codomain ← domain)
        end
        AMDGPU.$fname(V::TensorMapSpace) = AMDGPU.$fname(Float64, V)
        function AMDGPU.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = ROCTensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randfun in (:rand, :randn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function AMDGPU.$randfun(codomain::TensorSpace{S},
                                 domain::TensorSpace{S}) where {S<:IndexSpace}
            return AMDGPU.$randfun(codomain ← domain)
        end
        function AMDGPU.$randfun(::Type{T}, codomain::TensorSpace{S},
                                 domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return AMDGPU.$randfun(T, codomain ← domain)
        end
        function AMDGPU.$randfun(rng::Random.AbstractRNG, ::Type{T},
                                 codomain::TensorSpace{S},
                                 domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return AMDGPU.$randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        AMDGPU.$randfun(codomain::TensorSpace) = AMDGPU.$randfun(codomain ← one(codomain))
        function AMDGPU.$randfun(::Type{T}, codomain::TensorSpace) where {T}
            return AMDGPU.$randfun(T, codomain ← one(codomain))
        end
        function AMDGPU.$randfun(rng::Random.AbstractRNG, ::Type{T},
                                 codomain::TensorSpace) where {T}
            return AMDGPU.$randfun(rng, T, codomain ← one(domain))
        end

        # filling in default eltype
        AMDGPU.$randfun(V::TensorMapSpace) = AMDGPU.$randfun(Float64, V)
        function AMDGPU.$randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return AMDGPU.$randfun(rng, Float64, V)
        end

        # filling in default rng
        function AMDGPU.$randfun(::Type{T}, V::TensorMapSpace) where {T}
            return AMDGPU.$randfun(Random.default_rng(), T, V)
        end

        # implementation
        function AMDGPU.$randfun(rng::Random.AbstractRNG, ::Type{T},
                               V::TensorMapSpace) where {T}
            t = ROCTensorMap{T}(undef, V)
            AMDGPU.$randfun!(rng, t)
            return t
        end
    end
end

# converters
# ----------
function Base.convert(::Type{ROCTensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => ROCArray(b) for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => ROCArray(b)
                          for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

