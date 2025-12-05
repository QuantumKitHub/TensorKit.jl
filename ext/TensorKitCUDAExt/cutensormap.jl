const CuTensorMap{T, S, N₁, N₂} = TensorMap{T, S, N₁, N₂, CuVector{T, CUDA.DeviceMemory}}
const CuTensor{T, S, N} = CuTensorMap{T, S, N, 0}

const AdjointCuTensorMap{T, S, N₁, N₂} = AdjointTensorMap{T, S, N₁, N₂, CuTensorMap{T, S, N₁, N₂}}

function TensorKit.tensormaptype(S::Type{<:IndexSpace}, N₁, N₂, TorA::Type{<:StridedCuArray})
    if TorA <: CuArray
        return TensorMap{eltype(TorA), S, N₁, N₂, CuVector{eltype(TorA), CUDA.DeviceMemory}}
    else
        throw(ArgumentError("argument $TorA should specify a scalar type (`<:Number`) or a storage type `<:CuVector{<:Number}`"))
    end
end

function CuTensorMap{T}(::UndefInitializer, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    return CuTensorMap{T, S, N₁, N₂}(undef, V)
end

function CuTensorMap{T}(
        ::UndefInitializer, codomain::TensorSpace{S},
        domain::TensorSpace{S}
    ) where {T, S}
    return CuTensorMap{T}(undef, codomain ← domain)
end
function CuTensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T, S}
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
function CuTensorMap(
        data::AbstractDict{<:Sector, <:CuArray},
        V::TensorMapSpace{S, N₁, N₂}
    ) where {S, N₁, N₂}
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
function CuTensorMap{T}(
        data::DenseVector{T}, codomain::TensorSpace{S},
        domain::TensorSpace{S}
    ) where {T, S}
    return CuTensorMap(data, codomain ← domain)
end
function CuTensorMap(
        data::AbstractDict{<:Sector, <:CuMatrix}, codom::TensorSpace{S},
        dom::TensorSpace{S}
    ) where {S}
    return CuTensorMap(data, codom ← dom)
end
function CuTensorMap(data::DenseVector{T}, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    return CuTensorMap{T, S, N₁, N₂}(data, V)
end
function CuTensorMap(data::CuArray{T}, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    return CuTensorMap{T, S, N₁, N₂}(vec(data), V)
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function CUDA.$fname(
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {S <: IndexSpace}
            return CUDA.$fname(codomain ← domain)
        end
        function CUDA.$fname(
                ::Type{T}, codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {T, S <: IndexSpace}
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

for randfun in (:curand, :curandn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain),
            ) where {S <: IndexSpace}
            return $randfun(codomain ← domain)
        end
        function $randfun(
                ::Type{T}, codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain),
            ) where {T, S <: IndexSpace}
            return $randfun(T, codomain ← domain)
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain),
            ) where {T, S <: IndexSpace}
            return $randfun(rng, T, codomain ← domain)
        end

        # filling in default eltype
        $randfun(V::TensorMapSpace) = $randfun(Float64, V)
        function $randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return $randfun(rng, Float64, V)
        end

        # filling in default rng
        function $randfun(::Type{T}, V::TensorMapSpace) where {T}
            return $randfun(Random.default_rng(), T, V)
        end

        # implementation
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                V::TensorMapSpace
            ) where {T}
            t = CuTensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end
    end
end

for randfun in (:rand, :randn, :randisometry)
    randfun! = Symbol(randfun, :!)
    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(
                ::Type{A}, codomain::TensorSpace{S},
                domain::TensorSpace{S}
            ) where {A <: CuArray, S <: IndexSpace}
            return $randfun(A, codomain ← domain)
        end
        function $randfun(
                ::Type{T}, ::Type{A}, codomain::TensorSpace{S},
                domain::TensorSpace{S}
            ) where {T, S <: IndexSpace, A <: CuArray{T}}
            return $randfun(T, A, codomain ← domain)
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T}, ::Type{A},
                codomain::TensorSpace{S},
                domain::TensorSpace{S}
            ) where {T, S <: IndexSpace, A <: CuArray{T}}
            return $randfun(rng, T, A, codomain ← domain)
        end

        # accepting single `TensorSpace`
        $randfun(::Type{A}, codomain::TensorSpace) where {A <: CuArray} = $randfun(A, codomain ← one(codomain))
        function $randfun(::Type{T}, ::Type{A}, codomain::TensorSpace) where {T, A <: CuArray{T}}
            return $randfun(T, A, codomain ← one(codomain))
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                ::Type{A}, codomain::TensorSpace
            ) where {T, A <: CuArray{T}}
            return $randfun(rng, T, A, codomain ← one(domain))
        end

        # filling in default eltype
        $randfun(::Type{A}, V::TensorMapSpace) where {A <: CuArray} = $randfun(eltype(A), A, V)
        function $randfun(rng::Random.AbstractRNG, ::Type{A}, V::TensorMapSpace) where {A <: CuArray}
            return $randfun(rng, eltype(A), A, V)
        end

        # filling in default rng
        function $randfun(::Type{T}, ::Type{A}, V::TensorMapSpace) where {T, A <: CuArray{T}}
            return $randfun(Random.default_rng(), T, A, V)
        end

        # implementation
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T},
                ::Type{A}, V::TensorMapSpace
            ) where {T, A <: CuArray{T}}
            t = CuTensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end
    end
end

# converters
# ----------
function Base.convert(::Type{CuTensorMap}, d::Dict{Symbol, Any})
    try
        codomain = eval(TensorKit, Meta.parse(d[:codomain]))
        domain = eval(TensorKit, Meta.parse(d[:domain]))
        data = SectorDict(eval(TensorKit, Meta.parse(c)) => CuArray(b) for (c, b) in d[:data])
        return CuTensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(
            Base.eval(Main, Meta.parse(c)) => CuArray(b)
                for (c, b) in d[:data]
        )
        return CuTensorMap(data, codomain, domain)
    end
end

function Base.convert(::Type{CuTensorMap}, t::AbstractTensorMap)
    return copy!(CuTensorMap{scalartype(t)}(undef, space(t)), t)
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::CuTensorMap)

    # TODO: should scalar only work if N₁ == N₂ == 0?
    return @allowscalar dim(codomain(t)) == dim(domain(t)) == 1 ?
        first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end

TensorKit.scalartype(A::StridedCuArray{T}) where {T} = T
TensorKit.scalartype(::Type{<:CuTensorMap{T}}) where {T} = T
TensorKit.scalartype(::Type{<:CuArray{T}}) where {T} = T

function TensorKit.similarstoragetype(TT::Type{<:CuTensorMap{TTT, S, N₁, N₂}}, ::Type{T}) where {TTT, T, S, N₁, N₂}
    return CuVector{T, CUDA.DeviceMemory}
end

function Base.convert(
        TT::Type{CuTensorMap{T, S, N₁, N₂}},
        t::AbstractTensorMap{<:Any, S, N₁, N₂}
    ) where {T, S, N₁, N₂}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function Base.copy!(tdst::CuTensorMap{T, S, N₁, N₂}, tsrc::CuTensorMap{T, S, N₁, N₂}) where {T, S, N₁, N₂}
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for ((c, bdst), (_, bsrc)) in zip(blocks(tdst), blocks(tsrc))
        copy!(bdst, bsrc)
    end
    return tdst
end

function Base.copy!(tdst::CuTensorMap, tsrc::TensorKit.AdjointTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for ((c, bdst), (_, bsrc)) in zip(blocks(tdst), blocks(tsrc))
        copy!(bdst, bsrc)
    end
    return tdst
end

function Base.promote_rule(
        ::Type{<:TT₁},
        ::Type{<:TT₂}
    ) where {
        S, N₁, N₂, TTT₁, TTT₂,
        TT₁ <: CuTensorMap{TTT₁, S, N₁, N₂},
        TT₂ <: CuTensorMap{TTT₂, S, N₁, N₂},
    }
    T = TensorKit.VectorInterface.promote_add(TTT₁, TTT₂)
    return CuTensorMap{T, S, N₁, N₂}
end

function LinearAlgebra.isposdef(t::CuTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanInnerProduct() || return false
    for (c, b) in blocks(t)
        # do our own hermitian check
        isherm = TensorKit.MatrixAlgebraKit.ishermitian(b; atol = eps(real(eltype(b))), rtol = eps(real(eltype(b))))
        isherm || return false
        isposdef(Hermitian(b)) || return false
    end
    return true
end


# Conversion to CuArray:
#----------------------
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{CuArray}, t::AbstractTensorMap)
    I = sectortype(t)
    if I === Trivial
        convert(CuArray, t[])
    else
        cod = codomain(t)
        dom = domain(t)
        T = sectorscalartype(I) <: Complex ? complex(scalartype(t)) :
            sectorscalartype(I) <: Integer ? scalartype(t) : float(scalartype(t))
        A = CUDA.zeros(T, dims(cod)..., dims(dom)...)
        for (f₁, f₂) in fusiontrees(t)
            F = convert(CuArray, (f₁, f₂))
            Aslice = StridedView(A)[axes(cod, f₁.uncoupled)..., axes(dom, f₂.uncoupled)...]
            add!(Aslice, StridedView(TensorKit._kron(convert(CuArray, t[f₁, f₂]), F)))
        end
        return A
    end
end
