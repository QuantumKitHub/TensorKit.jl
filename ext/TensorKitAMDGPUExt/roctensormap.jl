const ROCTensorMap{T, S, N₁, N₂} = TensorMap{T, S, N₁, N₂, ROCVector{T, AMDGPU.Mem.HIPBuffer}}
const ROCTensor{T, S, N} = ROCTensorMap{T, S, N, 0}

const AdjointROCTensorMap{T, S, N₁, N₂} = AdjointTensorMap{T, S, N₁, N₂, ROCTensorMap{T, S, N₁, N₂}}

function TensorKit.tensormaptype(S::Type{<:IndexSpace}, N₁, N₂, TorA::Type{<:StridedROCArray})
    if TorA <: ROCArray
        return TensorMap{eltype(TorA), S, N₁, N₂, ROCVector{eltype(TorA), AMDGPU.Mem.HIPBuffer}}
    else
        throw(ArgumentError("argument $TorA should specify a scalar type (`<:Number`) or a storage type `<:ROCVector{<:Number}`"))
    end
end

function ROCTensorMap{T}(::UndefInitializer, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    return ROCTensorMap{T, S, N₁, N₂}(undef, V)
end

function ROCTensorMap{T}(
        ::UndefInitializer, codomain::TensorSpace{S},
        domain::TensorSpace{S}
    ) where {T, S}
    return ROCTensorMap{T}(undef, codomain ← domain)
end
function ROCTensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T, S}
    return ROCTensorMap{T}(undef, V ← one(V))
end
# constructor starting from block data
"""
    ROCTensorMap(data::AbstractDict{<:Sector,<:ROCMatrix}, codomain::ProductSpace{S,N₁},
                domain::ProductSpace{S,N₂}) where {S<:ElementarySpace,N₁,N₂}
    ROCTensorMap(data, codomain ← domain)
    ROCTensorMap(data, domain → codomain)

Construct a `ROCTensorMap` by explicitly specifying its block data.

## Arguments
- `data::AbstractDict{<:Sector,<:ROCMatrix}`: dictionary containing the block data for
  each coupled sector `c` as a matrix of size `(blockdim(codomain, c), blockdim(domain, c))`.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.

Alternatively, the domain and codomain can be specified by passing a [`HomSpace`](@ref)
using the syntax `codomain ← domain` or `domain → codomain`.
"""
function ROCTensorMap(
        data::AbstractDict{<:Sector, <:ROCArray},
        V::TensorMapSpace{S, N₁, N₂}
    ) where {S, N₁, N₂}
    T = eltype(valtype(data))
    t = ROCTensorMap{T}(undef, V)
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
function ROCTensorMap{T}(
        data::DenseVector{T}, codomain::TensorSpace{S},
        domain::TensorSpace{S}
    ) where {T, S}
    return ROCTensorMap(data, codomain ← domain)
end
function ROCTensorMap(
        data::AbstractDict{<:Sector, <:ROCMatrix}, codom::TensorSpace{S},
        dom::TensorSpace{S}
    ) where {S}
    return ROCTensorMap(data, codom ← dom)
end

function ROCTensorMap(ts::TensorMap{T, S, N₁, N₂, A}) where {T, S, N₁, N₂, A}
    return ROCTensorMap{T, S, N₁, N₂}(ROCArray(ts.data), ts.space)
end

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function AMDGPU.$fname(
                codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {S <: IndexSpace}
            return AMDGPU.$fname(codomain ← domain)
        end
        function AMDGPU.$fname(
                ::Type{T}, codomain::TensorSpace{S},
                domain::TensorSpace{S} = one(codomain)
            ) where {T, S <: IndexSpace}
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

for randfun in (:rocrand, :rocrandn)
    randfun! = Symbol(randfun, :!)
    @eval begin
        $randfun(codomain::TensorSpace{S}, domain::TensorSpace{S} = one(codomain)) where {S} = $randfun(codomain ← domain)
        function $randfun(::Type{T}, codomain::TensorSpace{S}, domain::TensorSpace{S} = one(codomain)) where {T, S <: IndexSpace}
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
            t = ROCTensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end
    end
end

# converters
# ----------
function Base.convert(::Type{ROCTensorMap}, d::Dict{Symbol, Any})
    try
        codomain = eval(TensorKit, Meta.parse(d[:codomain]))
        domain = eval(TensorKit, Meta.parse(d[:domain]))
        data = SectorDict(eval(TensorKit, Meta.parse(c)) => ROCArray(b) for (c, b) in d[:data])
        return ROCTensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(
            Base.eval(Main, Meta.parse(c)) => ROCArray(b)
                for (c, b) in d[:data]
        )
        return ROCTensorMap(data, codomain, domain)
    end
end

function Base.convert(::Type{ROCTensorMap}, t::AbstractTensorMap)
    return copy!(ROCTensorMap{scalartype(t)}(undef, space(t)), t)
end

# Scalar implementation
#-----------------------
function TensorKit.scalar(t::ROCTensorMap)

    # TODO: should scalar only work if N₁ == N₂ == 0?
    return @allowscalar dim(codomain(t)) == dim(domain(t)) == 1 ?
        first(blocks(t))[2][1, 1] : throw(DimensionMismatch())
end

TensorKit.scalartype(A::StridedROCArray{T}) where {T} = T
TensorKit.scalartype(::Type{<:ROCTensorMap{T}}) where {T} = T
TensorKit.scalartype(::Type{<:ROCArray{T}}) where {T} = T

function TensorKit.similarstoragetype(TT::Type{<:ROCTensorMap{TTT, S, N₁, N₂}}, ::Type{T}) where {TTT, T, S, N₁, N₂}
    return ROCVector{T, AMDGPU.Mem.HIPBuffer}
end

function Base.convert(
        TT::Type{ROCTensorMap{T, S, N₁, N₂}},
        t::AbstractTensorMap{<:Any, S, N₁, N₂}
    ) where {T, S, N₁, N₂}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function Base.copy!(tdst::ROCTensorMap{T, S, N₁, N₂}, tsrc::ROCTensorMap{T, S, N₁, N₂}) where {T, S, N₁, N₂}
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for ((c, bdst), (_, bsrc)) in zip(blocks(tdst), blocks(tsrc))
        copy!(bdst, bsrc)
    end
    return tdst
end

function Base.copy!(tdst::ROCTensorMap, tsrc::TensorKit.AdjointTensorMap)
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
        TT₁ <: ROCTensorMap{TTT₁, S, N₁, N₂},
        TT₂ <: ROCTensorMap{TTT₂, S, N₁, N₂},
    }
    T = TensorKit.VectorInterface.promote_add(TTT₁, TTT₂)
    return ROCTensorMap{T, S, N₁, N₂}
end

function LinearAlgebra.isposdef(t::ROCTensorMap)
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

# Conversion to ROCArray:
#----------------------
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{ROCArray}, t::AbstractTensorMap)
    I = sectortype(t)
    if I === Trivial
        convert(ROCArray, t[])
    else
        cod = codomain(t)
        dom = domain(t)
        T = sectorscalartype(I) <: Complex ? complex(scalartype(t)) :
            sectorscalartype(I) <: Integer ? scalartype(t) : float(scalartype(t))
        A = AMDGPU.zeros(T, dims(cod)..., dims(dom)...)
        for (f₁, f₂) in fusiontrees(t)
            F = convert(ROCArray, (f₁, f₂))
            Aslice = StridedView(A)[axes(cod, f₁.uncoupled)..., axes(dom, f₂.uncoupled)...]
            add!(Aslice, StridedView(TensorKit._kron(convert(ROCArray, t[f₁, f₂]), F)))
        end
        return A
    end
end
