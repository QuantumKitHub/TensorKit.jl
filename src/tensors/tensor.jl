# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#
"""
    struct TensorMap{T, S<:IndexSpace, N₁, N₂, A<:DenseVector{T}} <: AbstractTensorMap{T, S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) for representing tensor maps (morphisms in
a tensor category), where the data is stored in a dense vector.
"""
struct TensorMap{T, S <: IndexSpace, N₁, N₂, A <: DenseVector{T}} <: AbstractTensorMap{T, S, N₁, N₂}
    data::A
    space::TensorMapSpace{S, N₁, N₂}

    # uninitialized constructors
    function TensorMap{T, S, N₁, N₂, A}(
            ::UndefInitializer, space::TensorMapSpace{S, N₁, N₂}
        ) where {T, S <: IndexSpace, N₁, N₂, A <: DenseVector{T}}
        d = fusionblockstructure(space).totaldim
        data = A(undef, d)
        if !isbitstype(T)
            zerovector!(data)
        end
        return TensorMap{T, S, N₁, N₂, A}(data, space)
    end

    # constructors from data
    function TensorMap{T, S, N₁, N₂, A}(
            data::A, space::TensorMapSpace{S, N₁, N₂}
        ) where {T, S <: IndexSpace, N₁, N₂, A <: DenseVector{T}}
        T ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        I = sectortype(S)
        T <: Real && !(sectorscalartype(I) <: Real) &&
            @warn("Tensors with real data might be incompatible with sector type $I", maxlog = 1)
        d = fusionblockstructure(space).totaldim
        length(data) == d || throw(DimensionMismatch("invalid length of data"))
        return new{T, S, N₁, N₂, A}(data, space)
    end
end

"""
    Tensor{T, S, N, A<:DenseVector{T}} = TensorMap{T, S, N, 0, A}

Specific subtype of [`AbstractTensor`](@ref) for representing tensors whose data is stored
in a dense vector.

A `Tensor{T, S, N, A}` is actually a special case `TensorMap{T, S, N, 0, A}`,
i.e. a tensor map with only a non-trivial output space.
"""
const Tensor{T, S, N, A} = TensorMap{T, S, N, 0, A}

function tensormaptype(S::Type{<:IndexSpace}, N₁, N₂, TorA::Type)
    A = _tensormap_storagetype(TorA)
    A <: DenseVector || throw(ArgumentError("Cannot determine a valid storage type from argument $TorA"))
    return TensorMap{scalartype(A), S, N₁, N₂, A}
end

# hook for mapping input types to storage types -- to be implemented in extensions
_tensormap_storagetype(::Type{A}) where {A <: DenseVector{<:Number}} = A
_tensormap_storagetype(::Type{A}) where {A <: Array} = _tensormap_storagetype(scalartype(A))
_tensormap_storagetype(::Type{T}) where {T <: Number} = Vector{T}
function _tensormap_storagetype(::Type{A}) where {A <: AbstractArray}
    PA = parenttype(A)
    PA === A && throw(MethodError(_tensormap_storagetype, A)) # avoid infinite recursion
    return _tensormap_storagetype(PA)
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::TensorMap) = t.space

"""
    storagetype(::Union{T,Type{T}}) where {T<:TensorMap} -> Type{A<:DenseVector}

Return the type of the storage `A` of the tensor map.
"""
storagetype(::Type{<:TensorMap{T, S, N₁, N₂, A}}) where {T, S, N₁, N₂, A <: DenseVector{T}} = A

dim(t::TensorMap) = length(t.data)

# General TensorMap constructors
# ==============================

# INTERNAL: utility type alias that makes constructors also work for type aliases that specify
# different storage types. (i.e. CuTensorMap = TensorMapWithStorage{T, CuVector{T}, ...})
const TensorMapWithStorage{T, A <: DenseVector{T}, S, N₁, N₂} = TensorMap{T, S, N₁, N₂, A}
const TensorWithStorage{T, A <: DenseVector{T}, S, N} = Tensor{T, S, N, A}

# undef constructors
# ------------------
# - dispatch start with TensorMap{T}
# - select A and map to TensorMap{T, S, N₁, N₂, A} where {S, N₁, N₂}
# - select S, N₁, N₂ and map to TensorMap{T, S, N₁, N₂, A}
"""
    TensorMap{T}(undef, codomain::ProductSpace{S, N₁}, domain::ProductSpace{S, N₂}) where {T, S, N₁, N₂}
    TensorMap{T}(undef, codomain ← domain)
    TensorMap{T}(undef, domain → codomain)

Construct a `TensorMap` with uninitialized data with elements of type `T`.
"""
TensorMap{T}(::UndefInitializer, V::TensorMapSpace) where {T} =
    tensormaptype(spacetype(V), numout(V), numin(V), T)(undef, V)
TensorMap{T}(::UndefInitializer, codomain::TensorSpace, domain::TensorSpace) where {T} =
    TensorMap{T}(undef, codomain ← domain)
Tensor{T}(::UndefInitializer, V::TensorSpace) where {T} = TensorMap{T}(undef, V ← one(V))

"""
    TensorMapWithStorage{T, A}(undef, codomain, domain) where {T, A}
    TensorMapWithStorage{T, A}(undef, codomain ← domain) where {T, A}
    TensorMapWithStorage{T, A}(undef, domain → codomain) where {T, A}

Construct a `TensorMap` with uninitialized data stored as `A <: DenseVector{T}`.
"""
TensorMapWithStorage{T, A}(::UndefInitializer, V::TensorMapSpace) where {T, A} =
    tensormaptype(spacetype(V), numout(V), numin(V), A)(undef, V)
TensorMapWithStorage{T, A}(::UndefInitializer, codomain::TensorSpace, domain::TensorSpace) where {T, A} =
    TensorMapWithStorage{T, A}(undef, codomain ← domain)
TensorWithStorage{T, A}(::UndefInitializer, V::TensorSpace) where {T, A} = TensorMapWithStorage{T, A}(undef, V ← one(V))

# raw data constructors
# ---------------------
# - dispatch starts with TensorMap{T}(::DenseVector{T}, ...)
# - select A and map to (TensorMap{T, S, N₁, N₂, A} where {S, N₁, N₂})(::DenseVector{T}, ...)
# - select S, N₁, N₂ and map to TensorMap{T, S, N₁, N₂, A}(::DenseVector{T}, ...)

"""
    TensorMap{T}(data::DenseVector{T}, codomain::ProductSpace{S, N₁}, domain::ProductSpace{S, N₂}) where {T, S, N₁, N₂}
    TensorMap{T}(data::DenseVector{T}, codomain ← domain)
    TensorMap{T}(data::DenseVector{T}, domain → codomain)

Construct a `TensorMap` from the given raw data.
This constructor takes ownership of the provided vector, and will not make an independent copy.
"""
TensorMap{T}(data::DenseVector{T}, V::TensorMapSpace) where {T} =
    tensormaptype(spacetype(V), numout(V), numin(V), typeof(data))(data, V)
TensorMap{T}(data::DenseVector{T}, codomain::TensorSpace, domain::TensorSpace) where {T} =
    TensorMap{T}(data, codomain ← domain)

"""
    TensorMapWithStorage{T, A}(data::DenseVector{T}, codomain, domain) where {T, A}
    TensorMapWithStorage{T, A}(data::DenseVector{T}, codomain ← domain) where {T, A}
    TensorMapWithStorage{T, A}(data::DenseVector{T}, domain → codomain) where {T, A}

Construct a `TensorMap` from the given raw data.
This constructor takes ownership of the provided vector, and will not make an independent copy.
"""
function TensorMapWithStorage{T, A}(data::A, V::TensorMapSpace) where {T, A}
    return tensormaptype(spacetype(V), numout(V), numin(V), typeof(data))(data, V)
end
TensorMapWithStorage{T, A}(data::A, codomain::TensorSpace, domain::TensorSpace) where {T, A} =
    TensorMapWithStorage{T, A}(data, codomain ← domain)

# AbstractArray constructors
# --------------------------
# array can either be a multi-dimensional array, or a matrix representation of the
# corresponding linear map.
#
# - dispatch starts with TensorMap(array::AbstractArray, ...)
# - select T and A and map to (TensorMap{T, S, N₁, N₂, A} where {S, N₁, N₂})(array::AbstractArray, ...)
# - map to project_symmetric!(tdst, array)
#
# !!! note
#   Have to be careful about dispatch collision between data::DenseVector and
#   array::AbstractArray case for N₁ + N₂ == 1

"""
    TensorMap(data::AbstractArray, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂};
                    tol=sqrt(eps(real(float(eltype(data)))))) where {S<:ElementarySpace,N₁,N₂}
    TensorMap(data, codomain ← domain; tol=sqrt(eps(real(float(eltype(data))))))
    TensorMap(data, domain → codomain; tol=sqrt(eps(real(float(eltype(data))))))

Construct a `TensorMap` from a plain multidimensional array.

## Arguments
- `data::DenseArray`: tensor data as a plain array.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type `S<:ElementarySpace`.
- `tol=sqrt(eps(real(float(eltype(data)))))::Float64`: 
    
Here, `data` can be specified in three ways:
1) `data` can be a `DenseVector` of length `dim(codomain ← domain)`; in that case it represents
   the actual independent entries of the tensor map. An instance will be created that directly
   references `data`.
2) `data` can be an `AbstractMatrix` of size `(dim(codomain), dim(domain))`
3) `data` can be an `AbstractArray` of rank `N₁ + N₂` with a size matching that of the domain
   and codomain spaces, i.e. `size(data) == (dims(codomain)..., dims(domain)...)`

In cases 2 and 3, the `TensorMap` constructor will reconstruct the tensor data such that the
resulting tensor `t` satisfies `data == convert(Array, t)`, up to an error specified by `tol`.
For the case where `sectortype(S) == Trivial` and `data isa DenseArray`, the `data` array is
simply reshaped into a vector and used as in case 1 so that the memory will still be shared.
In other cases, new memory will be allocated.

Note that in the case of `N₁ + N₂ = 1`, case 3 also amounts to `data` being a vector, whereas
when `N₁ + N₂ == 2`, case 2 and case 3 both require `data` to be a matrix. Such ambiguous cases
are resolved by checking the size of `data` in an attempt to support all possible
cases.

!!! note
    This constructor for case 2 and 3 only works for `sectortype` values for which conversion
    to a plain array is possible, and only in the case where the `data` actually respects
    the specified symmetry structure, up to a tolerance `tol`.
"""
function TensorMap(data::AbstractArray, V::TensorMapSpace; tol = sqrt(eps(real(float(eltype(data))))))
    T = eltype(data)
    A = _tensormap_storagetype(typeof(data))
    return TensorMapWithStorage{T, A}(data, V; tol)
end
TensorMap(data::AbstractArray, codom::TensorSpace, dom::TensorSpace; kwargs...) =
    TensorMap(data, codom ← dom; kwargs...)
Tensor(data::AbstractArray, codom::TensorSpace; kwargs...) =
    TensorMap(data, codom ← one(codom); kwargs...)

function TensorMapWithStorage{T, A}(
        data::AbstractArray, V::TensorMapSpace; tol = sqrt(eps(real(float(eltype(data)))))
    ) where {T, A}
    # refer to specific raw data constructors if input is a vector of the correct length
    ndims(data) == 1 && length(data) == dim(V) &&
        return tensormaptype(spacetype(V), numout(V), numin(V), A)(data, V)

    # special case trivial: refer to same method, but now with vector argument
    sectortype(V) === Trivial &&
        return tensormaptype(spacetype(V), numout(V), numin(V), A)(reshape(data, length(data)), V)

    # do projection
    t = TensorMapWithStorage{T, A}(undef, V)
    t = project_symmetric!(t, data)

    # verify result
    isapprox(reshape(data, dims(t)), convert(Array, t); atol = tol) ||
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))

    return t
end
TensorMapWithStorage{T, A}(data::AbstractArray, codom::TensorSpace, dom::TensorSpace; kwargs...) where {T, A} =
    TensorMapWithStorage{T, A}(data, codom ← dom; kwargs...)
TensorWithStorage{T, A}(data::AbstractArray, codom::TensorSpace; kwargs...) where {T, A} =
    TensorMapWithStorage{T, A}(data, codom ← one(codom); kwargs...)

# block data constructors
# -----------------------
# - dispatch starts with TensorMap(::AbstractDict{<:Sector, <:AbstractMatrix}, ...)
# - select T and A and map to (TensorMap{T, S, N₁, N₂, A} where {S, N₁, N₂})(::AbstractDict{<:Sector, <:AbstractMatrix}
# - extract/construct raw data and map to raw data constructor

# private: allowed block data types
const _BlockData{I <: Sector, A <: AbstractMatrix} = AbstractDict{I, A}

"""
    TensorMap(data::AbstractDict{<:Sector, <:AbstractMatrix}, codomain::ProductSpace, domain::ProductSpace)
    TensorMap(data, codomain ← domain)
    TensorMap(data, domain → codomain)

Construct a `TensorMap` by explicitly specifying its block data.

## Arguments
- `data::AbstractDict{<:Sector, <:AbstractMatrix}`: dictionary containing the block data for
  each coupled sector `c` as a matrix of size `(blockdim(codomain, c), blockdim(domain, c))`.
- `codomain::ProductSpace{S, N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type `S <: ElementarySpace`.
- `domain::ProductSpace{S, N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type `S <: ElementarySpace`.
"""
function TensorMap(data::_BlockData, V::TensorMapSpace)
    A = _tensormap_storagetype(valtype(data))
    return TensorMapWithStorage{scalartype(A), A}(data, V)
end
TensorMap(data::_BlockData, codom::TensorSpace, dom::TensorSpace) =
    TensorMap(data, codom ← dom)

function TensorMapWithStorage{T, A}(data::_BlockData, V::TensorMapSpace) where {T, A}
    t = TensorMapWithStorage{T, A}(undef, V)

    # check that there aren't too many blocks
    for (c, b) in data
        c ∈ blocksectors(t) || isempty(b) || throw(SectorMismatch("data for block sector $c not expected"))
    end

    # fill in the blocks -- rely on conversion in copy
    for (c, b) in blocks(t)
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
        datac = data[c]
        size(datac) == size(b) || throw(DimensionMismatch("wrong size of block for sector $c"))
        copy!(b, datac)
    end

    return t
end
TensorMapWithStorage{T, A}(data::_BlockData, codom::TensorSpace, dom::TensorSpace) where {T, A} =
    TensorMapWithStorage{T, A}(data, codom ← dom)

# Higher-level constructors
# =========================
@doc """
    zeros([T=Float64,], codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂}) where {S,N₁,N₂,T}
    zeros([T=Float64,], codomain ← domain)

Create a `TensorMap` with element type `T`, of all zeros with spaces specified by `codomain` and `domain`.
"""
Base.zeros(::Type, ::HomSpace)

@doc """
    ones([T=Float64,], codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂}) where {S,N₁,N₂,T}
    ones([T=Float64,], codomain ← domain)
    
Create a `TensorMap` with element type `T`, of all ones with spaces specified by `codomain` and `domain`.
"""
Base.ones(::Type, ::HomSpace)

for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function Base.$fname(
                codomain::TensorSpace{S}, domain::TensorSpace{S} = one(codomain)
            ) where {S <: IndexSpace}
            return Base.$fname(codomain ← domain)
        end
        function Base.$fname(
                ::Type{T}, codomain::TensorSpace{S}, domain::TensorSpace{S} = one(codomain)
            ) where {T, S <: IndexSpace}
            return Base.$fname(T, codomain ← domain)
        end
        Base.$fname(V::TensorMapSpace) = Base.$fname(Float64, V)
        function Base.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randf in (:rand, :randn, :randexp, :randisometry)
    _docstr = """
        $randf([rng=default_rng()], [T=Float64], codomain::ProductSpace{S,N₁},
                     domain::ProductSpace{S,N₂}) where {S,N₁,N₂,T} -> t
        $randf([rng=default_rng()], [T=Float64], codomain ← domain) -> t
        
    Generate a tensor `t` with entries generated by `$randf`.

    See also [`Random.$(randf)!`](@ref).
    """
    _docstr! = """
        $(randf)!([rng=default_rng()], t::AbstractTensorMap) -> t
        
    Fill the tensor `t` with entries generated by `$(randf)!`.

    See also [`Random.$(randf)`](@ref).
    """

    if randf != :randisometry
        randfun = GlobalRef(Random, randf)
        randfun! = GlobalRef(Random, Symbol(randf, :!))
    else
        randfun = randf
        randfun! = Symbol(randf, :!)
    end

    @eval begin
        @doc $_docstr $randfun(::Type, ::HomSpace)
        @doc $_docstr! $randfun!(::Type, ::HomSpace)

        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(codomain::TensorSpace{S}, domain::TensorSpace{S}) where {S <: IndexSpace}
            return $randfun(codomain ← domain)
        end
        function $randfun(
                ::Type{T}, codomain::TensorSpace{S}, domain::TensorSpace{S}
            ) where {T, S <: IndexSpace}
            return $randfun(T, codomain ← domain)
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T}, codomain::TensorSpace{S}, domain::TensorSpace{S}
            ) where {T, S <: IndexSpace}
            return $randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        $randfun(codomain::TensorSpace) = $randfun(codomain ← one(codomain))
        function $randfun(::Type{T}, codomain::TensorSpace) where {T}
            return $randfun(T, codomain ← one(codomain))
        end
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T}, codomain::TensorSpace
            ) where {T}
            return $randfun(rng, T, codomain ← one(domain))
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
        $randfun!(t::AbstractTensorMap) = $randfun!(Random.default_rng(), t)

        # implementation
        function $randfun(
                rng::Random.AbstractRNG, ::Type{T}, V::TensorMapSpace
            ) where {T}
            t = TensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end

        function $randfun!(rng::Random.AbstractRNG, t::AbstractTensorMap)
            for (_, b) in blocks(t)
                $randfun!(rng, b)
            end
            return t
        end
    end
end

# Efficient copy constructors
#-----------------------------
Base.copy(t::TensorMap) = typeof(t)(copy(t.data), t.space)

# Conversion between TensorMap and Dict, for read and write purpose
#------------------------------------------------------------------
function Base.convert(::Type{Dict}, t::AbstractTensorMap)
    d = Dict{Symbol, Any}()
    d[:codomain] = repr(codomain(t))
    d[:domain] = repr(domain(t))
    data = Dict{String, Any}()
    for (c, b) in blocks(t)
        data[repr(c)] = Array(b)
    end
    d[:data] = data
    return d
end
function Base.convert(::Type{TensorMap}, d::Dict{Symbol, Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

# Getting and setting the data at the block level
#-------------------------------------------------
block(t::TensorMap, c::Sector) = blocks(t)[c]

blocks(t::TensorMap) = BlockIterator(t, fusionblockstructure(t).blockstructure)

function blocktype(::Type{TT}) where {TT <: TensorMap}
    A = storagetype(TT)
    T = eltype(A)
    return Base.ReshapedArray{T, 2, SubArray{T, 1, A, Tuple{UnitRange{Int}}, true}, Tuple{}}
end

function Base.iterate(iter::BlockIterator{<:TensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, (sz, r)), newstate = next
    return c => reshape(view(iter.t.data, r), sz), newstate
end

function Base.getindex(iter::BlockIterator{<:TensorMap}, c::Sector)
    sectortype(iter.t) === typeof(c) || throw(SectorMismatch())
    (d₁, d₂), r = get(iter.structure, c) do
        # is s is not a key, at least one of the two dimensions will be zero:
        # it then does not matter where exactly we construct a view in `t.data`,
        # as it will have length zero anyway
        d₁′ = blockdim(codomain(iter.t), c)
        d₂′ = blockdim(domain(iter.t), c)
        l = d₁′ * d₂′
        return (d₁′, d₂′), 1:l
    end
    return reshape(view(iter.t.data, r), (d₁, d₂))
end

# Getting and setting the data at the subblock level
# --------------------------------------------------
function subblock(
        t::TensorMap{T, S, N₁, N₂}, (f₁, f₂)::Tuple{FusionTree{I, N₁}, FusionTree{I, N₂}}
    ) where {T, S, N₁, N₂, I <: Sector}
    structure = fusionblockstructure(t)
    @boundscheck begin
        haskey(structure.fusiontreeindices, (f₁, f₂)) || throw(SectorMismatch())
    end
    @inbounds begin
        i = structure.fusiontreeindices[(f₁, f₂)]
        sz, str, offset = structure.fusiontreestructure[i]
        return StridedView(t.data, sz, str, offset)
    end
end

# The following is probably worth special casing for trivial tensors
@inline function subblock(
        t::TensorMap{T, S, N₁, N₂}, (f₁, f₂)::Tuple{FusionTree{Trivial, N₁}, FusionTree{Trivial, N₂}}
    ) where {T, S, N₁, N₂}
    @boundscheck begin
        sectortype(t) == Trivial || throw(SectorMismatch())
    end
    return sreshape(StridedView(t.data), dims(t))
end

# Show
#------
function type_repr(::Type{TensorMap{T, S, N₁, N₂, A}}) where {T, S, N₁, N₂, A}
    return "TensorMap{$T, $(type_repr(S)), $N₁, $N₂, $A}"
end

function Base.showarg(io::IO, t::TensorMap, toplevel::Bool)
    !toplevel && print(io, "::")
    print(io, type_repr(typeof(t)))
    return nothing
end

Base.show(io::IO, t::TensorMap) =
    print(io, type_repr(typeof(t)), "(", t.data, ", ", space(t), ")")

# Complex, real and imaginary parts
#-----------------------------------
for f in (:real, :imag, :complex)
    @eval begin
        function Base.$f(t::TensorMap)
            return TensorMap($f(t.data), space(t))
        end
    end
end

# Conversion and promotion:
#---------------------------
Base.convert(::Type{TensorMap}, t::TensorMap) = t
function Base.convert(::Type{TensorMap}, t::AbstractTensorMap)
    return copy!(TensorMap{scalartype(t)}(undef, space(t)), t)
end

function Base.convert(
        TT::Type{TensorMap{T, S, N₁, N₂, A}}, t::AbstractTensorMap{<:Any, S, N₁, N₂}
    ) where {T, S, N₁, N₂, A}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function Base.promote_rule(
        ::Type{<:TT₁}, ::Type{<:TT₂}
    ) where {S, N₁, N₂, TT₁ <: TensorMap{<:Any, S, N₁, N₂}, TT₂ <: TensorMap{<:Any, S, N₁, N₂}}
    A = VectorInterface.promote_add(storagetype(TT₁), storagetype(TT₂))
    T = scalartype(A)
    return TensorMap{T, S, N₁, N₂, A}
end
