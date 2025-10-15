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
    if TorA <: Number
        return TensorMap{TorA, S, N₁, N₂, Vector{TorA}}
    elseif TorA <: DenseVector
        return TensorMap{scalartype(TorA), S, N₁, N₂, TorA}
    else
        throw(ArgumentError("argument $TorA should specify a scalar type (`<:Number`) or a storage type `<:DenseVector{<:Number}`"))
    end
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
#--------------------------------
# undef constructors
"""
    TensorMap{T}(undef, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂})
                where {T,S,N₁,N₂}
    TensorMap{T}(undef, codomain ← domain)
    TensorMap{T}(undef, domain → codomain)
    # expert mode: select storage type `A`
    TensorMap{T,S,N₁,N₂,A}(undef, codomain ← domain)
    TensorMap{T,S,N₁,N₂,A}(undef, domain → domain)

Construct a `TensorMap` with uninitialized data.
"""
function TensorMap{T}(::UndefInitializer, V::TensorMapSpace{S, N₁, N₂}) where {T, S, N₁, N₂}
    return TensorMap{T, S, N₁, N₂, Vector{T}}(undef, V)
end
function TensorMap{T}(
        ::UndefInitializer, codomain::TensorSpace{S}, domain::TensorSpace{S}
    ) where {T, S}
    return TensorMap{T}(undef, codomain ← domain)
end
function Tensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T, S}
    return TensorMap{T}(undef, V ← one(V))
end

# constructor starting from vector = independent data (N₁ + N₂ = 1 is special cased below)
# documentation is captured by the case where `data` is a general array
# here, we force the `T` argument to distinguish it from the more general constructor below
function TensorMap{T}(
        data::A, V::TensorMapSpace{S, N₁, N₂}
    ) where {T, S, N₁, N₂, A <: DenseVector{T}}
    return TensorMap{T, S, N₁, N₂, A}(data, V)
end
function TensorMap{T}(
        data::DenseVector{T}, codomain::TensorSpace{S}, domain::TensorSpace{S}
    ) where {T, S}
    return TensorMap(data, codomain ← domain)
end

# constructor starting from block data
"""
    TensorMap(data::AbstractDict{<:Sector,<:AbstractMatrix}, codomain::ProductSpace{S,N₁},
                domain::ProductSpace{S,N₂}) where {S<:ElementarySpace,N₁,N₂}
    TensorMap(data, codomain ← domain)
    TensorMap(data, domain → codomain)

Construct a `TensorMap` by explicitly specifying its block data.

## Arguments
- `data::AbstractDict{<:Sector,<:AbstractMatrix}`: dictionary containing the block data for
  each coupled sector `c` as a matrix of size `(blockdim(codomain, c), blockdim(domain, c))`.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.

Alternatively, the domain and codomain can be specified by passing a [`HomSpace`](@ref)
using the syntax `codomain ← domain` or `domain → codomain`.
"""
function TensorMap(
        data::AbstractDict{<:Sector, <:AbstractMatrix}, V::TensorMapSpace{S, N₁, N₂}
    ) where {S, N₁, N₂}
    T = eltype(valtype(data))
    t = TensorMap{T}(undef, V)
    for (c, b) in blocks(t)
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
        datac = data[c]
        size(datac) == size(b) ||
            throw(DimensionMismatch("wrong size of block for sector $c"))
        copyto!(b, datac)
    end
    for (c, b) in data
        c ∈ blocksectors(t) || isempty(b) ||
            throw(SectorMismatch("data for block sector $c not expected"))
    end
    return t
end
function TensorMap(
        data::AbstractDict{<:Sector, <:AbstractMatrix}, codom::TensorSpace{S}, dom::TensorSpace{S}
    ) where {S}
    return TensorMap(data, codom ← dom)
end

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

# constructor starting from an AbstractArray
"""
    TensorMap(data::AbstractArray, codomain::ProductSpace{S,N₁}, domain::ProductSpace{S,N₂};
                    tol=sqrt(eps(real(float(eltype(data)))))) where {S<:ElementarySpace,N₁,N₂}
    TensorMap(data, codomain ← domain; tol=sqrt(eps(real(float(eltype(data))))))
    TensorMap(data, domain → codomain; tol=sqrt(eps(real(float(eltype(data))))))

Construct a `TensorMap` from a plain multidimensional array.

## Arguments
- `data::DenseArray`: tensor data as a plain array.
- `codomain::ProductSpace{S,N₁}`: the codomain as a `ProductSpace` of `N₁` spaces of type
  `S<:ElementarySpace`.
- `domain::ProductSpace{S,N₂}`: the domain as a `ProductSpace` of `N₂` spaces of type
  `S<:ElementarySpace`.
- `tol=sqrt(eps(real(float(eltype(data)))))::Float64`: 
    
Here, `data` can be specified in three ways:
1) `data` can be a `DenseVector` of length `dim(codomain ← domain)`; in that case it represents
   the actual independent entries of the tensor map. An instance will be created that directly
   references `data`.
2) `data` can be an `AbstractMatrix` of size `(dim(codomain), dim(domain))`
3) `data` can be an `AbstractArray` of rank `N₁ + N₂` with a size matching that of the domain
   and codomain spaces, i.e. `size(data) == (dims(codomain)..., dims(domain)...)`

In case 2 and 3, the `TensorMap` constructor will reconstruct the tensor data such that the
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
function TensorMap(
        data::AbstractArray, V::TensorMapSpace{S, N₁, N₂};
        tol = sqrt(eps(real(float(eltype(data)))))
    ) where {S <: IndexSpace, N₁, N₂}
    T = eltype(data)
    if ndims(data) == 1 && length(data) == dim(V)
        if data isa DenseVector # refer to specific data-capturing constructor
            return TensorMap{T}(data, V)
        else
            return TensorMap{T}(collect(data), V)
        end
    end

    # dimension check
    codom = codomain(V)
    dom = domain(V)
    arraysize = (dims(codom)..., dims(dom)...)
    matsize = (dim(codom), dim(dom))

    if !(size(data) == arraysize || size(data) == matsize)
        throw(DimensionMismatch())
    end

    if sectortype(S) === Trivial # refer to same method, but now with vector argument
        return TensorMap(reshape(data, length(data)), V)
    end

    t = TensorMap{T}(undef, codom, dom)
    arraydata = reshape(collect(data), arraysize)
    t = project_symmetric!(t, arraydata)
    if !isapprox(arraydata, convert(Array, t); atol = tol)
        throw(ArgumentError("Data has non-zero elements at incompatible positions"))
    end
    return t
end
function TensorMap(
        data::AbstractArray, codom::TensorSpace{S}, dom::TensorSpace{S}; kwargs...
    ) where {S}
    return TensorMap(data, codom ← dom; kwargs...)
end
function Tensor(data::AbstractArray, codom::TensorSpace; kwargs...)
    return TensorMap(data, codom ← one(codom); kwargs...)
end

"""
    project_symmetric!(t::TensorMap, data::DenseArray) -> TensorMap

Project the data from a dense array `data` into the tensor map `t`. This function discards 
any data that does not fit the symmetry structure of `t`.
"""
function project_symmetric!(t::TensorMap, data::DenseArray)
    I = sectortype(t)
    if I === Trivial # cannot happen when called from above, but for generality we keep this
        copy!(t.data, reshape(data, length(t.data)))
    else
        for (f₁, f₂) in fusiontrees(t)
            F = convert(Array, (f₁, f₂))
            dataslice = sview(
                data, axes(codomain(t), f₁.uncoupled)..., axes(domain(t), f₂.uncoupled)...
            )
            if FusionStyle(I) === UniqueFusion()
                Fscalar = first(F) # contains a single element
                scale!(t[f₁, f₂], dataslice, conj(Fscalar))
            else
                subblock = t[f₁, f₂]
                szbF = _interleave(size(F), size(subblock))
                indset1 = ntuple(identity, numind(t))
                indset2 = 2 .* indset1
                indset3 = indset2 .- 1
                TensorOperations.tensorcontract!(
                    subblock,
                    F, ((), indset1), true,
                    sreshape(dataslice, szbF),
                    (indset3, indset2), false,
                    (indset1, ()),
                    inv(dim(f₁.coupled)), false
                )
            end
        end
    end
    return t
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

# Indexing and getting and setting the data at the subblock level
#-----------------------------------------------------------------
"""
    Base.getindex(t::TensorMap{T,S,N₁,N₂,I},
                  f₁::FusionTree{I,N₁},
                  f₂::FusionTree{I,N₂}) where {T,SN₁,N₂,I<:Sector}
        -> StridedViews.StridedView
    t[f₁, f₂]

Return a view into the data slice of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. In particular, if `f₁.coupled == f₂.coupled == c`, then a
`StridedViews.StridedView` of size
`(dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled))` is returned which
represents the slice of `block(t, c)` whose row indices correspond to `f₁.uncoupled` and
column indices correspond to `f₂.uncoupled`.

See also [`Base.setindex!(::TensorMap{T,S,N₁,N₂}, ::Any, ::FusionTree{I,N₁}, ::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}`](@ref)
"""
@inline function Base.getindex(
        t::TensorMap{T, S, N₁, N₂}, f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}
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
@inline function Base.getindex(
        t::TensorMap{T, S, N₁, N₂}, f₁::FusionTree{Trivial, N₁}, f₂::FusionTree{Trivial, N₂}
    ) where {T, S, N₁, N₂}
    @boundscheck begin
        sectortype(t) == Trivial || throw(SectorMismatch())
    end
    return sreshape(StridedView(t.data), (dims(codomain(t))..., dims(domain(t))...))
end

"""
    Base.setindex!(t::TensorMap{T,S,N₁,N₂,I},
                   v,
                   f₁::FusionTree{I,N₁},
                   f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}
    t[f₁, f₂] = v

Copies `v` into the  data slice of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. Here, `v` can be any object that can be copied into a `StridedViews.StridedView`
of size `(dims(codomain(t), f₁.uncoupled)..., dims(domain(t), f₂.uncoupled))` using
`Base.copy!`.

See also [`Base.getindex(::TensorMap{T,S,N₁,N₂}, ::FusionTree{I,N₁}, ::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}`](@ref)
"""
@propagate_inbounds function Base.setindex!(
        t::TensorMap{T, S, N₁, N₂}, v, f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}
    ) where {T, S, N₁, N₂, I <: Sector}
    return copy!(getindex(t, f₁, f₂), v)
end

"""
    Base.getindex(t::TensorMap
                  sectors::NTuple{N₁+N₂,I}) where {N₁,N₂,I<:Sector} 
        -> StridedViews.StridedView
    t[sectors]

Return a view into the data slice of `t` corresponding to the splitting - fusion tree pair
with combined uncoupled charges `sectors`. In particular, if `sectors == (s₁..., s₂...)`
where `s₁` and `s₂` correspond to the uncoupled charges in the codomain and domain
respectively, then a `StridedViews.StridedView` of size
`(dims(codomain(t), s₁)..., dims(domain(t), s₂))` is returned.

This method is only available for the case where `FusionStyle(I) isa UniqueFusion`,
since it assumes a  uniquely defined coupled charge.
"""
@inline function Base.getindex(t::TensorMap, sectors::Tuple{I, Vararg{I}}) where {I <: Sector}
    I === sectortype(t) || throw(SectorMismatch("Not a valid sectortype for this tensor."))
    FusionStyle(I) isa UniqueFusion ||
        throw(SectorMismatch("Indexing with sectors only possible if unique fusion"))
    length(sectors) == numind(t) ||
        throw(ArgumentError("Number of sectors does not match."))
    s₁ = TupleTools.getindices(sectors, codomainind(t))
    s₂ = map(dual, TupleTools.getindices(sectors, domainind(t)))
    c1 = length(s₁) == 0 ? unit(I) : (length(s₁) == 1 ? s₁[1] : first(⊗(s₁...)))
    @boundscheck begin
        c2 = length(s₂) == 0 ? unit(I) : (length(s₂) == 1 ? s₂[1] : first(⊗(s₂...)))
        c2 == c1 || throw(SectorMismatch("Not a valid sector for this tensor"))
        hassector(codomain(t), s₁) && hassector(domain(t), s₂)
    end
    f₁ = FusionTree(s₁, c1, map(isdual, tuple(codomain(t)...)))
    f₂ = FusionTree(s₂, c1, map(isdual, tuple(domain(t)...)))
    @inbounds begin
        return t[f₁, f₂]
    end
end
@propagate_inbounds function Base.getindex(t::TensorMap, sectors::Tuple)
    return t[map(sectortype(t), sectors)]
end

# Show
#------
function Base.summary(io::IO, t::TensorMap)
    return print(io, "TensorMap(", space(t), ")")
end
function Base.show(io::IO, t::TensorMap)
    if get(io, :compact, false)
        print(io, "TensorMap(", space(t), ")")
        return
    end
    println(io, "TensorMap(", space(t), "):")
    if sectortype(t) == Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(t)) isa UniqueFusion
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for sector ", f₁.uncoupled, " ← ", f₂.uncoupled, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    else
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f₁, " ← ", f₂, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    end
    return nothing
end

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
