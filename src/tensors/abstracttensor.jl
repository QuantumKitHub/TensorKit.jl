# abstracttensor.jl
#
# Abstract Tensor type
#----------------------
"""
    abstract type AbstractTensorMap{T<:Number, S<:IndexSpace, N₁, N₂} end

Abstract supertype of all tensor maps, i.e. linear maps between tensor products of vector
spaces of type `S<:IndexSpace`, with element type `T`. An `AbstractTensorMap` maps from an
input space of type `ProductSpace{S, N₂}` to an output space of type `ProductSpace{S, N₁}`.
"""
abstract type AbstractTensorMap{T <: Number, S <: IndexSpace, N₁, N₂} end

"""
    AbstractTensor{T,S,N} = AbstractTensorMap{T,S,N,0}

Abstract supertype of all tensors, i.e. elements in the tensor product space of type
`ProductSpace{S, N}`, with element type `T`.

An `AbstractTensor{T, S, N}` is actually a special case `AbstractTensorMap{T, S, N, 0}`,
i.e. a tensor map with only non-trivial output spaces.
"""
const AbstractTensor{T, S, N} = AbstractTensorMap{T, S, N, 0}

# tensor characteristics: type information
#------------------------------------------
"""
    eltype(::AbstractTensorMap) -> Type{T}
    eltype(::Type{<:AbstractTensorMap}) -> Type{T}

Return the scalar or element type `T` of a tensor.
"""
Base.eltype(::Type{<:AbstractTensorMap{T}}) where {T} = T

spacetype(::Type{<:AbstractTensorMap{<:Any, S}}) where {S} = S

function InnerProductStyle(::Type{TT}) where {TT <: AbstractTensorMap}
    return InnerProductStyle(spacetype(TT))
end

@doc """
    storagetype(t::AbstractTensorMap) -> Type{A<:AbstractVector}
    storagetype(T::Type{<:AbstractTensorMap}) -> Type{A<:AbstractVector}

Return the type of vector that stores the data of a tensor.
""" storagetype

similarstoragetype(TT::Type{<:AbstractTensorMap}) = similarstoragetype(TT, scalartype(TT))

function similarstoragetype(TT::Type{<:AbstractTensorMap}, ::Type{T}) where {T}
    return Core.Compiler.return_type(similar, Tuple{storagetype(TT), Type{T}})
end

# tensor characteristics: space and index information
#-----------------------------------------------------
"""
    space(t::AbstractTensorMap{T,S,N₁,N₂}) -> HomSpace{S,N₁,N₂}
    space(t::AbstractTensorMap{T,S,N₁,N₂}, i::Int) -> S

The index information of a tensor, i.e. the `HomSpace` of its domain and codomain. If `i` is specified, return the `i`-th index space.
"""
space(t::AbstractTensorMap, i::Int) = space(t)[i]

@doc """
    codomain(t::AbstractTensorMap{T,S,N₁,N₂}) -> ProductSpace{S,N₁}
    codomain(t::AbstractTensorMap{T,S,N₁,N₂}, i::Int) -> S

Return the codomain of a tensor, i.e. the product space of the output spaces. If `i` is
specified, return the `i`-th output space. Implementations should provide `codomain(t)`.

See also [`domain`](@ref) and [`space`](@ref).
""" codomain

codomain(t::AbstractTensorMap) = codomain(space(t))
codomain(t::AbstractTensorMap, i) = codomain(t)[i]
target(t::AbstractTensorMap) = codomain(t) # categorical terminology

@doc """
    domain(t::AbstractTensorMap{T,S,N₁,N₂}) -> ProductSpace{S,N₂}
    domain(t::AbstractTensorMap{T,S,N₁,N₂}, i::Int) -> S

Return the domain of a tensor, i.e. the product space of the input spaces. If `i` is
specified, return the `i`-th input space. Implementations should provide `domain(t)`.

See also [`codomain`](@ref) and [`space`](@ref).
""" domain

domain(t::AbstractTensorMap) = domain(space(t))
domain(t::AbstractTensorMap, i) = domain(t)[i]
source(t::AbstractTensorMap) = domain(t) # categorical terminology

"""
    numout(::Union{TT,Type{TT}}) where {TT<:AbstractTensorMap} -> Int

Return the number of output spaces of a tensor. This is equivalent to the number of spaces in the codomain of that tensor.

See also [`numin`](@ref) and [`numind`](@ref).
"""
numout(::Type{<:AbstractTensorMap{T, S, N₁}}) where {T, S, N₁} = N₁

"""
    numin(::Union{TT,Type{TT}}) where {TT<:AbstractTensorMap} -> Int

Return the number of input spaces of a tensor. This is equivalent to the number of spaces in the domain of that tensor.

See also [`numout`](@ref) and [`numind`](@ref).
"""
numin(::Type{<:AbstractTensorMap{T, S, N₁, N₂}}) where {T, S, N₁, N₂} = N₂

"""
    numind(::Union{T,Type{T}}) where {T<:AbstractTensorMap} -> Int

Return the total number of input and output spaces of a tensor. This is equivalent to the
total number of spaces in the domain and codomain of that tensor.

See also [`numout`](@ref) and [`numin`](@ref).
"""
numind(::Type{TT}) where {TT <: AbstractTensorMap} = numin(TT) + numout(TT)
const order = numind

"""
    codomainind(::Union{TT,Type{TT}}) where {TT<:AbstractTensorMap} -> Tuple{Int}

Return all indices of the codomain of a tensor.

See also [`domainind`](@ref) and [`allind`](@ref).
"""
function codomainind(::Type{TT}) where {TT <: AbstractTensorMap}
    return ntuple(identity, numout(TT))
end
codomainind(t::AbstractTensorMap) = codomainind(typeof(t))

"""
    domainind(::Union{TT,Type{TT}}) where {TT<:AbstractTensorMap} -> Tuple{Int}

Return all indices of the domain of a tensor.

See also [`codomainind`](@ref) and [`allind`](@ref).
"""
function domainind(::Type{TT}) where {TT <: AbstractTensorMap}
    return ntuple(n -> numout(TT) + n, numin(TT))
end
domainind(t::AbstractTensorMap) = domainind(typeof(t))

"""
    allind(::Union{TT,Type{TT}}) where {TT<:AbstractTensorMap} -> Tuple{Int}

Return all indices of a tensor, i.e. the indices of its domain and codomain.

See also [`codomainind`](@ref) and [`domainind`](@ref).
"""
function allind(::Type{TT}) where {TT <: AbstractTensorMap}
    return ntuple(identity, numind(TT))
end
allind(t::AbstractTensorMap) = allind(typeof(t))

function adjointtensorindex(t::AbstractTensorMap, i)
    return ifelse(i <= numout(t), numin(t) + i, i - numout(t))
end

function adjointtensorindices(t::AbstractTensorMap, indices::IndexTuple)
    return map(i -> adjointtensorindex(t, i), indices)
end

function adjointtensorindices(t::AbstractTensorMap, p::Index2Tuple)
    return (adjointtensorindices(t, p[1]), adjointtensorindices(t, p[2]))
end

# tensor characteristics: work on instances and pass to type
#------------------------------------------------------------
InnerProductStyle(t::AbstractTensorMap) = InnerProductStyle(typeof(t))
storagetype(t::AbstractTensorMap) = storagetype(typeof(t))
blocktype(t::AbstractTensorMap) = blocktype(typeof(t))
similarstoragetype(t::AbstractTensorMap, T = scalartype(t)) = similarstoragetype(typeof(t), T)

numout(t::AbstractTensorMap) = numout(typeof(t))
numin(t::AbstractTensorMap) = numin(typeof(t))
numind(t::AbstractTensorMap) = numind(typeof(t))

# tensor characteristics: data structure and properties
#------------------------------------------------------
"""
    fusionblockstructure(t::AbstractTensorMap) -> TensorStructure

Return the necessary structure information to decompose a tensor in blocks labeled by
coupled sectors and in subblocks labeled by a splitting-fusion tree couple.
"""
fusionblockstructure(t::AbstractTensorMap) = fusionblockstructure(space(t))

"""
    dim(t::AbstractTensorMap) -> Int

The total number of free parameters of a tensor, discounting the entries that are fixed by
symmetry. This is also the dimension of the `HomSpace` on which the `TensorMap` is defined.
"""
dim(t::AbstractTensorMap) = fusionblockstructure(t).totaldim

"""
    blocksectors(t::AbstractTensorMap)

Return an iterator over all coupled sectors of a tensor.
"""
blocksectors(t::AbstractTensorMap) = keys(fusionblockstructure(t).blockstructure)

"""
    hasblock(t::AbstractTensorMap, c::Sector) -> Bool

Verify whether a tensor has a block corresponding to a coupled sector `c`.
"""
hasblock(t::AbstractTensorMap, c::Sector) = c ∈ blocksectors(t)

# TODO: convenience methods, do we need them?
# """
#     blocksize(t::AbstractTensorMap, c::Sector) -> Tuple{Int,Int}

# Return the size of the matrix block of a tensor corresponding to a coupled sector `c`.

# See also [`blockdim`](@ref) and [`blockrange`](@ref).
# """
# function blocksize(t::AbstractTensorMap, c::Sector)
#     return fusionblockstructure(t).blockstructure[c][1]
# end

# """
#     blockdim(t::AbstractTensorMap, c::Sector) -> Int

# Return the total dimension (length) of the matrix block of a tensor corresponding to
# a coupled sector `c`.

# See also [`blocksize`](@ref) and [`blockrange`](@ref).
# """
# function blockdim(t::AbstractTensorMap, c::Sector)
#     return *(blocksize(t, c)...)
# end

# """
#     blockrange(t::AbstractTensorMap, c::Sector) -> UnitRange{Int}

# Return the range at which to find the matrix block of a tensor corresponding to a
# coupled sector `c`, within the total data vector of length `dim(t)`.
# """
# function blockrange(t::AbstractTensorMap, c::Sector)
#     return fusionblockstructure(t).blockstructure[c][2]
# end

"""
    fusiontrees(t::AbstractTensorMap)

Return an iterator over all splitting - fusion tree pairs of a tensor.
"""
fusiontrees(t::AbstractTensorMap) = fusionblockstructure(t).fusiontreelist

fusiontreetype(t::AbstractTensorMap) = fusiontreetype(typeof(t))
function fusiontreetype(::Type{T}) where {T <: AbstractTensorMap}
    I = sectortype(T)
    return Tuple{fusiontreetype(I, numout(T)), fusiontreetype(I, numin(T))}
end

# auxiliary function
@inline function trivial_fusiontree(t::AbstractTensorMap)
    sectortype(t) === Trivial ||
        throw(SectorMismatch("Only valid for tensors with trivial symmetry"))
    spaces1 = codomain(t).spaces
    spaces2 = domain(t).spaces
    f₁ = FusionTree{Trivial}(map(x -> Trivial(), spaces1), Trivial(), map(isdual, spaces1))
    f₂ = FusionTree{Trivial}(map(x -> Trivial(), spaces2), Trivial(), map(isdual, spaces2))
    return (f₁, f₂)
end

# tensor data: block access
#---------------------------
@doc """
    blocks(t::AbstractTensorMap)

Return an iterator over all blocks of a tensor, i.e. all coupled sectors and their
corresponding matrix blocks.

See also [`block`](@ref), [`blocksectors`](@ref), [`blockdim`](@ref) and [`hasblock`](@ref).
"""
function blocks(t::AbstractTensorMap)
    iter = Base.Iterators.map(blocksectors(t)) do c
        return c => block(t, c)
    end
    return iter
end

@doc """
    block(t::AbstractTensorMap, c::Sector)

Return the matrix block of a tensor corresponding to a coupled sector `c`.

See also [`blocks`](@ref), [`blocksectors`](@ref), [`blockdim`](@ref) and [`hasblock`](@ref).
""" block

@doc """
    blocktype(t)

Return the type of the matrix blocks of a tensor.
""" blocktype
function blocktype(::Type{T}) where {T <: AbstractTensorMap}
    return Core.Compiler.return_type(block, Tuple{T, sectortype(T)})
end

# tensor data: subblock access
# ----------------------------
@doc """
    subblocks(t::AbstractTensorMap)

Return an iterator over all subblocks of a tensor, i.e. all fusiontrees and their
corresponding tensor subblocks.

See also [`subblock`](@ref), [`fusiontrees`](@ref), and [`hassubblock`](@ref).
"""
subblocks(t::AbstractTensorMap) = SubblockIterator(t, fusiontrees(t))

const _doc_subblock = """
Return a view into the data of `t` corresponding to the splitting - fusion tree pair
`(f₁, f₂)`. In particular, this is an `AbstractArray{T}` with `T = scalartype(t)`, of size
`(dims(codomain(t), f₁.uncoupled)..., dims(codomain(t), f₂.uncoupled)...)`.

Whenever `FusionStyle(sectortype(t)) isa UniqueFusion` , it is also possible to provide only
the external `sectors`, in which case the fusion tree pair will be constructed automatically.
"""

@doc """
    subblock(t::AbstractTensorMap, (f₁, f₂)::Tuple{FusionTree,FusionTree})
    subblock(t::AbstractTensorMap, sectors::Tuple{Vararg{Sector}})

$_doc_subblock

In general, new tensor types should provide an implementation of this function for the
fusion tree signature.

See also [`subblocks`](@ref) and [`fusiontrees`](@ref).
""" subblock

Base.@propagate_inbounds function subblock(t::AbstractTensorMap, sectors::Tuple{I, Vararg{I}}) where {I <: Sector}
    # input checking
    I === sectortype(t) || throw(SectorMismatch("Not a valid sectortype for this tensor."))
    FusionStyle(I) isa UniqueFusion ||
        throw(SectorMismatch("Indexing with sectors is only possible for unique fusion styles."))
    length(sectors) == numind(t) || throw(ArgumentError("invalid number of sectors"))

    # convert to fusiontrees
    s₁ = TupleTools.getindices(sectors, codomainind(t))
    s₂ = map(dual, TupleTools.getindices(sectors, domainind(t)))
    c1 = length(s₁) == 0 ? unit(I) : (length(s₁) == 1 ? s₁[1] : first(⊗(s₁...)))
    @boundscheck begin
        hassector(codomain(t), s₁) && hassector(domain(t), s₂) || throw(BoundsError(t, sectors))
        c2 = length(s₂) == 0 ? unit(I) : (length(s₂) == 1 ? s₂[1] : first(⊗(s₂...)))
        c2 == c1 || throw(SectorMismatch("Not a valid fusion channel for this tensor"))
    end
    f₁ = FusionTree(s₁, c1, map(isdual, tuple(codomain(t)...)))
    f₂ = FusionTree(s₂, c1, map(isdual, tuple(domain(t)...)))
    return @inbounds subblock(t, (f₁, f₂))
end
Base.@propagate_inbounds function subblock(t::AbstractTensorMap, sectors::Tuple)
    return subblock(t, map(Base.Fix1(convert, sectortype(t)), sectors))
end
# attempt to provide better error messages
function subblock(t::AbstractTensorMap, (f₁, f₂)::Tuple{FusionTree, FusionTree})
    (sectortype(t)) == sectortype(f₁) == sectortype(f₂) ||
        throw(SectorMismatch("Not a valid sectortype for this tensor."))
    numout(t) == length(f₁) && numin(t) == length(f₂) ||
        throw(DimensionMismatch("Invalid number of fusiontree legs for this tensor."))
    throw(MethodError(subblock, (t, (f₁, f₂))))
end

@doc """
    subblocktype(t)
    subblocktype(::Type{T})

Return the type of the tensor subblocks of a tensor.
""" subblocktype

function subblocktype(::Type{T}) where {T <: AbstractTensorMap}
    return Core.Compiler.return_type(subblock, Tuple{T, fusiontreetype(T)})
end
subblocktype(t) = subblocktype(typeof(t))
subblocktype(T::Type) = throw(MethodError(subblocktype, (T,)))

# Indexing behavior
# -----------------
# by default getindex returns views!
@doc """
    Base.getindex(t::AbstractTensorMap, sectors::Tuple{Vararg{Sector}})
    t[sectors]
    Base.getindex(t::AbstractTensorMap, f₁::FusionTree, f₂::FusionTree)
    t[f₁, f₂]

$_doc_subblock

!!! warning
    Contrary to Julia's array types, the default behavior is to return a view into the tensor data.
    As a result, modifying the view will modify the data in the tensor.

See also [`subblock`](@ref), [`subblocks`](@ref) and [`fusiontrees`](@ref).
""" Base.getindex(::AbstractTensorMap, ::Tuple{I, Vararg{I}}) where {I <: Sector},
    Base.getindex(::AbstractTensorMap, ::FusionTree, ::FusionTree)

@inline Base.getindex(t::AbstractTensorMap, sectors::Tuple{I, Vararg{I}}) where {I <: Sector} =
    subblock(t, sectors)
@inline Base.getindex(t::AbstractTensorMap, f₁::FusionTree, f₂::FusionTree) =
    subblock(t, (f₁, f₂))

@doc """
    Base.setindex!(t::AbstractTensorMap, v, sectors::Tuple{Vararg{Sector}})
    t[sectors] = v
    Base.setindex!(t::AbstractTensorMap, v, f₁::FusionTree, f₂::FusionTree)
    t[f₁, f₂] = v

Copies `v` into the data slice of `t` corresponding to the splitting - fusion tree pair `(f₁, f₂)`.
By default, `v` can be any object that can be copied into the view associated with `t[f₁, f₂]`.

See also [`subblock`](@ref), [`subblocks`](@ref) and [`fusiontrees`](@ref).
""" Base.setindex!(::AbstractTensorMap, ::Any, ::Tuple{I, Vararg{I}}) where {I <: Sector},
    Base.setindex!(::AbstractTensorMap, ::Any, ::FusionTree, ::FusionTree)

@inline Base.setindex!(t::AbstractTensorMap, v, sectors::Tuple{I, Vararg{I}}) where {I <: Sector} =
    copy!(subblock(t, sectors), v)
@inline Base.setindex!(t::AbstractTensorMap, v, f₁::FusionTree, f₂::FusionTree) =
    copy!(subblock(t, (f₁, f₂)), v)

# Derived indexing behavior for tensors with trivial symmetry
#-------------------------------------------------------------
using TensorKit.Strided: SliceIndex

# For a tensor with trivial symmetry, allow direct indexing
# TODO: should we allow range indices as well
# TODO 2: should we enable this for (abelian) symmetric tensors with some CUDA like `allowscalar` flag?
# TODO 3: should we then also allow at least `getindex` for nonabelian tensors
"""
    Base.getindex(t::AbstractTensorMap, indices::Vararg{Int})
    t[indices]

Return a view into the data slice of `t` corresponding to `indices`, by slicing the
`StridedViews.StridedView` into the full data array.
"""
@inline function Base.getindex(t::AbstractTensorMap, indices::Vararg{SliceIndex})
    data = t[trivial_fusiontree(t)...]
    @boundscheck checkbounds(data, indices...)
    @inbounds v = data[indices...]
    return v
end
"""
    Base.setindex!(t::AbstractTensorMap, v, indices::Vararg{Int})
    t[indices] = v

Assigns `v` to the data slice of `t` corresponding to `indices`.
"""
@inline function Base.setindex!(t::AbstractTensorMap, v, indices::Vararg{SliceIndex})
    data = t[trivial_fusiontree(t)...]
    @boundscheck checkbounds(data, indices...)
    @inbounds data[indices...] = v
    return v
end

# TODO : probably deprecate the following
# For a tensor with trivial symmetry, allow no argument indexing
"""
    Base.getindex(t::AbstractTensorMap)
    t[]

Return a view into the data of `t` as a `StridedViews.StridedView` of size
`(dims(codomain(t))..., dims(domain(t))...)`.
"""
@inline function Base.getindex(t::AbstractTensorMap)
    return t[trivial_fusiontree(t)...]
end
@inline Base.setindex!(t::AbstractTensorMap, v) = copy!(getindex(t), v)

# Similar
#---------
# The implementation is written for similar(t, TorA, V::TensorMapSpace) -> TensorMap
# and all other methods are just filling in default arguments
# 4 arguments
@doc """
    similar(t::AbstractTensorMap, [AorT=storagetype(t)], [V=space(t)])
    similar(t::AbstractTensorMap, [AorT=storagetype(t)], codomain, domain)

Creates an uninitialized mutable tensor with the given scalar or storagetype `AorT` and
structure `V` or `codomain ← domain`, based on the source tensormap. The second and third
arguments are both optional, defaulting to the given tensor's `storagetype` and `space`.
The structure may be specified either as a single `HomSpace` argument or as `codomain` and
`domain`.

By default, this will result in `TensorMap{T}(undef, V)` when custom objects do not
specialize this method.
""" Base.similar(::AbstractTensorMap, args...)

function Base.similar(
        t::AbstractTensorMap, ::Type{T}, codomain::TensorSpace{S}, domain::TensorSpace{S}
    ) where {T, S}
    return similar(t, T, codomain ← domain)
end
# 3 arguments
function Base.similar(
        t::AbstractTensorMap, codomain::TensorSpace{S}, domain::TensorSpace{S}
    ) where {S}
    return similar(t, similarstoragetype(t), codomain ← domain)
end
function Base.similar(t::AbstractTensorMap, ::Type{T}, codomain::TensorSpace) where {T}
    return similar(t, T, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::AbstractTensorMap, codomain::TensorSpace)
    return similar(t, similarstoragetype(t), codomain ← one(codomain))
end
Base.similar(t::AbstractTensorMap, P::TensorMapSpace) = similar(t, storagetype(t), P)
Base.similar(t::AbstractTensorMap, ::Type{T}) where {T} = similar(t, T, space(t))
# 1 argument
Base.similar(t::AbstractTensorMap) = similar(t, similarstoragetype(t), space(t))

# generic implementation for AbstractTensorMap -> returns `TensorMap`
function Base.similar(t::AbstractTensorMap, ::Type{TorA}, P::TensorMapSpace{S}) where {TorA, S}
    if TorA <: Number
        T = TorA
        A = similarstoragetype(t, T)
    elseif TorA <: DenseVector
        A = TorA
        T = scalartype(A)
    else
        throw(ArgumentError("Type $TorA not supported for similar"))
    end

    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    return TensorMap{T, S, N₁, N₂, A}(undef, P)
end

# implementation in type-domain
function Base.similar(::Type{TT}, P::TensorMapSpace) where {TT <: AbstractTensorMap}
    return TensorMap{scalartype(TT)}(undef, P)
end
function Base.similar(
        ::Type{TT}, cod::TensorSpace{S}, dom::TensorSpace{S}
    ) where {TT <: AbstractTensorMap, S}
    return TensorMap{scalartype(TT)}(undef, cod, dom)
end

# Equality and approximality
#----------------------------
function Base.:(==)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || return false
    for c in blocksectors(t1)
        block(t1, c) == block(t2, c) || return false
    end
    return true
end
function Base.hash(t::AbstractTensorMap, h::UInt)
    h = hash(codomain(t), h)
    h = hash(domain(t), h)
    for (c, b) in blocks(t)
        h = hash(c, hash(b, h))
    end
    return h
end

function Base.isapprox(
        t1::AbstractTensorMap, t2::AbstractTensorMap;
        atol::Real = 0, rtol::Real = Base.rtoldefault(scalartype(t1), scalartype(t2), atol)
    )
    d = norm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol * max(norm(t1), norm(t2)))
    else
        return false
    end
end

# Complex, real and imaginary
#----------------------------
function Base.complex(t::AbstractTensorMap)
    if scalartype(t) <: Complex
        return t
    else
        return copy!(similar(t, complex(scalartype(t))), t)
    end
end
function Base.complex(r::AbstractTensorMap{<:Real}, i::AbstractTensorMap{<:Real})
    return add(r, i, im * one(scalartype(i)))
end

function Base.real(t::AbstractTensorMap)
    if scalartype(t) <: Real
        return t
    else
        tr = similar(t, real(scalartype(t)))
        for (c, b) in blocks(t)
            block(tr, c) .= real(b)
        end
        return tr
    end
end
function Base.imag(t::AbstractTensorMap)
    if scalartype(t) <: Real
        return zerovector(t)
    else
        ti = similar(t, real(scalartype(t)))
        for (c, b) in blocks(t)
            block(ti, c) .= imag(b)
        end
        return ti
    end
end

# Conversion to Array:
#----------------------
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{Array}, t::AbstractTensorMap)
    I = sectortype(t)
    if I === Trivial
        convert(Array, t[])
    else
        cod = codomain(t)
        dom = domain(t)
        T = sectorscalartype(I) <: Complex ? complex(scalartype(t)) :
            sectorscalartype(I) <: Integer ? scalartype(t) : float(scalartype(t))
        A = zeros(T, dims(cod)..., dims(dom)...)
        for (f₁, f₂) in fusiontrees(t)
            F = convert(Array, (f₁, f₂))
            Aslice = StridedView(A)[axes(cod, f₁.uncoupled)..., axes(dom, f₂.uncoupled)...]
            add!(Aslice, StridedView(_kron(convert(Array, t[f₁, f₂]), F)))
        end
        return A
    end
end

# Show and friends
# ----------------

function Base.dims2string(V::HomSpace)
    str_cod = numout(V) == 0 ? "()" : join(dim.(codomain(V)), '×')
    str_dom = numin(V) == 0 ? "()" : join(dim.(domain(V)), '×')
    return str_cod * "←" * str_dom
end

function Base.summary(io::IO, t::AbstractTensorMap)
    V = space(t)
    print(io, Base.dims2string(V), " ")
    Base.showarg(io, t, true)
    return nothing
end

# Human-readable:
function Base.show(io::IO, ::MIME"text/plain", t::AbstractTensorMap)
    # 1) show summary: typically d₁×d₂×… ← d₃×d₄×… $(typeof(t)):
    summary(io, t)
    println(io, ":")

    # 2) show spaces
    # println(io, " space(t):")
    println(io, " codomain: ", codomain(t))
    println(io, " domain: ", domain(t))

    # 3) [optional]: show data
    get(io, :compact, true) && return nothing
    ioc = IOContext(io, :typeinfo => sectortype(t))
    println(io, "\n\n blocks(t):")
    show_blocks(io, MIME"text/plain"(), blocks(t))
    return nothing
end
