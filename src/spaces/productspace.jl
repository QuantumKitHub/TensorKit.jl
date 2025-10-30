"""
    struct ProductSpace{S<:ElementarySpace, N} <: CompositeSpace{S}

A `ProductSpace` is a tensor product space of `N` vector spaces of type
`S<:ElementarySpace`. Only tensor products between [`ElementarySpace`](@ref) objects of the
same type are allowed.
"""
struct ProductSpace{S <: ElementarySpace, N} <: CompositeSpace{S}
    spaces::NTuple{N, S}
    ProductSpace{S, N}(spaces::NTuple{N, S}) where {S <: ElementarySpace, N} = new{S, N}(spaces)
end

function ProductSpace{S, N}(spaces::Vararg{S, N}) where {S <: ElementarySpace, N}
    return ProductSpace{S, N}(spaces)
end

function ProductSpace{S}(spaces::Tuple{Vararg{S}}) where {S <: ElementarySpace}
    return ProductSpace{S, length(spaces)}(spaces)
end
ProductSpace{S}(spaces::Vararg{S}) where {S <: ElementarySpace} = ProductSpace{S}(spaces)

function ProductSpace(spaces::Tuple{S, Vararg{S}}) where {S <: ElementarySpace}
    return ProductSpace{S, length(spaces)}(spaces)
end
function ProductSpace(space1::ElementarySpace, rspaces::Vararg{ElementarySpace})
    return ProductSpace((space1, rspaces...))
end

ProductSpace(P::ProductSpace) = P

# constructors with conversion behaviour
function ProductSpace{S, N}(V::Vararg{ElementarySpace, N}) where {S <: ElementarySpace, N}
    return ProductSpace{S, N}(V)
end
function ProductSpace{S}(V::Vararg{ElementarySpace}) where {S <: ElementarySpace}
    return ProductSpace{S}(V)
end

function ProductSpace{S, N}(V::Tuple{Vararg{ElementarySpace, N}}) where {S <: ElementarySpace, N}
    return ProductSpace{S}(convert.(S, V))
end
function ProductSpace{S}(V::Tuple{Vararg{ElementarySpace}}) where {S <: ElementarySpace}
    return ProductSpace{S}(convert.(S, V))
end
function ProductSpace(V::Tuple{ElementarySpace, Vararg{ElementarySpace}})
    return ProductSpace(promote(V...))
end

# Corresponding methods
#-----------------------
@doc """
    dims(::ProductSpace{S, N}) -> Dims{N} = NTuple{N, Int}
    dims(V::HomSpace) -> Dims{length(V)}
    dims(t::AbstractTensorMap) -> Dims{numind(t)}

Return the dimensions of the spaces in the tensor product space(s) as a tuple of integers.
""" dims

dims(P::ProductSpace) = map(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n])
dim(P::ProductSpace) = prod(dims(P))

Base.axes(P::ProductSpace) = map(axes, P.spaces)
Base.axes(P::ProductSpace, n::Int) = axes(P.spaces[n])

dual(P::ProductSpace{<:ElementarySpace, 0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

# Base.conj(P::ProductSpace) = ProductSpace(map(conj, P.spaces))

function Base.show(io::IO, P::ProductSpace{S}) where {S <: ElementarySpace}
    spaces = P.spaces
    if length(spaces) == 0
        print(io, "ProductSpace{", S, ", 0}")
    end
    if length(spaces) == 1
        print(io, "ProductSpace")
    end
    print(io, "(")
    for i in 1:length(spaces)
        i == 1 || print(io, " ⊗ ")
        show(io, spaces[i])
    end
    return print(io, ")")
end

# more specific methods
"""
    sectors(P::ProductSpace{S, N}) where {S<:ElementarySpace}

Return an iterator over all possible combinations of sectors (represented as an
`NTuple{N, sectortype(S)}`) that can appear within the tensor product space `P`.
"""
sectors(P::ProductSpace) = _sectors(P, sectortype(P))
function _sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{Trivial}) where {N}
    return OneOrNoneIterator(dim(P) != 0, ntuple(n -> Trivial(), N))
end
function _sectors(P::ProductSpace{<:ElementarySpace, N}, ::Type{<:Sector}) where {N}
    return product(map(sectors, P.spaces)...)
end

"""
    hassector(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) where {S<:ElementarySpace}
    -> Bool

Query whether `P` has a non-zero degeneracy of sector `s`, representing a combination of
sectors on the individual tensor indices.
"""
function hassector(V::ProductSpace{<:ElementarySpace, N}, s::NTuple{N}) where {N}
    return reduce(&, map(hassector, V.spaces, s); init = true)
end

"""
    dims(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) where {S<:ElementarySpace}
    -> Dims{N} = NTuple{N, Int}

Return the degeneracy dimensions corresponding to a tuple of sectors `s` for each of the
spaces in the tensor product `P`.
"""
function dims(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N, <:Sector}) where {N}
    return map(dim, P.spaces, sector)
end

"""
    dim(P::ProductSpace{S, N}, s::NTuple{N, sectortype(S)}) where {S<:ElementarySpace}
    -> Int

Return the total degeneracy dimension corresponding to a tuple of sectors for each of the
spaces in the tensor product, obtained as `prod(dims(P, s))``.
"""
function dim(P::ProductSpace{<:ElementarySpace, N}, sector::NTuple{N, <:Sector}) where {N}
    return reduce(*, dims(P, sector); init = 1)
end

function Base.axes(P::ProductSpace{<:ElementarySpace, N}, sectors::NTuple{N, <:Sector}) where {N}
    return map(axes, P.spaces, sectors)
end

"""
    blocksectors(P::ProductSpace)

Return an iterator over the different unique coupled sector labels, i.e. the different
fusion outputs that can be obtained by fusing the sectors present in the different spaces
that make up the `ProductSpace` instance.
"""
function blocksectors(P::ProductSpace{S, N}) where {S, N}
    I = sectortype(S)
    if I == Trivial
        return OneOrNoneIterator(dim(P) != 0, Trivial())
    end
    bs = Vector{I}()
    if N == 0
        push!(bs, unit(I))
    elseif N == 1
        for s in sectors(P)
            push!(bs, first(s))
        end
    else
        for s in sectors(P)
            for c in ⊗(s...)
                if !(c in bs)
                    push!(bs, c)
                end
            end
        end
    end
    return sort!(bs)
end

"""
    fusiontrees(P::ProductSpace, blocksector::Sector)

Return an iterator over all fusion trees that can be formed by fusing the sectors present
in the different spaces that make up the `ProductSpace` instance into the coupled sector
`blocksector`.
"""
function fusiontrees(P::ProductSpace{S, N}, blocksector::I) where {S, N, I}
    I == sectortype(S) || throw(SectorMismatch())
    uncoupled = map(sectors, P.spaces)
    isdualflags = map(isdual, P.spaces)
    return FusionTreeIterator(uncoupled, blocksector, isdualflags)
end

"""
    hasblock(P::ProductSpace, c::Sector)

Query whether a coupled sector `c` appears with nonzero dimension in `P`, i.e. whether
`blockdim(P, c) > 0`.

See also [`blockdim`](@ref) and [`blocksectors`](@ref).
"""
hasblock(P::ProductSpace, c::Sector) = !isempty(fusiontrees(P, c))

"""
    blockdim(P::ProductSpace, c::Sector)

Return the total dimension of a coupled sector `c` in the product space, by summing over
all `dim(P, s)` for all tuples of sectors `s::NTuple{N, <:Sector}` that can fuse to  `c`,
counted with the correct multiplicity (i.e. number of ways in which `s` can fuse to `c`).

See also [`hasblock`](@ref) and [`blocksectors`](@ref).
"""
function blockdim(P::ProductSpace, c::Sector)
    sectortype(P) == typeof(c) || throw(SectorMismatch())
    d = 0
    for f in fusiontrees(P, c)
        d += dim(P, f.uncoupled)
    end
    return d
end

function Base.:(==)(P1::ProductSpace{S, N}, P2::ProductSpace{S, N}) where {S <: ElementarySpace, N}
    return (P1.spaces == P2.spaces)
end
Base.:(==)(P1::ProductSpace, P2::ProductSpace) = false

# hashing S is necessary to have different hashes for empty productspace with different S
Base.hash(P::ProductSpace{S}, h::UInt) where {S} = hash(P.spaces, hash(S, h))

# Default construction from product of spaces
#---------------------------------------------
⊗(V::ElementarySpace, Vrest::ElementarySpace...) = ProductSpace(V, Vrest...)
⊗(P::ProductSpace) = P
function ⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S <: ElementarySpace}
    return ProductSpace{S}(tuple(P1.spaces..., P2.spaces...))
end

# unit element with respect to the monoidal structure of taking tensor products
"""
    one(::S) where {S<:ElementarySpace} -> ProductSpace{S, 0}
    one(::ProductSpace{S}) where {S<:ElementarySpace} -> ProductSpace{S, 0}

Return a tensor product of zero spaces of type `S`, i.e. this is the unit object under the
tensor product operation, such that `V ⊗ one(V) == V`.
"""
Base.one(V::VectorSpace) = one(typeof(V))
Base.one(::Type{<:ProductSpace{S}}) where {S <: ElementarySpace} = ProductSpace{S, 0}(())
Base.one(::Type{S}) where {S <: ElementarySpace} = ProductSpace{S, 0}(())

Base.:^(V::ElementarySpace, N::Int) = ProductSpace{typeof(V), N}(ntuple(n -> V, N))
Base.:^(V::ProductSpace, N::Int) = ⊗(ntuple(n -> V, N)...)
function Base.literal_pow(::typeof(^), V::ElementarySpace, p::Val{N}) where {N}
    return ProductSpace{typeof(V), N}(ntuple(n -> V, p))
end

fuse(P::ProductSpace{S, 0}) where {S <: ElementarySpace} = unitspace(S)
fuse(P::ProductSpace{S}) where {S <: ElementarySpace} = fuse(P.spaces...)

"""
    insertleftunit(P::ProductSpace, i::Int=length(P) + 1; conj=false, dual=false)

Insert a trivial vector space, isomorphic to the underlying field, at position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a left monoidal unit or its dual.

See also [`insertrightunit`](@ref insertrightunit(::ProductSpace, ::Val{i}) where {i}), [`removeunit`](@ref removeunit(::ProductSpace, ::Val{i}) where {i}).
"""
function insertleftunit(
        P::ProductSpace, ::Val{i} = Val(length(P) + 1);
        conj::Bool = false, dual::Bool = false
    ) where {i}
    u = unitspace(spacetype(P))
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i - 1, (u,)))
end

"""
    insertrightunit(P::ProductSpace, i=lenght(P); conj=false, dual=false)

Insert a trivial vector space, isomorphic to the underlying field, after position `i`,
which can be specified as an `Int` or as `Val(i)` for improved type stability.
More specifically, adds a right monoidal unit or its dual.

See also [`insertleftunit`](@ref insertleftunit(::ProductSpace, ::Val{i}) where {i}), [`removeunit`](@ref removeunit(::ProductSpace, ::Val{i}) where {i}).
"""
function insertrightunit(
        P::ProductSpace, ::Val{i} = Val(length(P));
        conj::Bool = false, dual::Bool = false
    ) where {i}
    u = unitspace(spacetype(P))
    if dual
        u = TensorKit.dual(u)
    end
    if conj
        u = TensorKit.conj(u)
    end
    return ProductSpace(TupleTools.insertafter(P.spaces, i, (u,)))
end

"""
    removeunit(P::ProductSpace, i::Int)

This removes a trivial tensor product factor at position `1 ≤ i ≤ N`, where `i`
can be specified as an `Int` or as `Val(i)` for improved type stability.
For this to work, that factor has to be isomorphic to the field of scalars.

This operation undoes the work of [`insertleftunit`](@ref insertleftunit(::ProductSpace, ::Val{i}) where {i}) 
and [`insertrightunit`](@ref insertrightunit(::ProductSpace, ::Val{i}) where {i}).
"""
function removeunit(P::ProductSpace, ::Val{i}) where {i}
    1 ≤ i ≤ length(P) || _boundserror(P, i)
    isisomorphic(P[i], unitspace(P[i])) || _nontrivialspaceerror(P, i)
    return ProductSpace{spacetype(P)}(TupleTools.deleteat(P.spaces, i))
end

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]

Base.iterate(P::ProductSpace, args...) = Base.iterate(P.spaces, args...)
Base.indexed_iterate(P::ProductSpace, args...) = Base.indexed_iterate(P.spaces, args...)

Base.eltype(::Type{<:ProductSpace{S}}) where {S <: ElementarySpace} = S
Base.eltype(P::ProductSpace) = eltype(typeof(P))

Base.IteratorEltype(::Type{<:ProductSpace}) = Base.HasEltype()
Base.IteratorSize(::Type{<:ProductSpace}) = Base.HasLength()

Base.reverse(P::ProductSpace) = ProductSpace(reverse(P.spaces))

# Promotion and conversion
# ------------------------
function Base.promote_rule(::Type{S}, ::Type{<:ProductSpace{S}}) where {S <: ElementarySpace}
    return ProductSpace{S}
end

# ProductSpace to ElementarySpace
Base.convert(::Type{S}, P::ProductSpace{S, 0}) where {S <: ElementarySpace} = unitspace(S)
Base.convert(::Type{S}, P::ProductSpace{S}) where {S <: ElementarySpace} = fuse(P.spaces...)

# ElementarySpace to ProductSpace
Base.convert(::Type{<:ProductSpace}, V::S) where {S <: ElementarySpace} = ⊗(V)
