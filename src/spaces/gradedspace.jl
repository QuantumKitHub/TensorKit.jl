"""
    struct GradedSpace{I<:Sector, D} <: ElementarySpace
    GradedSpace{I,D}(dims; dual::Bool = false) where {I<:Sector, D}

A complex Euclidean space with a grading, i.e. a direct sum structure corresponding
to labels in a set `I`, the objects of which have the structure of a monoid with
respect to a monoidal product `⊗`. In practice, we restrict the label set to be a set
of superselection sectors of type `I<:Sector`, e.g. the set of distinct irreps of a
finite or compact group, or the isomorphism classes of simple objects of a unitary
and pivotal (pre-, multi-) fusion category.

Here `dims` represents the degeneracy or multiplicity of every sector.

The data structure `D` of `dims` will depend on the result `Base.IteratorSize(values(I))`.
If the result is of type `HasLength` or `HasShape`, `dims` will be stored in a
`NTuple{N,Int}` with `N = length(values(I))`. This requires that a sector `s::I` can be
transformed into an index via `s == getindex(values(I), i)` and
`i == findindex(values(I), s)`. If `Base.IteratorElsize(values(I))` results `IsInfinite()`
or `SizeUnknown()`, a `SectorDict{I,Int}` is used to store the non-zero degeneracy
dimensions with the corresponding sector as key. The parameter `D` is hidden from the user
and should typically be of no concern.

The concrete type `GradedSpace{I,D}` with correct `D` can be obtained as `Vect[I]`, or if
`I == Irrep[G]` for some `G<:Group`, as `Rep[G]`.
"""
struct GradedSpace{I <: Sector, D} <: ElementarySpace
    dims::D
    dual::Bool
end
sectortype(::Type{<:GradedSpace{I}}) where {I <: Sector} = I

function GradedSpace{I, NTuple{N, Int}}(dims; dual::Bool = false) where {I, N}
    d = zeros(Int, N)
    isset = falses(N) # see if this is still needed if we're restricting to small N
    for (c, dc) in dims
        k = convert(I, c)
        i = findindex(values(I), k)
        dc < 0 && throw(ArgumentError(lazy"Sector $k has negative dimension $dc"))
        isset[i] && throw(ArgumentError(lazy"Sector $c appears multiple times"))
        isset[i] = true
        d[i] = dc
    end
    return GradedSpace{I, NTuple{N, Int}}(NTuple{N, Int}(d), dual)
end
function GradedSpace{I, NTuple{N, Int}}(dims::Pair; dual::Bool = false) where {I, N}
    return GradedSpace{I, NTuple{N, Int}}((dims,); dual = dual)
end

function GradedSpace{I, SectorDict{I, Int}}(dims; dual::Bool = false) where {I <: Sector}
    d = SectorDict{I, Int}()
    for (c, dc) in dims
        k = convert(I, c)
        haskey(d, k) && throw(ArgumentError(lazy"Sector $k appears multiple times"))
        dc < 0 && throw(ArgumentError(lazy"Sector $k has negative dimension $dc"))
        !iszero(dc) && push!(d, k => dc)
    end
    return GradedSpace{I, SectorDict{I, Int}}(d, dual)
end
function GradedSpace{I, SectorDict{I, Int}}(dims::Pair; dual::Bool = false) where {I <: Sector}
    return GradedSpace{I, SectorDict{I, Int}}((dims,); dual = dual)
end

GradedSpace{I, D}(; kwargs...) where {I <: Sector, D} = GradedSpace{I, D}((); kwargs...)
function GradedSpace{I, D}(d1::Pair, d2::Pair, dims::Vararg{Pair}; kwargs...) where {I <: Sector, D}
    return GradedSpace{I, D}((d1, d2, dims...); kwargs...)
end

GradedSpace{I}(args...; kwargs...) where {I <: Sector} = Vect[I](args...; kwargs...)

function GradedSpace(
        dims::Tuple{Pair{I, <:Integer}, Vararg{Pair{I, <:Integer}}}; dual::Bool = false
    ) where {I <: Sector}
    return Vect[I](dims; dual = dual)
end
function GradedSpace(
        dim1::Pair{I, <:Integer}, rdims::Vararg{Pair{I, <:Integer}}; dual::Bool = false
    ) where {I <: Sector}
    return Vect[I]((dim1, rdims...); dual = dual)
end
function GradedSpace(dims::AbstractDict{I, <:Integer}; dual::Bool = false) where {I <: Sector}
    return Vect[I](dims; dual = dual)
end
# not inferrable
GradedSpace(g::Base.Generator; dual::Bool = false) = GradedSpace(g...; dual = dual)
GradedSpace(g::AbstractDict; dual::Bool = false) = GradedSpace(g...; dual = dual)

# Corresponding methods:
#------------------------
field(::Type{<:GradedSpace}) = ℂ
InnerProductStyle(::Type{<:GradedSpace}) = EuclideanInnerProduct()

function dim(V::GradedSpace{I}) where {I <: Sector}
    init = zero(dimscalartype(I))
    return sum(((c, d),) -> dim(c) * d, blockdims(V); init)
end
function dim(V::GradedSpace{I, <:AbstractDict}, c::I) where {I <: Sector}
    return get(V.dims, isdual(V) ? dual(c) : c, 0)
end
function dim(V::GradedSpace{I, <:Tuple}, c::I) where {I <: Sector}
    return V.dims[findindex(values(I), isdual(V) ? dual(c) : c)]
end
Base.axes(V::GradedSpace) = Base.OneTo(dim(V))
function Base.axes(V::GradedSpace{I}, c::I) where {I <: Sector}
    offset = 0
    for (c′, d′) in blockdims(V)
        c′ == c && break
        offset += dim(c′) * d′
    end
    return (offset + 1):(offset + dim(c) * dim(V, c))
end

dual(V::GradedSpace) = typeof(V)(V.dims, !V.dual)
Base.conj(V::GradedSpace) = dual(V)
isdual(V::GradedSpace) = V.dual
isconj(V::GradedSpace) = isdual(V)
function flip(V::GradedSpace{I}) where {I <: Sector}
    return if isdual(V)
        typeof(V)(blockdims(V))
    else
        typeof(V)(dual(c) => d for (c, d) in blockdims(V))'
    end
end

function unitspace(S::Type{<:GradedSpace{I}}) where {I <: Sector}
    return S(unit => 1 for unit in allunits(I))
end
zerospace(S::Type{<:GradedSpace}) = S()

function ⊕(V₁::GradedSpace{I, <:SectorDict}, V₂::GradedSpace{I, <:SectorDict}) where {I <: Sector}
    dual1 = isdual(V₁)
    dual1 == isdual(V₂) || throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    return typeof(V₁)(mergewith(+, V₁.dims, V₂.dims), dual1)
end
function ⊕(V₁::GradedSpace{I, <:Tuple}, V₂::GradedSpace{I, <:Tuple}) where {I <: Sector}
    dual1 = isdual(V₁)
    dual1 == isdual(V₂) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    newdims = map(+, V₁.dims, V₂.dims)
    return typeof(V₁)(newdims, dual1)
end
function ⊖(V::GradedSpace{I, <:Tuple}, W::GradedSpace{I, <:Tuple}) where {I <: Sector}
    dualV = isdual(V)
    V ≿ W && dualV == isdual(W) || throw(SpaceMismatch("$(W) is not a subspace of $(V)"))
    newdims = map(-, V.dims, W.dims)
    return typeof(V)(newdims, dualV)
end
function ⊖(V::GradedSpace{I, <:SectorDict}, W::GradedSpace{I, <:SectorDict}) where {I <: Sector}
    dualV = isdual(V)
    V ≿ W && dualV == isdual(W) || throw(SpaceMismatch("$(W) is not a subspace of $(V)"))
    return typeof(V)(mergewith(-, V.dims, W.dims), dualV)
end

function fuse(V₁::GradedSpace{I, <:SectorDict}, V₂::GradedSpace{I, <:SectorDict}) where {I <: Sector}
    acc = Dict{I, Int}() # SectorDict `get` within the double for loop accumulates O(N^2) `findindex` calls -> sort afterwards
    for (a, da) in blockdims(V₁), (b, db) in blockdims(V₂)
        dab = da * db
        for c in a ⊗ b
            acc[c] = get(acc, c, 0) + Nsymbol(a, b, c) * dab
        end
    end
    ks0 = collect(keys(acc))
    vs0 = collect(values(acc))
    perm = sortperm(ks0)
    return typeof(V₁)(SectorDict{I, Int}(ks0[perm], vs0[perm]), false)
end
function fuse(V₁::GradedSpace{I, NTuple{N, Int}}, V₂::GradedSpace{I, NTuple{N, Int}}) where {I <: Sector, N}
    vals = values(I)
    dual1, dual2 = isdual(V₁), isdual(V₂)
    newdims = zeros(Int, N)
    @inbounds for na in 1:N
        da = V₁.dims[na]
        iszero(da) && continue
        a₀ = vals[na] # avoid call to sectors(V₁)
        a = dual1 ? dual(a₀) : a₀
        for nb in 1:N
            db = V₂.dims[nb]
            iszero(db) && continue
            b₀ = vals[nb] # idem for V₂
            b = dual2 ? dual(b₀) : b₀
            dab = da * db
            for c in a ⊗ b
                nc = findindex(vals, c)
                newdims[nc] += Nsymbol(a, b, c) * dab
            end
        end
    end
    return typeof(V₁)(NTuple{N, Int}(newdims), false)
end

function infimum(V₁::GradedSpace{I, <:Tuple}, V₂::GradedSpace{I, <:Tuple}) where {I <: Sector}
    Visdual = isdual(V₁)
    Visdual == isdual(V₂) || throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    newdims = map(min, V₁.dims, V₂.dims)
    return typeof(V₁)(newdims, Visdual)
end
function infimum(V₁::GradedSpace{I, <:SectorDict}, V₂::GradedSpace{I, <:SectorDict}) where {I <: Sector}
    Visdual = isdual(V₁)
    Visdual == isdual(V₂) || throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    return typeof(V₁)(_sortedintersect(min, V₁.dims, V₂.dims), Visdual)
end
function supremum(V₁::GradedSpace{I, <:Tuple}, V₂::GradedSpace{I, <:Tuple}) where {I <: Sector}
    Visdual = isdual(V₁)
    Visdual == isdual(V₂) || throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    newdims = map(max, V₁.dims, V₂.dims)
    return typeof(V₁)(newdims, Visdual)
end
function supremum(V₁::GradedSpace{I, <:SectorDict}, V₂::GradedSpace{I, <:SectorDict}) where {I <: Sector}
    Visdual = isdual(V₁)
    Visdual == isdual(V₂) || throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    return typeof(V₁)(mergewith(max, V₁.dims, V₂.dims), Visdual)
end

hassector(V::GradedSpace{I}, s::I) where {I <: Sector} = dim(V, s) != 0
function sectors(V::GradedSpace{I, <:AbstractDict}) where {I <: Sector}
    return SectorSet{I}(s -> isdual(V) ? dual(s) : s, keys(V.dims))
end
function sectors(V::GradedSpace{I, NTuple{N, Int}}) where {I <: Sector, N}
    return SectorSet{I}(Iterators.filter(n -> V.dims[n] != 0, 1:N)) do n
        return isdual(V) ? dual(values(I)[n]) : values(I)[n]
    end
end

"""
    blockdims(V::GradedSpace)

Return an iterator over the non-zero blocks of the graded space `V`.
These blocks contain the `Sector`s and their corresponding degeneracy,
i.e. the number of times the sector appears in the direct sum decomposition of `V`.
"""
function blockdims(V::GradedSpace{I, <:AbstractDict}) where {I <: Sector}
    return ((isdual(V) ? dual(c) : c) => d for (c, d) in V.dims)
end
function blockdims(V::GradedSpace{I, NTuple{N, Int}}) where {I <: Sector, N}
    vals = values(I)
    return (
        (isdual(V) ? dual(vals[n]) : vals[n]) => V.dims[n]
            for n in 1:N if !iszero(V.dims[n])
    )
end

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))
function Base.:(==)(V₁::GradedSpace, V₂::GradedSpace)
    return sectortype(V₁) == sectortype(V₂) && (V₁.dims == V₂.dims) && V₁.dual == V₂.dual
end

function sectorhash(V::GradedSpace{I, NTuple{N, Int}}, h::UInt) where {I, N}
    return hash(iszero.(V.dims), hash(isdual(V), h))
end
function sectorequal(V₁::GradedSpace{I, D}, V₂::GradedSpace{I, D}) where {I, N, D <: NTuple{N, Int}}
    return isdual(V₁) == isdual(V₂) && all(zip(V₁.dims, V₂.dims)) do (d₁, d₂)
        return iszero(d₁) == iszero(d₂)
    end
end
function sectorhash(V::GradedSpace{I, <:SectorDict}, h::UInt) where {I}
    return hash(keys(V.dims), hash(isdual(V), h))
end
function sectorequal(V₁::GradedSpace{I, D}, V₂::GradedSpace{I, D}) where {I, D <: SectorDict}
    return isdual(V₁) == isdual(V₂) && keys(V₁.dims) == keys(V₂.dims)
end

Base.summary(io::IO, V::GradedSpace) = print(io, type_repr(typeof(V)))

function Base.show(io::IO, V::GradedSpace)
    opn = (get(io, :typeinfo, Any)::Type == typeof(V) ? "" : type_repr(typeof(V)))
    opn *= "("
    if isdual(V)
        cls = ")'"
        V = dual(V)
    else
        cls = ")"
    end

    v = collect(blockdims(V))

    # logic stolen from Base.show_vector
    limited = get(io, :limit, false)::Bool
    io = IOContext(io, :typeinfo => eltype(v))

    if limited && length(v) > 20
        axs1 = axes(v, 1)
        f, l = first(axs1), last(axs1)
        Base.show_delim_array(io, v, opn, ",", "", false, f, f + 9)
        print(io, "  …  ")
        Base.show_delim_array(io, v, "", ",", cls, false, l - 9, l)
    else
        Base.show_delim_array(io, v, opn, ",", cls, false)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", V::GradedSpace)
    # print small summary, e.g.: Vect[I](…) of dim d
    d = dim(V)
    print(io, type_repr(typeof(V)), "(…)")
    isdual(V) && print(io, "'")
    print(io, " of dim ", d)

    compact = get(io, :compact, false)::Bool
    (iszero(d) || compact) && return nothing

    # print detailed sector information - hijack Base.Vector printing
    print(io, ":\n")
    isdual(V) && (V = dual(V))
    print_data = collect(blockdims(V))
    ioc = IOContext(io, :typeinfo => eltype(print_data))
    Base.print_matrix(ioc, print_data)

    return nothing
end

struct SpaceTable end
"""
    const Vect

A constant of a singleton type used as `Vect[I]` with `I<:Sector` a type of sector, to
construct or obtain the concrete type `GradedSpace{I,D}` instances without having to
specify `D`.
"""
const Vect = SpaceTable()
Base.getindex(::SpaceTable) = ComplexSpace
Base.getindex(::SpaceTable, ::Type{Trivial}) = ComplexSpace
Base.getindex(::SpaceTable, I::Type{<:Sector}) = GradedSpace{I, sectorstoragetype(I)}

# based on Julia tuple unrolling range
const _ntuple_storage_threshold = 32

"""
    sectorstoragetype(I::Type{<:Sector}) -> Type

The storage type `D` used for the `dims` field of `GradedSpace{I, D}`.
This is `NTuple{N,Int}` with `N = length(values(I))` if `I` has a finite, known length
of at most `$_ntuple_storage_threshold`, or `SectorDict{I,Int}` otherwise.
"""
Base.@assume_effects :foldable function sectorstoragetype(::Type{I}) where {I <: Sector}
    if Base.IteratorSize(values(I)) isa Union{HasLength, HasShape}
        N = length(values(I))
        N <= _ntuple_storage_threshold && return NTuple{N, Int}
    end
    return SectorDict{I, Int}
end

Base.getindex(::ComplexNumbers, I::Type{<:Sector}) = Vect[I]
struct RepTable end
"""
    const Rep

A constant of a singleton type used as `Rep[G]` with `G<:Group` a type of group, to
construct or obtain the concrete type `GradedSpace{Irrep[G],D}` instances without having to
specify `D`. Note that `Rep[G] == Vect[Irrep[G]]`.

See also [`Irrep`](@ref) and [`Vect`](@ref).
"""
const Rep = RepTable()
Base.getindex(::RepTable, G::Type{<:Group}) = Vect[Irrep[G]]

type_repr(::Type{<:GradedSpace{I}}) where {I <: Sector} = "Vect[" * type_repr(I) * "]"
function type_repr(::Type{<:GradedSpace{<:AbstractIrrep{G}}}) where {G <: Group}
    return "Rep[" * type_repr(G) * "]"
end
function type_repr(::Type{<:GradedSpace{ProductSector{T}}}) where
    {T <: Tuple{Vararg{AbstractIrrep}}}
    sectors = T.parameters
    s = "Rep["
    for i in 1:length(sectors)
        if i != 1
            s *= " × "
        end
        s *= type_repr(supertype(sectors[i]).parameters[1])
    end
    s *= "]"
    return s
end

# Specific constructors for Z_N
const ZNSpace{N} = GradedSpace{ZNIrrep{N}, NTuple{N, Int}}
ZNSpace{N}(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace{N}(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::NTuple{N, Int}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)
ZNSpace(dims::Vararg{Int, N}; dual::Bool = false) where {N} = ZNSpace{N}(dims, dual)

# TODO: Do we still need all of those
# ASCII type aliases
const Z2Space = ZNSpace{2}
const Z3Space = ZNSpace{3}
const Z4Space = ZNSpace{4}
const U1Space = Rep[U₁]
const CU1Space = Rep[CU₁]
const SU2Space = Rep[SU₂]

# Unicode alternatives
const ℤ₂Space = Z2Space
const ℤ₃Space = Z3Space
const ℤ₄Space = Z4Space
const U₁Space = U1Space
const CU₁Space = CU1Space
const SU₂Space = SU2Space
