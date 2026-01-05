# Fusion trees:
#==============================================================================#
"""
    struct FusionTree{I, N, M, L}

Represents a fusion tree of sectors of type `I<:Sector`, fusing (or splitting) `N` uncoupled
sectors to a coupled sector. It actually represents a splitting tree, but fusion tree
is a more common term.

## Fields
- `uncoupled::NTuple{N,I}`: the uncoupled sectors coming out of the splitting tree, before
  the possible 𝑍 isomorphism (see `isdual`).
- `coupled::I`: the coupled sector.
- `isdual::NTuple{N,Bool}`: indicates whether a 𝑍 isomorphism is present (`true`) or not
  (`false`) for each uncoupled sector.
- `innerlines::NTuple{M,I}`: the labels of the `M=max(0, N-2)` inner lines of the splitting
  tree.
- `vertices::NTuple{L,Int}`: the integer values of the `L=max(0, N-1)` vertices of the
  splitting tree. If `FusionStyle(I) isa MultiplicityFreeFusion`, then `vertices` is simply
  equal to the constant value `ntuple(n->1, L)`.
"""
struct FusionTree{I <: Sector, N, M, L}
    uncoupled::NTuple{N, I}
    coupled::I
    isdual::NTuple{N, Bool}
    innerlines::NTuple{M, I} # M = N-2
    vertices::NTuple{L, Int} # L = N-1
    function FusionTree{I, N, M, L}(
            uncoupled::NTuple{N, I},
            coupled::I,
            isdual::NTuple{N, Bool},
            innerlines::NTuple{M, I},
            vertices::NTuple{L, Int}
        ) where
        {I <: Sector, N, M, L}
        # if N == 0
        #     @assert coupled == unit(coupled)
        # elseif N == 1
        #     @assert coupled == uncoupled[1]
        # elseif N == 2
        #     @assert coupled ∈ ⊗(uncoupled...)
        # else
        #     @assert innerlines[1] ∈ ⊗(uncoupled[1], uncoupled[2])
        #     for n = 2:N-2
        #         @assert innerlines[n] ∈ ⊗(innerlines[n-1], uncoupled[n+1])
        #     end
        #     @assert coupled ∈ ⊗(innerlines[N-2], uncoupled[N])
        # end
        return new{I, N, M, L}(uncoupled, coupled, isdual, innerlines, vertices)
    end
end
function FusionTree{I}(
        uncoupled::NTuple{N, Any}, coupled,
        isdual::NTuple{N, Bool}, innerlines,
        vertices = ntuple(n -> 1, max(0, N - 1))
    ) where {I <: Sector, N}
    return if FusionStyle(I) isa GenericFusion
        fusiontreetype(I, N)(
            map(s -> convert(I, s), uncoupled),
            convert(I, coupled), isdual,
            map(s -> convert(I, s), innerlines), vertices
        )
    else
        if all(isone, vertices)
            fusiontreetype(I, N)(
                map(s -> convert(I, s), uncoupled),
                convert(I, coupled), isdual,
                map(s -> convert(I, s), innerlines), vertices
            )
        else
            throw(ArgumentError("Incorrect fusion vertices"))
        end
    end
end
function FusionTree(
        uncoupled::NTuple{N, I}, coupled::I,
        isdual::NTuple{N, Bool}, innerlines,
        vertices = ntuple(n -> 1, max(0, N - 1))
    ) where {I <: Sector, N}
    return if FusionStyle(I) isa GenericFusion
        fusiontreetype(I, N)(uncoupled, coupled, isdual, innerlines, vertices)
    else
        if all(isone, vertices)
            fusiontreetype(I, N)(uncoupled, coupled, isdual, innerlines, vertices)
        else
            throw(ArgumentError("Incorrect fusion vertices"))
        end
    end
end

function FusionTree{I}(
        uncoupled::NTuple{N}, coupled = unit(I), isdual = ntuple(Returns(false), N)
    ) where {I <: Sector, N}
    FusionStyle(I) isa UniqueFusion ||
        throw(ArgumentError("fusion tree requires inner lines if `FusionStyle(I) <: MultipleFusion`"))
    return FusionTree{I}(
        map(s -> convert(I, s), uncoupled), convert(I, coupled), isdual,
        _abelianinner(map(s -> convert(I, s), (uncoupled..., dual(coupled))))
    )
end
function FusionTree(
        uncoupled::NTuple{N, I}, coupled::I, isdual = ntuple(n -> false, length(uncoupled))
    ) where {N, I <: Sector}
    return FusionTree{I}(uncoupled, coupled, isdual)
end
FusionTree(uncoupled::Tuple{I, Vararg{I}}) where {I <: Sector} = FusionTree(uncoupled, unit(I))

"""
    FusionTreePair{I, N₁, N₂}

Type alias for a fusion-splitting tree pair of sectortype `I`, with `N₁` splitting legs and
`N₂` fusion legs.
"""
const FusionTreePair{I, N₁, N₂} = Tuple{FusionTree{I, N₁}, FusionTree{I, N₂}}

"""
    FusionTreeBlock{I, N₁, N₂, F <: FusionTreePair{I, N₁, N₂}}

Collection of fusion-splitting tree pairs that share the same uncoupled sectors.
Mostly internal structure to speed up non-`UniqueFusion` fusiontree manipulation computations.
"""
struct FusionTreeBlock{I, N₁, N₂, F <: FusionTreePair{I, N₁, N₂}}
    trees::Vector{F}
end

function FusionTreeBlock{I}(
        uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}},
        isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}};
        sizehint::Int = 0
    ) where {I <: Sector, N₁, N₂}
    F = fusiontreetype(I, N₁, N₂)
    trees = Vector{F}(undef, 0)
    sizehint > 0 && sizehint!(trees, sizehint)

    if N₁ == N₂ == 0
        f = FusionTree{I}(())
        push!(trees, (f, f))
        return FusionTreeBlock(trees)
    end

    # note: cs is unique due to ⊗ returning unique fusion results
    cs = if N₁ == 0
        sort!(collect(filter(isunit, ⊗(uncoupled[2]...))))
    elseif N₂ == 0
        sort!(collect(filter(isunit, ⊗(uncoupled[1]...))))
    else
        sort!(collect(intersect(⊗(uncoupled[1]...), ⊗(uncoupled[2]...))))
    end

    for c in cs
        for f₁ in fusiontrees(uncoupled[1], c, isdual[1]),
                f₂ in fusiontrees(uncoupled[2], c, isdual[2])

            push!(trees, (f₁, f₂))
        end
    end
    return FusionTreeBlock(trees)
end

# Properties
# ----------
sectortype(::Type{<:FusionTree{I}}) where {I <: Sector} = I
sectortype(::Type{<:FusionTreePair{I}}) where {I <: Sector} = I
sectortype(::Type{<:FusionTreeBlock{I}}) where {I} = I

FusionStyle(f::Union{FusionTree, FusionTreePair, FusionTreeBlock}) =
    FusionStyle(typeof(f))
FusionStyle(::Type{F}) where {F <: Union{FusionTree, FusionTreePair, FusionTreeBlock}} =
    FusionStyle(sectortype(F))

BraidingStyle(f::Union{FusionTree, FusionTreePair, FusionTreeBlock}) =
    BraidingStyle(typeof(f))
BraidingStyle(::Type{F}) where {F <: Union{FusionTree, FusionTreePair, FusionTreeBlock}} =
    BraidingStyle(sectortype(F))

Base.length(f::FusionTree) = length(typeof(f))
Base.length(::Type{<:FusionTree{<:Sector, N}}) where {N} = N
# Note: cannot define the following since FusionTreePair is a const for a Tuple
# Base.length(::Type{<:FusionTreePair{<:Sector, N₁, N₂}}) where {N₁, N₂} = N₁ + N₂
# Base.length(f::FusionTreePair) = length(typeof(f))
Base.length(block::FusionTreeBlock) = length(fusiontrees(block))
Base.isempty(block::FusionTreeBlock) = isempty(fusiontrees(block))

numout(fs::Union{FusionTreePair, FusionTreeBlock}) = numout(typeof(fs))
numout(::Type{<:FusionTreePair{I, N₁}}) where {I, N₁} = N₁
numout(::Type{<:FusionTreeBlock{I, N₁}}) where {I, N₁} = N₁
numin(fs::Union{FusionTreePair, FusionTreeBlock}) = numin(typeof(fs))
numin(::Type{<:FusionTreePair{I, N₁, N₂}}) where {I, N₁, N₂} = N₂
numin(::Type{<:FusionTreeBlock{I, N₁, N₂}}) where {I, N₁, N₂} = N₂
numind(fs::Union{FusionTreePair, FusionTreeBlock}) = numind(typeof(fs))
numind(::Type{T}) where {T <: Union{FusionTreePair, FusionTreeBlock}} = numin(T) + numout(T)

Base.propertynames(::FusionTreeBlock, private::Bool = false) = (:trees, :uncoupled, :isdual)
Base.@constprop :aggressive function Base.getproperty(block::FusionTreeBlock, prop::Symbol)
    if prop === :uncoupled
        f₁, f₂ = first(block.trees)
        return f₁.uncoupled, f₂.uncoupled
    elseif prop === :isdual
        f₁, f₂ = first(block.trees)
        return f₁.isdual, f₂.isdual
    else
        return getfield(block, prop)
    end
end
fusiontrees(block::FusionTreeBlock) = block.trees

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree{I}, h::UInt) where {I}
    h = hash(f.isdual, hash(f.coupled, hash(f.uncoupled, h)))
    if FusionStyle(I) isa MultipleFusion
        h = hash(f.innerlines, h)
    end
    if FusionStyle(I) isa GenericFusion
        h = hash(f.vertices, h)
    end
    return h
end
function Base.:(==)(f₁::FusionTree{I, N}, f₂::FusionTree{I, N}) where {I <: Sector, N}
    f₁.coupled == f₂.coupled || return false
    @inbounds for i in 1:N
        f₁.uncoupled[i] == f₂.uncoupled[i] || return false
        f₁.isdual[i] == f₂.isdual[i] || return false
    end
    if FusionStyle(I) isa MultipleFusion
        @inbounds for i in 1:(N - 2)
            f₁.innerlines[i] == f₂.innerlines[i] || return false
        end
    end
    if FusionStyle(I) isa GenericFusion
        @inbounds for i in 1:(N - 1)
            f₁.vertices[i] == f₂.vertices[i] || return false
        end
    end
    return true
end
Base.:(==)(f₁::FusionTree, f₂::FusionTree) = false

# Within one block, all values of uncoupled and isdual are equal, so avoid hashing these
function treeindex_data((f₁, f₂))
    I = sectortype(f₁)
    if FusionStyle(I) isa GenericFusion
        return (f₁.coupled, f₁.innerlines..., f₂.innerlines...),
            (f₁.vertices..., f₂.vertices...)
    elseif FusionStyle(I) isa MultipleFusion
        return (f₁.coupled, f₁.innerlines..., f₂.innerlines...)
    else # there should be only a single element anyways
        return false
    end
end
function treeindex_map(fs::FusionTreeBlock)
    I = sectortype(fs)
    return fusiontreedict(I)(treeindex_data(f) => ind for (ind, f) in enumerate(fusiontrees(fs)))
end


# Facilitate getting correct fusion tree types
Base.@assume_effects :foldable function fusiontreetype(::Type{I}, N::Int) where {I <: Sector}
    return if N === 0
        FusionTree{I, 0, 0, 0}
    elseif N === 1
        FusionTree{I, 1, 0, 0}
    else
        FusionTree{I, N, N - 2, N - 1}
    end
end
Base.@assume_effects :foldable function fusiontreetype(::Type{I}, N₁::Int, N₂::Int) where {I <: Sector}
    return Tuple{fusiontreetype(I, N₁), fusiontreetype(I, N₂)}
end

fusiontreedict(I) = FusionStyle(I) isa UniqueFusion ? SingletonDict : FusionTreeDict

# converting to actual array
Base.convert(A::Type{<:AbstractArray}, f::FusionTree) = convert(A, fusiontensor(f))
# TODO: is this piracy?
Base.convert(A::Type{<:AbstractArray}, (f₁, f₂)::FusionTreePair) =
    convert(A, fusiontensor((f₁, f₂)))

fusiontensor(::FusionTree{I, 0}) where {I} = fusiontensor(unit(I), unit(I), unit(I))[1, 1, :]
function fusiontensor(f::FusionTree{I, 1}) where {I}
    c = f.coupled
    if f.isdual[1]
        sqrtdc = sqrtdim(c)
        Zcbartranspose = sqrtdc * fusiontensor(dual(c), c, unit(c))[:, :, 1, 1]
        X = conj!(Zcbartranspose) # we want Zcbar^†
    else
        X = fusiontensor(c, unit(c), c)[:, 1, :, 1, 1]
    end
    return X
end
function fusiontensor(f::FusionTree{I, 2}) where {I}
    a, b = f.uncoupled
    isduala, isdualb = f.isdual
    c = f.coupled
    μ = (FusionStyle(I) isa GenericFusion) ? f.vertices[1] : 1
    C = fusiontensor(a, b, c)[:, :, :, μ]
    X = C
    if isduala
        Za = fusiontensor(FusionTree((a,), a, (isduala,), ()))
        @tensor X[a′, b, c] := Za[a′, a] * X[a, b, c]
    end
    if isdualb
        Zb = fusiontensor(FusionTree((b,), b, (isdualb,), ()))
        @tensor X[a, b′, c] := Zb[b′, b] * X[a, b, c]
    end
    return X
end
function fusiontensor(f::FusionTree{I, N}) where {I, N}
    tailout = (f.innerlines[1], TupleTools.tail2(f.uncoupled)...)
    isdualout = (false, TupleTools.tail2(f.isdual)...)
    ftail = FusionTree(tailout, f.coupled, isdualout, Base.tail(f.innerlines), Base.tail(f.vertices))
    Ctail = fusiontensor(ftail)
    f₁ = FusionTree(
        (f.uncoupled[1], f.uncoupled[2]), f.innerlines[1],
        (f.isdual[1], f.isdual[2]), (), (f.vertices[1],)
    )
    C1 = fusiontensor(f₁)
    dtail = size(Ctail)
    d1 = size(C1)
    X = similar(C1, (d1[1], d1[2], Base.tail(dtail)...))
    trivialtuple = ntuple(identity, Val(N))
    return TO.tensorcontract!(
        X,
        C1, ((1, 2), (3,)), false,
        Ctail, ((1,), Base.tail(trivialtuple)), false,
        ((trivialtuple..., N + 1), ())
    )
end

function fusiontensor((f₁, f₂)::FusionTreePair)
    F₁ = fusiontensor(f₁)
    F₂ = fusiontensor(f₂)
    sz1 = size(F₁)
    sz2 = size(F₂)
    d1 = TupleTools.front(sz1)
    d2 = TupleTools.front(sz2)
    return reshape(
        reshape(F₁, TupleTools.prod(d1), sz1[end]) *
            reshape(F₂, TupleTools.prod(d2), sz2[end])', (d1..., d2...)
    )
end
fusiontensor(src::FusionTreeBlock) = sum(fusiontensor, fusiontrees(src))

# Show methods
function Base.show(io::IO, t::FusionTree{I}) where {I <: Sector}
    if FusionStyle(I) isa GenericFusion
        return print(
            IOContext(io, :typeinfo => I), "FusionTree{", type_repr(I), "}(",
            t.uncoupled, ", ", t.coupled, ", ", t.isdual, ", ", t.innerlines, ", ",
            t.vertices, ")"
        )
    else
        return print(
            IOContext(io, :typeinfo => I), "FusionTree{", type_repr(I), "}(",
            t.uncoupled, ", ", t.coupled, ", ", t.isdual, ", ", t.innerlines, ")"
        )
    end
end

# Fusion tree iterators
include("iterator.jl")

# Manipulate fusion trees
include("basic_manipulations.jl")
include("duality_manipulations.jl")
include("braiding_manipulations.jl")

# auxiliary routines
# _abelianinner: generate the inner indices for given outer indices in the abelian case
_abelianinner(outer::Tuple{}) = ()
function _abelianinner(outer::Tuple{I}) where {I <: Sector}
    return isunit(outer[1]) ? () : throw(SectorMismatch())
end
function _abelianinner(outer::Tuple{I, I}) where {I <: Sector}
    return outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
end
function _abelianinner(outer::Tuple{I, I, I}) where {I <: Sector}
    return isunit(first(⊗(outer...))) ? () : throw(SectorMismatch())
end
function _abelianinner(outer::Tuple{I, I, I, I, Vararg{I}}) where {I <: Sector}
    c = first(outer[1] ⊗ outer[2])
    return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
end
