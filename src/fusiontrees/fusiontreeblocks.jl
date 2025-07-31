struct FusionTreeBlock{I, N₁, N₂, F <: FusionTreePair{I, N₁, N₂}}
    trees::Vector{F}
end

function FusionTreeBlock(
        uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}},
        isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}}
    ) where {I <: Sector, N₁, N₂}
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    trees = Vector{Tuple{F₁, F₂}}(undef, 0)

    cleft = N₁ == 0 ? (one(I),) : ⊗(uncoupled[1]...)
    cright = N₂ == 0 ? (one(I),) : ⊗(uncoupled[2]...)
    cs = sort!(collect(intersect(cleft, cright)))
    for c in cs
        for f₁ in fusiontrees(uncoupled[1], c, isdual[1]),
                f₂ in fusiontrees(uncoupled[2], c, isdual[2])

            push!(trees, (f₁, f₂))
        end
    end
    return FusionTreeBlock(trees)
end

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

Base.propertynames(::FusionTreeBlock, private::Bool = false) = (:trees, :uncoupled, :isdual)

sectortype(::Type{<:FusionTreeBlock{I}}) where {I} = I
numout(fs::FusionTreeBlock) = numout(typeof(fs))
numout(::Type{<:FusionTreeBlock{I, N₁}}) where {I, N₁} = N₁
numin(fs::FusionTreeBlock) = numin(typeof(fs))
numin(::Type{<:FusionTreeBlock{I, N₁, N₂}}) where {I, N₁, N₂} = N₂
numind(fs::FusionTreeBlock) = numind(typeof(fs))
numind(::Type{T}) where {T <: FusionTreeBlock} = numin(T) + numout(T)

fusiontrees(block::FusionTreeBlock) = block.trees
Base.length(block::FusionTreeBlock) = length(fusiontrees(block))

# Manipulations
# -------------
function transformation_matrix(transform, dst::FusionTreeBlock{I}, src::FusionTreeBlock{I}) where {I}
    U = zeros(sectorscalartype(I), length(dst), length(src))
    indexmap = Dict(f => ind for (ind, f) in enumerate(fusiontrees(dst)))
    for (col, f) in enumerate(fusiontrees(src))
        for (f′, c) in transform(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end
    return U
end

function bendright(src::FusionTreeBlock)
    uncoupled_dst = (
        TupleTools.front(src.uncoupled[1]),
        (src.uncoupled[2]..., dual(src.uncoupled[1][end])),
    )
    isdual_dst = (
        TupleTools.front(src.isdual[1]),
        (src.isdual[2]..., !(src.isdual[1][end])),
    )
    dst = FusionTreeBlock(uncoupled_dst, isdual_dst)

    U = transformation_matrix(bendright, dst, src)
    return dst, U
end

# TODO: verify if this can be computed through an adjoint
function bendleft(src::FusionTreeBlock)
    uncoupled_dst = (
        (src.uncoupled[1]..., dual(src.uncoupled[2][end])),
        TupleTools.front(src.uncoupled[2]),
    )
    isdual_dst = (
        (src.isdual[1]..., !(src.isdual[2][end])),
        TupleTools.front(src.isdual[2]),
    )
    dst = FusionTreeBlock(uncoupled_dst, isdual_dst)

    U = transformation_matrix(bendleft, dst, src)
    return dst, U
end

function foldright(src::FusionTreeBlock)
    uncoupled_dst = (
        Base.tail(src.uncoupled[1]),
        (dual(first(src.uncoupled[1])), src.uncoupled[2]...),
    )
    isdual_dst = (
        Base.tail(src.isdual[1]),
        (!first(src.isdual[1]), src.isdual[2]...),
    )
    dst = FusionTreeBlock(uncoupled_dst, isdual_dst)

    U = transformation_matrix(foldright, dst, src)
    return dst, U
end

# TODO: verify if this can be computed through an adjoint
function foldleft(src::FusionTreeBlock)
    uncoupled_dst = (
        (dual(first(src.uncoupled[2])), src.uncoupled[1]...),
        Base.tail(src.uncoupled[2]),
    )
    isdual_dst = (
        (!first(src.isdual[2]), src.isdual[1]...),
        Base.tail(src.isdual[2]),
    )
    dst = FusionTreeBlock(uncoupled_dst, isdual_dst)

    U = transformation_matrix(foldleft, dst, src)
    return dst, U
end

function cycleclockwise(src::FusionTreeBlock)
    if numout(src) > 0
        tmp, U₁ = foldright(src)
        dst, U₂ = bendleft(tmp)
    else
        tmp, U₁ = bendleft(src)
        dst, U₂ = foldright(tmp)
    end
    return dst, U₂ * U₁
end

function cycleanticlockwise(src::FusionTreeBlock)
    if numin(src) > 0
        tmp, U₁ = foldleft(src)
        dst, U₂ = bendright(tmp)
    else
        tmp, U₁ = bendright(src)
        dst, U₂ = foldleft(tmp)
    end
    return dst, U₂ * U₁
end

@inline function repartition(src::FusionTreeBlock, N::Int)
    @assert 0 <= N <= numind(src)
    return _recursive_repartition(src, Val(N))
end

function _repartition_type(I, N, N₁, N₂)
    return Tuple{FusionTreeBlock{I, N, N₁ + N₂ - N}, Matrix{sectorscalartype(I)}}
end
function _recursive_repartition(
        src::FusionTreeBlock{I, N₁, N₂},
        ::Val{N}
    )::_repartition_type(I, N, N₁, N₂) where {I, N₁, N₂, N}
    if N == N₁
        dst = src
        U = zeros(sectorscalartype(I), length(dst), length(src))
        copyto!(U, LinearAlgebra.I)
        return dst, U
    end

    N == N₁ - 1 && return bendright(src)
    N == N₁ + 1 && return bendleft(src)

    tmp, U₁ = N < N₁ ? bendright(src) : bendleft(src)
    dst, U₂ = _recursive_repartition(tmp, Val(N))
    return dst, U₂ * U₁
end

function Base.transpose(src::FusionTreeBlock, p::Index2Tuple{N₁, N₂}) where {N₁, N₂}
    N = N₁ + N₂
    @assert numind(src) == N
    p′ = linearizepermutation(p..., numout(src), numin(src))
    @assert iscyclicpermutation(p′)
    return _fstranspose((src, p))
end

const _FSTransposeKey{I, N₁, N₂} = Tuple{<:FusionTreeBlock{I}, Index2Tuple{N₁, N₂}}

@cached function _fstranspose(
        key::_FSTransposeKey{I, N₁, N₂}
    )::Tuple{FusionTreeBlock{I, N₁, N₂}, Matrix{sectorscalartype(I)}} where {I, N₁, N₂}
    src, (p1, p2) = key

    N = N₁ + N₂
    p = linearizepermutation(p1, p2, numout(src), numin(src))

    dst, U = repartition(src, N₁)
    length(p) == 0 && return dst, U
    i1 = findfirst(==(1), p)::Int
    i1 == 1 && return dst, U

    Nhalf = N >> 1
    while 1 < i1 ≤ Nhalf
        dst, U_tmp = cycleanticlockwise(dst)
        U = U_tmp * U
        i1 -= 1
    end
    while Nhalf < i1
        dst, U_tmp = cycleclockwise(dst)
        U = U_tmp * U
        i1 = mod1(i1 + 1, N)
    end

    return dst, U
end

function CacheStyle(::typeof(_fstranspose), k::_FSTransposeKey{I}) where {I}
    if FusionStyle(I) == UniqueFusion()
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

function artin_braid(src::FusionTreeBlock{I, N, 0}, i; inv::Bool = false) where {I, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))

    uncoupled = src.uncoupled[1]
    uncoupled′ = TupleTools.setindex(uncoupled, uncoupled[i + 1], i)
    uncoupled′ = TupleTools.setindex(uncoupled′, uncoupled[i], i + 1)
    isdual = src.isdual[1]
    isdual′ = TupleTools.setindex(isdual, isdual[i], i + 1)
    isdual′ = TupleTools.setindex(isdual′, isdual[i + 1], i)
    dst = FusionTreeBlock((uncoupled′, ()), (isdual′, ()))

    # TODO: do we want to rewrite `artin_braid` to take double trees instead?
    U = transformation_matrix(dst, src) do (f₁, f₂)
        return ((f₁′, f₂) => c for (f₁′, c) in artin_braid(f₁, i; inv))
    end
    return dst, U
end

function braid(src::FusionTreeBlock{I, N, 0}, p::NTuple{N, Int}, levels::NTuple{N, Int}) where {I, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))

    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        uncoupled′ = TupleTools._permute(src.uncoupled[1], p)
        isdual′ = TupleTools._permute(src.isdual[1], p)
        dst = FusionTreeBlock(uncoupled′, isdual′)
        U = transformation_matrix(dst, src) do (f₁, f₂)
            return ((f₁′, f₂) => c for (f₁, c) in braid(f₁, p, levels))
        end
    else
        dst, U = repartition(src, N) # TODO: can we avoid this?
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            dst, U_tmp = artin_braid(dst, s; inv)
            U = U_tmp * U
        end
    end
    return dst, U
end

function braid(src::FusionTreeBlock{I}, p::Index2Tuple{N₁, N₂}, levels::Index2Tuple) where {I, N₁, N₂}
    @assert numind(src) == N₁ + N₂
    @assert numout(src) == length(levels[1]) && numin(src) == length(levels[2])
    @assert TupleTools.isperm((p[1]..., p[2]...))
    return _fsbraid((src, p, levels))
end

const _FSBraidKey{I, N₁, N₂} = Tuple{<:FusionTreeBlock{I}, Index2Tuple{N₁, N₂}, Index2Tuple}

@cached function _fsbraid(
        key::_FSBraidKey{I, N₁, N₂}
    )::Tuple{FusionTreeBlock{I, N₁, N₂}, Matrix{sectorscalartype(I)}} where {I, N₁, N₂}
    src, (p1, p2), (l1, l2) = key

    p = linearizepermutation(p1, p2, numout(src), numin(src))
    levels = (l1..., reverse(l2)...)

    dst, U = repartition(src, numind(src))

    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        uncoupled′ = TupleTools._permute(dst.uncoupled[1], p)
        isdual′ = TupleTools._permute(dst.isdual[1], p)

        dst′ = FusionTreeBlock(uncoupled′, isdual′)
        U_tmp = transformation_matrix(dst′, dst) do (f₁, f₂)
            return ((f₁′, f₂) => c for (f₁, c) in braid(f₁, p, levels))
        end
        dst = dst′
        U = U_tmp * U
    else
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            dst, U_tmp = artin_braid(dst, s; inv)
            U = U_tmp * U
        end
    end

    if N₂ == 0
        return dst, U
    else
        dst, U_tmp = repartition(dst, N₁)
        U = U_tmp * U
        return dst, U
    end
end

function CacheStyle(::typeof(_fsbraid), k::_FSBraidKey{I}) where {I}
    if FusionStyle(I) isa UniqueFusion
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

function permute(src::FusionTreeBlock{I}, p::Index2Tuple) where {I}
    @assert BraidingStyle(I) isa SymmetricBraiding
    levels1 = ntuple(identity, numout(src))
    levels2 = numout(src) .+ ntuple(identity, numin(src))
    return braid(src, p, (levels1, levels2))
end
