struct OuterTreeIterator{I <: Sector, N₁, N₂}
    uncoupled::Tuple{NTuple{N₁, I}, NTuple{N₂, I}}
    isdual::Tuple{NTuple{N₁, Bool}, NTuple{N₂, Bool}}
end

sectortype(::Type{<:OuterTreeIterator{I}}) where {I} = I
numout(fs::OuterTreeIterator) = numout(typeof(fs))
numout(::Type{<:OuterTreeIterator{I, N₁}}) where {I, N₁} = N₁
numin(fs::OuterTreeIterator) = numin(typeof(fs))
numin(::Type{<:OuterTreeIterator{I, N₁, N₂}}) where {I, N₁, N₂} = N₂
numind(fs::OuterTreeIterator) = numind(typeof(fs))
numind(::Type{T}) where {T <: OuterTreeIterator} = numin(T) + numout(T)

# TODO: should we make this an actual iterator?
function fusiontrees(iter::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)

    trees = Vector{Tuple{F₁, F₂}}(undef, 0)
    for c in blocksectors(iter), f₁ in fusiontrees(iter.uncoupled[1], c, iter.isdual[1]),
            f₂ in fusiontrees(iter.uncoupled[2], c, iter.isdual[2])

        push!(trees, (f₁, f₂))
    end
    return trees
end

# TODO: better implementation
Base.length(iter::OuterTreeIterator) = length(fusiontrees(iter))

function blocksectors(iter::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    I == Trivial && return (Trivial(),)

    bs_codomain = Vector{I}()
    if N₁ == 0
        push!(bs_codomain, one(I))
    elseif N₁ == 1
        push!(bs_codomain, only(iter.uncoupled[1]))
    else
        for c in ⊗(iter.uncoupled[1]...)
            if !(c in bs_codomain)
                push!(bs_codomain, c)
            end
        end
    end

    bs_domain = Vector{I}()
    if N₂ == 0
        push!(bs_domain, one(I))
    elseif N₂ == 1
        push!(bs_domain, only(iter.uncoupled[2]))
    else
        for c in ⊗(iter.uncoupled[2]...)
            if !(c in bs_domain)
                push!(bs_domain, c)
            end
        end
    end

    return sort!(collect(intersect(bs_codomain, bs_domain)))
end

# Manipulations
# -------------

function bendright(fs_src::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    uncoupled_dst = (
        TupleTools.front(fs_src.uncoupled[1]),
        (fs_src.uncoupled[2]..., dual(fs_src.uncoupled[1][end])),
    )
    isdual_dst = (
        TupleTools.front(fs_src.isdual[1]),
        (fs_src.isdual[2]..., !(fs_src.isdual[1][end])),
    )
    fs_dst = OuterTreeIterator(uncoupled_dst, isdual_dst)

    trees_src = fusiontrees(fs_src)
    trees_dst = fusiontrees(fs_dst)
    indexmap = Dict(f => ind for (ind, f) in enumerate(trees_dst))
    U = zeros(sectorscalartype(I), length(trees_dst), length(trees_src))

    for (col, f) in enumerate(trees_src)
        for (f′, c) in bendright(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end

    return fs_dst, U
end

# TODO: verify if this can be computed through an adjoint
function bendleft(fs_src::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    uncoupled_dst = (
        (fs_src.uncoupled[1]..., dual(fs_src.uncoupled[2][end])),
        TupleTools.front(fs_src.uncoupled[2]),
    )
    isdual_dst = (
        (fs_src.isdual[1]..., !(fs_src.isdual[2][end])),
        TupleTools.front(fs_src.isdual[2]),
    )
    fs_dst = OuterTreeIterator(uncoupled_dst, isdual_dst)

    trees_src = fusiontrees(fs_src)
    trees_dst = fusiontrees(fs_dst)
    indexmap = Dict(f => ind for (ind, f) in enumerate(trees_dst))
    U = zeros(sectorscalartype(I), length(trees_dst), length(trees_src))

    for (col, f) in enumerate(trees_src)
        for (f′, c) in bendleft(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end

    return fs_dst, U
end

function foldright(fs_src::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    uncoupled_dst = (
        Base.tail(fs_src.uncoupled[1]),
        (dual(first(fs_src.uncoupled[1])), fs_src.uncoupled[2]...),
    )
    isdual_dst = (
        Base.tail(fs_src.isdual[1]),
        (!first(fs_src.isdual[1]), fs_src.isdual[2]...),
    )
    fs_dst = OuterTreeIterator(uncoupled_dst, isdual_dst)

    trees_src = fusiontrees(fs_src)
    trees_dst = fusiontrees(fs_dst)
    indexmap = Dict(f => ind for (ind, f) in enumerate(trees_dst))
    U = zeros(sectorscalartype(I), length(trees_dst), length(trees_src))

    for (col, f) in enumerate(trees_src)
        for (f′, c) in foldright(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end

    return fs_dst, U
end

# TODO: verify if this can be computed through an adjoint
function foldleft(fs_src::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    uncoupled_dst = (
        (dual(first(fs_src.uncoupled[2])), fs_src.uncoupled[1]...),
        Base.tail(fs_src.uncoupled[2]),
    )
    isdual_dst = (
        (!first(fs_src.isdual[2]), fs_src.isdual[1]...),
        Base.tail(fs_src.isdual[2]),
    )
    fs_dst = OuterTreeIterator(uncoupled_dst, isdual_dst)

    trees_src = fusiontrees(fs_src)
    trees_dst = fusiontrees(fs_dst)
    indexmap = Dict(f => ind for (ind, f) in enumerate(trees_dst))
    U = zeros(sectorscalartype(I), length(trees_dst), length(trees_src))

    for (col, f) in enumerate(trees_src)
        for (f′, c) in foldleft(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end

    return fs_dst, U
end

function cycleclockwise(fs_src::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    if N₁ > 0
        fs_tmp, U₁ = foldright(fs_src)
        fs_dst, U₂ = bendleft(fs_tmp)
    else
        fs_tmp, U₁ = bendleft(fs_src)
        fs_dst, U₂ = foldright(fs_tmp)
    end
    return fs_dst, U₂ * U₁
end

function cycleanticlockwise(fs_src::OuterTreeIterator{I, N₁, N₂}) where {I, N₁, N₂}
    if N₂ > 0
        fs_tmp, U₁ = foldleft(fs_src)
        fs_dst, U₂ = bendright(fs_tmp)
    else
        fs_tmp, U₁ = bendright(fs_src)
        fs_dst, U₂ = foldleft(fs_tmp)
    end
    return fs_dst, U₂ * U₁
end

@inline function repartition(fs_src::OuterTreeIterator{I, N₁, N₂}, N::Int) where {I, N₁, N₂}
    @assert 0 <= N <= N₁ + N₂
    return _recursive_repartition(fs_src, Val(N))
end

function _repartition_type(I, N, N₁, N₂)
    return Tuple{OuterTreeIterator{I, N, N₁ + N₂ - N}, Matrix{sectorscalartype(I)}}
end
function _recursive_repartition(
        fs_src::OuterTreeIterator{I, N₁, N₂}, ::Val{N}
    )::_repartition_type(I, N, N₁, N₂) where {I, N₁, N₂, N}
    if N == N₁
        fs_dst = fs_src
        U = zeros(sectorscalartype(I), length(fs_dst), length(fs_src))
        copyto!(U, LinearAlgebra.I)
        return fs_dst, U
    end

    N == N₁ - 1 && return bendright(fs_src)
    N == N₁ + 1 && return bendleft(fs_src)

    fs_tmp, U₁ = N < N₁ ? bendright(fs_src) : bendleft(fs_src)
    fs_dst, U₂ = _recursive_repartition(fs_tmp, Val(N))
    return fs_dst, U₂ * U₁
end

function Base.transpose(fs_src::OuterTreeIterator{I}, p::Index2Tuple{N₁, N₂}) where {I, N₁, N₂}
    N = N₁ + N₂
    @assert numind(fs_src) == N
    p′ = linearizepermutation(p..., numout(fs_src), numin(fs_src))
    @assert iscyclicpermutation(p′)
    return _fstranspose((fs_src, p))
end

const _FSTransposeKey{I, N₁, N₂} = Tuple{<:OuterTreeIterator{I}, Index2Tuple{N₁, N₂}}

@cached function _fstranspose(
        key::_FSTransposeKey{I, N₁, N₂}
    )::Tuple{OuterTreeIterator{I, N₁, N₂}, Matrix{sectorscalartype(I)}} where {I, N₁, N₂}
    fs_src, (p1, p2) = key

    N = N₁ + N₂
    p = linearizepermutation(p1, p2, numout(fs_src), numin(fs_src))

    fs_dst, U = repartition(fs_src, N₁)
    length(p) == 0 && return fs_dst, U
    i1 = findfirst(==(1), p)::Int
    i1 == 1 && return fs_dst, U

    Nhalf = N >> 1
    while 1 < i1 ≤ Nhalf
        fs_dst, U_tmp = cycleanticlockwise(fs_dst)
        U = U_tmp * U
        i1 -= 1
    end
    while Nhalf < i1
        fs_dst, U_tmp = cycleclockwise(fs_dst)
        U = U_tmp * U
        i1 = mod1(i1 + 1, N)
    end

    return fs_dst, U
end

function CacheStyle(::typeof(_fstranspose), k::_FSTransposeKey{I}) where {I}
    if FusionStyle(I) == UniqueFusion()
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

function artin_braid(fs_src::OuterTreeIterator{I, N, 0}, i; inv::Bool = false) where {I, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))

    uncoupled = fs_src.uncoupled[1]
    uncoupled′ = TupleTools.setindex(uncoupled, uncoupled[i + 1], i)
    uncoupled′ = TupleTools.setindex(uncoupled′, uncoupled[i], i + 1)

    isdual = fs_src.isdual[1]
    isdual′ = TupleTools.setindex(isdual, isdual[i], i + 1)
    isdual′ = TupleTools.setindex(isdual′, isdual[i + 1], i)

    fs_dst = OuterTreeIterator((uncoupled′, ()), (isdual′, ()))

    trees_src = fusiontrees(fs_src)
    trees_dst = fusiontrees(fs_dst)
    indexmap = Dict(f => ind for (ind, f) in enumerate(trees_dst))
    U = zeros(sectorscalartype(I), length(trees_dst), length(trees_src))

    for (col, (f₁, f₂)) in enumerate(trees_src)
        for (f₁′, c) in artin_braid(f₁, i; inv)
            row = indexmap[(f₁′, f₂)]
            U[row, col] = c
        end
    end

    return fs_dst, U
end

function braid(
        fs_src::OuterTreeIterator{I, N, 0}, levels::NTuple{N, Int}, p::NTuple{N, Int}
    ) where {I, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))

    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        uncoupled′ = TupleTools._permute(fs_src.uncoupled[1], p)
        isdual′ = TupleTools._permute(fs_src.isdual[1], p)
        fs_dst = OuterTreeIterator(uncoupled′, isdual′)

        trees_src = fusiontrees(fs_src)
        trees_dst = fusiontrees(fs_dst)
        indexmap = Dict(f => ind for (ind, f) in enumerate(trees_dst))
        U = zeros(sectorscalartype(I), length(trees_dst), length(trees_src))

        for (col, (f₁, f₂)) in enumerate(trees_src)
            for (f₁′, c) in braid(f₁, levels, p)
                row = indexmap[(f₁′, f₂)]
                U[row, col] = c
            end
        end

        return fs_dst, U
    end

    fs_dst, U = repartition(fs_src, N) # TODO: can we avoid this?
    for s in permutation2swaps(p)
        inv = levels[s] > levels[s + 1]
        fs_dst, U_tmp = artin_braid(fs_dst, s; inv)
        U = U_tmp * U
    end
    return fs_dst, U
end

function braid(
        fs_src::OuterTreeIterator{I}, levels::Index2Tuple, p::Index2Tuple{N₁, N₂}
    ) where {I, N₁, N₂}
    @assert numind(fs_src) == N₁ + N₂
    @assert numout(fs_src) == length(levels[1]) && numin(fs_src) == length(levels[2])
    @assert TupleTools.isperm((p[1]..., p[2]...))
    return _fsbraid((fs_src, levels, p))
end

const _FSBraidKey{I, N₁, N₂} = Tuple{<:OuterTreeIterator{I}, Index2Tuple, Index2Tuple{N₁, N₂}}

@cached function _fsbraid(
        key::_FSBraidKey{I, N₁, N₂}
    )::Tuple{OuterTreeIterator{I, N₁, N₂}, Matrix{sectorscalartype(I)}} where {I, N₁, N₂}
    fs_src, (l1, l2), (p1, p2) = key

    p = linearizepermutation(p1, p2, numout(fs_src), numin(fs_src))
    levels = (l1..., reverse(l2)...)

    fs_dst, U = repartition(fs_src, numind(fs_src))
    fs_dst, U_tmp = braid(fs_dst, levels, p)
    U = U_tmp * U
    fs_dst, U_tmp = repartition(fs_dst, N₁)
    U = U_tmp * U
    return fs_dst, U
end

function CacheStyle(::typeof(_fsbraid), k::_FSBraidKey{I}) where {I}
    if FusionStyle(I) isa UniqueFusion
        return NoCache()
    else
        return GlobalLRUCache()
    end
end

function permute(fs_src::OuterTreeIterator{I}, p::Index2Tuple) where {I}
    @assert BraidingStyle(I) isa SymmetricBraiding
    levels1 = ntuple(identity, numout(fs_src))
    levels2 = numout(fs_src) .+ ntuple(identity, numin(fs_src))
    return braid(fs_src, (levels1, levels2), p)
end
