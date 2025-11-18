# BRAIDING MANIPULATIONS:
#-----------------------------------------------
# -> manipulations that depend on a braiding
# -> requires both Fsymbol and Rsymbol
"""
    artin_braid(f::FusionTree, i; inv::Bool = false) -> <:AbstractDict{typeof(f), <:Number}

Perform an elementary braid (Artin generator) of neighbouring uncoupled indices `i` and
`i+1` on a fusion tree `f`, and returns the result as a dictionary of output trees and
corresponding coefficients.

The keyword `inv` determines whether index `i` will braid above or below index `i+1`, i.e.
applying `artin_braid(f′, i; inv = true)` to all the outputs `f′` of
`artin_braid(f, i; inv = false)` and collecting the results should yield a single fusion
tree with non-zero coefficient, namely `f` with coefficient `1`. This keyword has no effect
if `BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function artin_braid(f::FusionTree{I, N}, i; inv::Bool = false) where {I, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    uncoupled = f.uncoupled
    a, b = uncoupled[i], uncoupled[i + 1]
    uncoupled′ = TupleTools.setindex(uncoupled, b, i)
    uncoupled′ = TupleTools.setindex(uncoupled′, a, i + 1)
    coupled′ = f.coupled
    isdual′ = TupleTools.setindex(f.isdual, f.isdual[i], i + 1)
    isdual′ = TupleTools.setindex(isdual′, f.isdual[i + 1], i)
    inner = f.innerlines
    inner_extended = (uncoupled[1], inner..., coupled′)
    vertices = f.vertices
    oneT = one(sectorscalartype(I))

    if isunit(a) || isunit(b)
        # braiding with trivial sector: simple and always possible
        inner′ = inner
        vertices′ = vertices
        if i > 1 # we also need to alter innerlines and vertices
            inner′ = TupleTools.setindex(
                inner,
                inner_extended[isunit(a) ? (i + 1) : (i - 1)],
                i - 1
            )
            vertices′ = TupleTools.setindex(vertices′, vertices[i], i - 1)
            vertices′ = TupleTools.setindex(vertices′, vertices[i - 1], i)
        end
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
        return fusiontreedict(I)(f′ => oneT)
    end

    BraidingStyle(I) isa NoBraiding &&
        throw(SectorMismatch("Cannot braid sectors $(uncoupled[i]) and $(uncoupled[i + 1])"))

    if i == 1
        c = N > 2 ? inner[1] : coupled′
        if FusionStyle(I) isa MultiplicityFreeFusion
            R = oftype(oneT, (inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)))
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices)
            return fusiontreedict(I)(f′ => R)
        else # GenericFusion
            μ = vertices[1]
            Rmat = inv ? Rsymbol(b, a, c)' : Rsymbol(a, b, c)
            local newtrees
            for ν in axes(Rmat, 2)
                R = oftype(oneT, Rmat[μ, ν])
                iszero(R) && continue
                vertices′ = TupleTools.setindex(vertices, ν, 1)
                f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices′)
                if (@isdefined newtrees)
                    push!(newtrees, f′ => R)
                else
                    newtrees = fusiontreedict(I)(f′ => R)
                end
            end
            return newtrees
        end
    end
    # case i > 1: other naming convention
    b = uncoupled[i]
    d = uncoupled[i + 1]
    a = inner_extended[i - 1]
    c = inner_extended[i]
    e = inner_extended[i + 1]
    if FusionStyle(I) isa UniqueFusion
        c′ = first(a ⊗ d)
        coeff = oftype(
            oneT,
            if inv
                conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
            else
                Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
            end
        )
        inner′ = TupleTools.setindex(inner, c′, i - 1)
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
        return fusiontreedict(I)(f′ => coeff)
    elseif FusionStyle(I) isa SimpleFusion
        local newtrees
        cs = collect(I, intersect(a ⊗ d, e ⊗ conj(b)))
        for c′ in cs
            coeff = oftype(
                oneT,
                if inv
                    conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
                else
                    Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
                end
            )
            iszero(coeff) && continue
            inner′ = TupleTools.setindex(inner, c′, i - 1)
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
            if (@isdefined newtrees)
                push!(newtrees, f′ => coeff)
            else
                newtrees = fusiontreedict(I)(f′ => coeff)
            end
        end
        return newtrees
    else # GenericFusion
        local newtrees
        cs = collect(I, intersect(a ⊗ d, e ⊗ conj(b)))
        for c′ in cs
            Rmat1 = inv ? Rsymbol(d, c, e)' : Rsymbol(c, d, e)
            Rmat2 = inv ? Rsymbol(d, a, c′)' : Rsymbol(a, d, c′)
            Fmat = Fsymbol(d, a, b, e, c′, c)
            μ = vertices[i - 1]
            ν = vertices[i]
            for σ in 1:Nsymbol(a, d, c′)
                for λ in 1:Nsymbol(c′, b, e)
                    coeff = zero(oneT)
                    for ρ in 1:Nsymbol(d, c, e), κ in 1:Nsymbol(d, a, c′)
                        coeff += Rmat1[ν, ρ] * conj(Fmat[κ, λ, μ, ρ]) * conj(Rmat2[σ, κ])
                    end
                    iszero(coeff) && continue
                    vertices′ = TupleTools.setindex(vertices, σ, i - 1)
                    vertices′ = TupleTools.setindex(vertices′, λ, i)
                    inner′ = TupleTools.setindex(inner, c′, i - 1)
                    f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
                    if (@isdefined newtrees)
                        push!(newtrees, f′ => coeff)
                    else
                        newtrees = fusiontreedict(I)(f′ => coeff)
                    end
                end
            end
        end
        return newtrees
    end
end

function artin_braid(src::FusionTreeBlock{I, N, 0}, i; inv::Bool = false) where {I, N}
    1 <= i < N ||
        throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))

    uncoupled = src.uncoupled[1]
    a, b = uncoupled[i], uncoupled[i + 1]
    uncoupled′ = TupleTools.setindex(uncoupled, b, i)
    uncoupled′ = TupleTools.setindex(uncoupled′, a, i + 1)
    coupled′ = rightunit(src.uncoupled[1][N])

    isdual = src.isdual[1]
    isdual′ = TupleTools.setindex(isdual, isdual[i], i + 1)
    isdual′ = TupleTools.setindex(isdual′, isdual[i + 1], i)
    dst = FusionTreeBlock{I}((uncoupled′, ()), (isdual′, ()); sizehint = length(src))

    oneT = one(sectorscalartype(I))

    indexmap = treeindex_map(dst)
    U = zeros(sectorscalartype(I), length(dst), length(src))

    if isone(a) || isone(b) # braiding with trivial sector: simple and always possible
        for (col, (f, f₂)) in enumerate(fusiontrees(src))
            inner = f.innerlines
            inner_extended = (uncoupled[1], inner..., coupled′)
            vertices = f.vertices
            inner′ = inner
            vertices′ = vertices
            if i > 1 # we also need to alter innerlines and vertices
                inner′ = TupleTools.setindex(
                    inner, inner_extended[isone(a) ? (i + 1) : (i - 1)], i - 1
                )
                vertices′ = TupleTools.setindex(vertices′, vertices[i], i - 1)
                vertices′ = TupleTools.setindex(vertices′, vertices[i - 1], i)
            end
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
            row = indexmap[treeindex_data((f′, f₂))]
            @inbounds U[row, col] = oneT
        end
        return dst, U
    end

    BraidingStyle(I) isa NoBraiding &&
        throw(SectorMismatch(lazy"Cannot braid sectors $a and $b"))

    for (col, (f, f₂)) in enumerate(fusiontrees(src))
        inner = f.innerlines
        inner_extended = (uncoupled[1], inner..., coupled′)
        vertices = f.vertices

        if i == 1
            c = N > 2 ? inner[1] : coupled′
            if FusionStyle(I) isa MultiplicityFreeFusion
                R = oftype(oneT, (inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)))
                f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices)
                row = indexmap[treeindex_data((f′, f₂))]
                @inbounds U[row, col] = R
            else # GenericFusion
                μ = vertices[1]
                Rmat = inv ? Rsymbol(b, a, c)' : Rsymbol(a, b, c)
                for ν in axes(Rmat, 2)
                    R = oftype(oneT, Rmat[μ, ν])
                    iszero(R) && continue
                    vertices′ = TupleTools.setindex(vertices, ν, 1)
                    f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices′)
                    row = indexmap[treeindex_data((f′, f₂))]
                    @inbounds U[row, col] = R
                end
            end
            continue
        end
        # case i > 1: other naming convention
        b = uncoupled[i]
        d = uncoupled[i + 1]
        a = inner_extended[i - 1]
        c = inner_extended[i]
        e = inner_extended[i + 1]
        if FusionStyle(I) isa UniqueFusion
            c′ = first(a ⊗ d)
            coeff = oftype(
                oneT,
                if inv
                    conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
                else
                    Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
                end
            )
            inner′ = TupleTools.setindex(inner, c′, i - 1)
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
            row = indexmap[treeindex_data((f′, f₂))]
            @inbounds U[row, col] = coeff
        elseif FusionStyle(I) isa SimpleFusion
            cs = collect(I, intersect(a ⊗ d, e ⊗ conj(b)))
            for c′ in cs
                coeff = oftype(
                    oneT,
                    if inv
                        conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
                    else
                        Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
                    end
                )
                iszero(coeff) && continue
                inner′ = TupleTools.setindex(inner, c′, i - 1)
                f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
                row = indexmap[treeindex_data((f′, f₂))]
                @inbounds U[row, col] = coeff
            end
        else # GenericFusion
            cs = collect(I, intersect(a ⊗ d, e ⊗ conj(b)))
            for c′ in cs
                Rmat1 = inv ? Rsymbol(d, c, e)' : Rsymbol(c, d, e)
                Rmat2 = inv ? Rsymbol(d, a, c′)' : Rsymbol(a, d, c′)
                Fmat = Fsymbol(d, a, b, e, c′, c)
                μ = vertices[i - 1]
                ν = vertices[i]
                for σ in 1:Nsymbol(a, d, c′)
                    for λ in 1:Nsymbol(c′, b, e)
                        coeff = zero(oneT)
                        for ρ in 1:Nsymbol(d, c, e), κ in 1:Nsymbol(d, a, c′)
                            coeff += Rmat1[ν, ρ] * conj(Fmat[κ, λ, μ, ρ]) *
                                conj(Rmat2[σ, κ])
                        end
                        iszero(coeff) && continue
                        vertices′ = TupleTools.setindex(vertices, σ, i - 1)
                        vertices′ = TupleTools.setindex(vertices′, λ, i)
                        inner′ = TupleTools.setindex(inner, c′, i - 1)
                        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
                        row = indexmap[treeindex_data((f′, f₂))]
                        @inbounds U[row, col] = coeff
                    end
                end
            end
        end
    end

    return dst, U
end

# braid fusion tree
"""
    braid(f::FusionTree{<:Sector, N}, p::NTuple{N, Int}, levels::NTuple{N, Int})
    -> <:AbstractDict{typeof(t), <:Number}

Perform a braiding of the uncoupled indices of the fusion tree `f` and return the result as
a `<:AbstractDict` of output trees and corresponding coefficients. The braiding is
determined by specifying that the new sector at position `k` corresponds to the sector that
was originally at the position `i = p[k]`, and assigning to every index `i` of the original
fusion tree a distinct level or depth `levels[i]`. This permutation is then decomposed into
elementary swaps between neighbouring indices, where the swaps are applied as braids such
that if `i` and `j` cross, ``τ_{i,j}`` is applied if `levels[i] < levels[j]` and
``τ_{j,i}^{-1}`` if `levels[i] > levels[j]`. This does not allow to encode the most general
braid, but a general braid can be obtained by combining such operations.
"""
function braid(f::FusionTree{I, N}, p::NTuple{N, Int}, levels::NTuple{N, Int}) where {I, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))
    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        coeff = one(sectorscalartype(I))
        for i in 1:N
            for j in 1:(i - 1)
                if p[j] > p[i]
                    a, b = f.uncoupled[p[j]], f.uncoupled[p[i]]
                    coeff *= Rsymbol(a, b, first(a ⊗ b))
                end
            end
        end
        uncoupled′ = TupleTools._permute(f.uncoupled, p)
        coupled′ = f.coupled
        isdual′ = TupleTools._permute(f.isdual, p)
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′)
        return fusiontreedict(I)(f′ => coeff)
    else
        T = sectorscalartype(I)
        coeff = one(T)
        trees = FusionTreeDict(f => coeff)
        newtrees = empty(trees)
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            for (f, c) in trees
                for (f′, c′) in artin_braid(f, s; inv)
                    newtrees[f′] = get(newtrees, f′, zero(coeff)) + c * c′
                end
            end
            l = levels[s]
            levels = TupleTools.setindex(levels, levels[s + 1], s)
            levels = TupleTools.setindex(levels, l, s + 1)
            trees, newtrees = newtrees, trees
            empty!(newtrees)
        end
        return trees
    end
end

# permute fusion tree
"""
    permute(f::FusionTree, p::NTuple{N, Int}) -> <:AbstractDict{typeof(t), <:Number}

Perform a permutation of the uncoupled indices of the fusion tree `f` and returns the result
as a `<:AbstractDict` of output trees and corresponding coefficients; this requires that
`BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function permute(f::FusionTree{I, N}, p::NTuple{N, Int}) where {I, N}
    @assert BraidingStyle(I) isa SymmetricBraiding
    return braid(f, p, ntuple(identity, Val(N)))
end

# braid double fusion tree
"""
    braid((f₁, f₂)::FusionTreePair, (p1, p2)::Index2Tuple, (levels1, levels2)::Index2Tuple)
        -> <:AbstractDict{<:FusionTreePair{I, N₁, N₂}}, <:Number}

Input is a fusion-splitting tree pair that describes the fusion of a set of incoming
uncoupled sectors to a set of outgoing uncoupled sectors, represented using the splitting
tree `f₁` and fusion tree `f₂`, such that the incoming sectors `f₂.uncoupled` are fused to
`f₁.coupled == f₂.coupled` and then to the outgoing sectors `f₁.uncoupled`. Compute new
trees and corresponding coefficients obtained from repartitioning and braiding the tree such
that sectors `p1` become outgoing and sectors `p2` become incoming. The uncoupled indices in
splitting tree `f₁` and fusion tree `f₂` have levels (or depths) `levels1` and `levels2`
respectively, which determines how indices braid. In particular, if `i` and `j` cross,
``τ_{i,j}`` is applied if `levels[i] < levels[j]` and ``τ_{j,i}^{-1}`` if `levels[i] >
levels[j]`. This does not allow to encode the most general braid, but a general braid can
be obtained by combining such operations.
"""
function braid(src::Union{FusionTreePair, FusionTreeBlock}, p::Index2Tuple, levels::Index2Tuple)
    @assert numind(src) == length(p[1]) + length(p[2])
    @assert numout(src) == length(levels[1]) && numin(src) == length(levels[2])
    @assert TupleTools.isperm((p[1]..., p[2]...))
    return fsbraid((src, p, levels))
end

const FSPBraidKey{I, N₁, N₂} = Tuple{FusionTreePair{I}, Index2Tuple{N₁, N₂}, Index2Tuple}
const FSBBraidKey{I, N₁, N₂} = Tuple{FusionTreeBlock{I}, Index2Tuple{N₁, N₂}, Index2Tuple}

Base.@assume_effects :foldable function _fsdicttype(::Type{T}) where {I, N₁, N₂, T <: FSPBraidKey{I, N₁, N₂}}
    E = sectorscalartype(I)
    return Pair{fusiontreetype(I, N₁, N₂), E}
end
Base.@assume_effects :foldable function _fsdicttype(::Type{T}) where {I, N₁, N₂, T <: FSBBraidKey{I, N₁, N₂}}
    F₁ = fusiontreetype(I, N₁)
    F₂ = fusiontreetype(I, N₂)
    E = sectorscalartype(I)
    return Pair{FusionTreeBlock{I, N₁, N₂, Tuple{F₁, F₂}}, Matrix{E}}
end

@cached function fsbraid(key::K)::_fsdicttype(K) where {I, N₁, N₂, K <: FSPBraidKey{I, N₁, N₂}}
    ((f₁, f₂), (p1, p2), (l1, l2)) = key
    p = linearizepermutation(p1, p2, length(f₁), length(f₂))
    levels = (l1..., reverse(l2)...)
    local newtrees
    for ((f, f0), coeff1) in repartition((f₁, f₂), N₁ + N₂)
        for (f′, coeff2) in braid(f, p, levels)
            for ((f₁′, f₂′), coeff3) in repartition((f′, f0), N₁)
                if @isdefined newtrees
                    newtrees[(f₁′, f₂′)] = get(newtrees, (f₁′, f₂′), zero(coeff3)) +
                        coeff1 * coeff2 * coeff3
                else
                    newtrees = fusiontreedict(I)((f₁′, f₂′) => coeff1 * coeff2 * coeff3)
                end
            end
        end
    end
    return only(newtrees)
end

function transformation_matrix(transform, dst::FusionTreeBlock{I}, src::FusionTreeBlock{I}) where {I}
    U = zeros(sectorscalartype(I), length(dst), length(src))
    indexmap = treeindex_map(dst)
    for (col, f) in enumerate(fusiontrees(src))
        for (f′, c) in transform(f)
            row = indexmap[f′]
            U[row, col] = c
        end
    end
    return U
end
@cached function fsbraid(key::K)::_fsdicttype(K) where {I, N₁, N₂, K <: FSBBraidKey{I, N₁, N₂}}
    src, (p1, p2), (l1, l2) = key

    p = linearizepermutation(p1, p2, numout(src), numin(src))
    levels = (l1..., reverse(l2)...)

    dst, U = repartition(src, numind(src))

    if FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding
        uncoupled′ = TupleTools._permute(dst.uncoupled[1], p)
        isdual′ = TupleTools._permute(dst.isdual[1], p)

        dst′ = FusionTreeBlock{I}(uncoupled′, isdual′)
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
        return dst => U
    else
        dst, U_tmp = repartition(dst, N₁)
        U = U_tmp * U
        return dst => U
    end
end

CacheStyle(::typeof(fsbraid), k::FSPBraidKey{I}) where {I} =
    FusionStyle(I) isa UniqueFusion ? NoCache() : GlobalLRUCache()
CacheStyle(::typeof(fsbraid), k::FSBBraidKey{I}) where {I} =
    FusionStyle(I) isa UniqueFusion ? NoCache() : GlobalLRUCache()

"""
    permute((f₁, f₂)::FusionTreePair, (p1, p2)::Index2Tuple)
        -> <:AbstractDict{<:FusionTreePair{I, N₁, N₂}}, <:Number}

Input is a double fusion tree that describes the fusion of a set of incoming uncoupled
sectors to a set of outgoing uncoupled sectors, represented using the individual trees of
outgoing (`t1`) and incoming sectors (`t2`) respectively (with identical coupled sector
`t1.coupled == t2.coupled`). Computes new trees and corresponding coefficients obtained from
repartitioning and permuting the tree such that sectors `p1` become outgoing and sectors
`p2` become incoming.
"""
function permute(src::Union{FusionTreePair, FusionTreeBlock}, p::Index2Tuple)
    @assert BraidingStyle(src) isa SymmetricBraiding
    levels1 = ntuple(identity, numout(src))
    levels2 = numout(src) .+ ntuple(identity, numin(src))
    return braid(src, p, (levels1, levels2))
end
