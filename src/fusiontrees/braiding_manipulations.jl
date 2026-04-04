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
    1 <= i < N || throw(ArgumentError(lazy"Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    @assert FusionStyle(I) === UniqueFusion()

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
        return f′ => oneT
    end

    BraidingStyle(I) isa NoBraiding &&
        throw(SectorMismatch("Cannot braid sectors $(uncoupled[i]) and $(uncoupled[i + 1])"))

    if i == 1
        c = N > 2 ? inner[1] : coupled′
        R = oftype(oneT, (inv ? conj(Rsymbol(b, a, c)) : Rsymbol(a, b, c)))
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner, vertices)
        return f′ => R
    end

    # case i > 1: other naming convention
    b = uncoupled[i]
    d = uncoupled[i + 1]
    a = inner_extended[i - 1]
    c = inner_extended[i]
    e = inner_extended[i + 1]
    c′ = only(a ⊗ d)
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
    return f′ => coeff
end

function artin_braid(src::FusionTreeBlock{I, N, 0}, i; inv::Bool = false) where {I, N}
    1 <= i < N || throw(ArgumentError("Cannot swap outputs i=$i and i+1 out of only $N outputs"))
    isempty(src) && return src => zeros(sectorscalartype(I), 0, 0)

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

    if isunit(a) || isunit(b) # braiding with trivial sector: simple and always possible
        for (col, (f, f₂)) in enumerate(fusiontrees(src))
            inner = f.innerlines
            inner_extended = (uncoupled[1], inner..., coupled′)
            vertices = f.vertices
            inner′ = inner
            vertices′ = vertices
            if i > 1 # we also need to alter innerlines and vertices
                inner′ = TupleTools.setindex(
                    inner, inner_extended[isunit(a) ? (i + 1) : (i - 1)], i - 1
                )
                vertices′ = TupleTools.setindex(vertices′, vertices[i], i - 1)
                vertices′ = TupleTools.setindex(vertices′, vertices[i - 1], i)
            end
            f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′, vertices′)
            row = indexmap[treeindex_data((f′, f₂))]
            @inbounds U[row, col] = oneT
        end
        return dst => U
    end

    BraidingStyle(I) isa NoBraiding &&
        throw(SectorMismatch(lazy"Cannot braid sectors $a and $b"))

    T = typeof(oneT)
    localbraidcache = Dict{NTuple{6, I}, FusionStyle(I) isa MultiplicityFreeFusion ? T : Array{T, 4}}()
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
        if FusionStyle(I) isa MultiplicityFreeFusion
            for c′ in intersect(a ⊗ d, e ⊗ conj(b))
                coeff = let k = (a, b, c, d, e, c′)
                    get!(() -> _artin_braid_local(k, inv), localbraidcache, k)
                end
                iszero(coeff) && continue
                inner′ = TupleTools.setindex(inner, c′, i - 1)
                f′ = FusionTree{I}(uncoupled′, coupled′, isdual′, inner′)
                row = indexmap[treeindex_data((f′, f₂))]
                @inbounds U[row, col] = coeff
            end
        else # GenericFusion
            for c′ in intersect(a ⊗ d, e ⊗ conj(b))
                Rmat1 = inv ? Rsymbol(d, c, e)' : Rsymbol(c, d, e)
                Rmat2 = inv ? Rsymbol(d, a, c′)' : Rsymbol(a, d, c′)
                Fmat = Fsymbol(d, a, b, e, c′, c)
                coeff_tensor = let k = (a, b, c, d, e, c′)
                    get!(() -> _artin_braid_local(k, inv), localbraidcache, k)
                end
                μ = vertices[i - 1]
                ν = vertices[i]
                for λ in 1:size(coeff_tensor, 2)
                    for σ in 1:size(coeff_tensor, 1)
                        coeff = coeff_tensor[σ, λ, μ, ν]
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
    return dst => U
end

function _artin_braid_local((a, b, c, d, e, c′)::NTuple{6, I}, inv::Bool) where {I}
    if FusionStyle(I) isa MultiplicityFreeFusion
        coeff = if inv
            conj(Rsymbol(d, c, e) * Fsymbol(d, a, b, e, c′, c)) * Rsymbol(d, a, c′)
        else
            Rsymbol(c, d, e) * conj(Fsymbol(d, a, b, e, c′, c) * Rsymbol(a, d, c′))
        end
        return coeff
    else
        Rmat1 = inv ? Rsymbol(d, c, e)' : Rsymbol(c, d, e)
        Rmat2 = inv ? Rsymbol(d, a, c′)' : Rsymbol(a, d, c′)
        Fmat = Fsymbol(d, a, b, e, c′, c)
        @tensor coeff[σ, λ, μ, ν] := Rmat1[ν, ρ] * conj(Fmat[κ, λ, μ, ρ]) * conj(Rmat2[σ, κ])
        return coeff
    end
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
braid(f::FusionTree{I, N}, p::IndexTuple{N}, levels::IndexTuple{N}) where {I, N} =
    braid(f, (p, ()), (levels, ()))
function braid(f::FusionTree{I, N}, (p, _)::Index2Tuple{N, 0}, (levels, _)::Index2Tuple{N, 0}) where {I, N}
    TupleTools.isperm(p) || throw(ArgumentError("not a valid permutation: $p"))
    @assert FusionStyle(I) isa UniqueFusion
    if BraidingStyle(I) isa SymmetricBraiding # this assumes Fsymbols are 1!
        coeff = one(sectorscalartype(I))
        for i in 1:N
            for j in 1:(i - 1)
                if p[j] > p[i]
                    a, b = f.uncoupled[p[j]], f.uncoupled[p[i]]
                    coeff *= Rsymbol(a, b, only(a ⊗ b))
                end
            end
        end
        uncoupled′ = TupleTools._permute(f.uncoupled, p)
        coupled′ = f.coupled
        isdual′ = TupleTools._permute(f.isdual, p)
        f′ = FusionTree{I}(uncoupled′, coupled′, isdual′)
        return f′ => coeff
    else
        T = sectorscalartype(I)
        c = one(T)
        for s in permutation2swaps(p)
            inv = levels[s] > levels[s + 1]
            f, c′ = artin_braid(f, s; inv)
            c *= c′
            l = levels[s]
            levels = TupleTools.setindex(levels, levels[s + 1], s)
            levels = TupleTools.setindex(levels, l, s + 1)
        end

        return f => c
    end
end

# permute fusion tree
"""
    permute(f::FusionTree, p::NTuple{N, Int}) -> <:AbstractDict{typeof(t), <:Number}

Perform a permutation of the uncoupled indices of the fusion tree `f` and returns the result
as a `<:AbstractDict` of output trees and corresponding coefficients; this requires that
`BraidingStyle(sectortype(f)) isa SymmetricBraiding`.
"""
function permute(f::FusionTree{I, N}, p::IndexTuple{N}) where {I, N}
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

@cached function fsbraid(key::K)::_fsdicttype(K) where {I, N₁, N₂, K <: Union{FSPBraidKey{I, N₁, N₂}, FSBBraidKey{I, N₁, N₂}}}
    if K <: FSPBraidKey
        ((f₁, f₂), (p1, p2), (l1, l2)) = key
        p = linearizepermutation(p1, p2, length(f₁), length(f₂))
        levels = (l1..., reverse(l2)...)
        (f, f0), coeff1 = repartition((f₁, f₂), N₁ + N₂)
        f′, coeff2 = braid(f, p, levels)
        (f₁′, f₂′), coeff3 = repartition((f′, f0), N₁)
        return (f₁′, f₂′) => coeff1 * coeff2 * coeff3
    else
        src, (p1, p2), (l1, l2) = key

        p = linearizepermutation(p1, p2, numout(src), numin(src))
        levels = (l1..., reverse(l2)...)

        dst, U′ = repartition(src, numind(src))
        T = sectorscalartype(I)
        U = eltype(U′) == T ? U′ : T.(U′) # U′ has fusionscalartype(I) elements
        Uold = similar(U)

        for s in permutation2swaps(p)
            U, Uold = Uold, U
            inv = levels[s] > levels[s + 1]
            dst, U_tmp = artin_braid(dst, s; inv)
            U = mul!(U, U_tmp, Uold)
            l = levels[s]
            levels = TupleTools.setindex(levels, levels[s + 1], s)
            levels = TupleTools.setindex(levels, l, s + 1)
        end

        if N₂ == 0
            return dst => U
        else
            U, Uold = Uold, U
            dst, U_tmp = repartition(dst, N₁)
            U = mul!(U, U_tmp, Uold)
            return dst => U
        end
    end
end

CacheStyle(::typeof(fsbraid), k::Union{FSPBraidKey{I}, FSBBraidKey{I}}) where {I} =
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

"""
    flip((f₁, f₂)::FusionTreePair, i::Int; inv::Bool = false)
    -> SingletonDict{FusionTreePair, <:Number}

Flip the duality flag of the `i`-th uncoupled leg of the fusion-splitting tree pair
`(f₁, f₂)`, and return a `SingletonDict` containing the resulting tree pair together with
the scalar coefficient arising from the Z-isomorphism that relates the outgoing `a` line and
the incoming `dual(a)` line.

The coefficient for flipping leg `i` of `f₁` is determined by the twist `θₐ` and the
Frobenius-Schur phase `χₐ` of sector `a = f₁.uncoupled[i]`:
- If `isdual[i]` is currently `true` (the leg has an extra Z): coefficient is `χₐ * θₐ`.
- If `isdual[i]` is currently `false`: coefficient is `1`.

For legs of the fusion tree `f₂`, the conjugated phases appear instead.

The keyword `inv` inverts the operation, i.e. it exchanges the coefficients between the
dual and non-dual cases, so that applying `flip` with `inv = true` to the output of `flip`
with `inv = false` recovers the original tree with coefficient `1`.

!!! warning
    This operation is in general not an involution, and only `flip ∘ flip ∘ flip ∘ flip = identity`.

See also the multi-index method `flip((f₁, f₂)::FusionTreePair, ind; inv)`.
"""
function flip((f₁, f₂)::FusionTreePair, i::Int; inv::Bool = false)
    N₁ = numout((f₁, f₂))
    @assert 0 < i ≤ numind((f₁, f₂))
    if i ≤ N₁
        a = f₁.uncoupled[i]
        χₐ = frobenius_schur_phase(a)
        θₐ = twist(a)
        if !inv
            factor = f₁.isdual[i] ? χₐ * θₐ : one(θₐ)
        else
            factor = f₁.isdual[i] ? one(θₐ) : conj(χₐ * θₐ)
        end
        isdual′ = TupleTools.setindex(f₁.isdual, !f₁.isdual[i], i)
        f₁′ = typeof(f₁)(f₁.uncoupled, f₁.coupled, isdual′, f₁.innerlines, f₁.vertices)
        return SingletonDict((f₁′, f₂) => factor)
    else
        i -= N₁
        a = f₂.uncoupled[i]
        χₐ = frobenius_schur_phase(a)
        θₐ = twist(a)
        if !inv
            factor = f₂.isdual[i] ? conj(χₐ) * one(θₐ) : θₐ
        else
            factor = f₂.isdual[i] ? conj(θₐ) : χₐ * one(θₐ)
        end
        isdual′ = TupleTools.setindex(f₂.isdual, !f₂.isdual[i], i)
        f₂′ = typeof(f₂)(f₂.uncoupled, f₂.coupled, isdual′, f₂.innerlines, f₂.vertices)
        return SingletonDict((f₁, f₂′) => factor)
    end
end

"""
    flip((f₁, f₂)::FusionTreePair, ind; inv::Bool = false)
    -> SingletonDict{FusionTreePair, <:Number}

Flip the duality flags of all legs listed in `ind` sequentially, accumulating the scalar coefficients.
See the single-index method for the meaning of the coefficient per leg.
"""
function flip((f₁, f₂)::FusionTreePair, ind; inv::Bool = false)
    f₁′, f₂′ = f₁, f₂
    factor = one(sectorscalartype(sectortype(f₁)))
    for i in ind
        (f₁′, f₂′), s = only(flip((f₁′, f₂′), i; inv))
        factor *= s
    end
    return SingletonDict((f₁′, f₂′) => factor)
end
