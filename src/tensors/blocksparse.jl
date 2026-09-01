# Block-sparse description of a `HomSpace`
# ----------------------------------------
# Layout bookkeeping for backends handing raw block data to a block-sparse library.

"""
    BlockSparseStructure

Description of a `HomSpace` as a block-sparse tensor of `N = nmodes(s)` modes: which sections
each mode is divided into, and where each non-zero block lives in the flat data vector.

Blocks are enumerated in the order of [`fusiontrees`](@ref), which is the canonical order
that must be used consistently for every subsequent operation on the same space.

## Fields
- `numsections`: number of sections of each mode, length `N`.
- `extent`: extents of the sections of each mode, length `sum(numsections)`, ordered
  mode-major (all sections of mode 1, then all sections of mode 2, ...).
- `coordinates`: **zero-based** section coordinate of each non-zero block, as an
  `N × nblocks` matrix, one column per block.
- `strides`: strides of each non-zero block, same shape and order as `coordinates`. These
  come straight from [`subblockstructure`](@ref), and are therefore normalized and monotone.
- `offsets`: **zero-based** offset of each non-zero block into the flat data vector, length
  `nblocks`.

See also [`blocksparsestructure`](@ref), [`blocksparse_compatible`](@ref).
"""
struct BlockSparseStructure
    numsections::Vector{Int32}
    extent::Vector{Int64}
    coordinates::Matrix{Int32}
    strides::Matrix{Int64}
    offsets::Vector{Int64}
end

nmodes(s::BlockSparseStructure) = length(s.numsections)
nblocks(s::BlockSparseStructure) = length(s.offsets)

"""
    blocksparsestructure(W::HomSpace) -> BlockSparseStructure

Compute the [`BlockSparseStructure`](@ref) of `W`. The result is cached per `HomSpace`,
since it depends on the degeneracy dimensions through the block strides and offsets.

Throws [`BlockSparseUnsupported`](@ref) if `W` has no modes or no non-zero blocks.

See also [`blocksparse_compatible`](@ref).
""" blocksparsestructure(::HomSpace)

@cached function blocksparsestructure(W::HomSpace)::BlockSparseStructure
    N₁, N₂ = numout(W), numin(W)
    N = N₁ + N₂
    I = sectortype(W)
    codom, dom = codomain(W), domain(W)

    N > 0 || throw(BlockSparseUnsupported("a scalar has no block-sparse description"))

    # labels come from `codomain(W)[i]`/`domain(W)[j]`, not `space(W, i)`, so that
    # two contractible modes get matching section indices.
    modespace(i) = i <= N₁ ? codom[i] : dom[i - N₁]
    seclists = ntuple(i -> collect(sectors(modespace(i))), N)

    numsections = Int32[length(l) for l in seclists]
    any(iszero, numsections) && throw(BlockSparseUnsupported("`$W` has an empty mode"))

    extent = Vector{Int64}(undef, sum(numsections))
    k = 0
    for i in 1:N
        V = modespace(i)
        for c in seclists[i]
            extent[k += 1] = dim(V, c)
        end
    end
    secindex = ntuple(N) do i
        d = Dict{I, Int32}()
        for (j, c) in enumerate(seclists[i])
            d[c] = Int32(j - 1)
        end
        return d
    end

    trees = fusiontrees(W)
    substructure = degeneracystructure(W).subblockstructure
    L = length(trees)
    L > 0 || throw(BlockSparseUnsupported("`$W` has no non-zero blocks"))

    coordinates = Matrix{Int32}(undef, N, L)
    strides = Matrix{Int64}(undef, N, L)
    offsets = Vector{Int64}(undef, L)

    for b in 1:L
        f₁, f₂ = gettokenvalue(trees, b)
        _, st, offset = substructure[b]
        @inbounds for i in 1:N₁
            coordinates[i, b] = secindex[i][f₁.uncoupled[i]]
        end
        @inbounds for j in 1:N₂
            coordinates[N₁ + j, b] = secindex[N₁ + j][f₂.uncoupled[j]]
        end
        @inbounds for i in 1:N
            strides[i, b] = st[i]
        end
        offsets[b] = offset
    end

    return BlockSparseStructure(numsections, extent, coordinates, strides, offsets)
end

CacheStyle(::typeof(blocksparsestructure), ::HomSpace) = GlobalLRUCache()

# Per-block sign corrections
# --------------------------
# For a sector whose fusion tree transformations are all scalar, a raw block-wise contraction
# differs from the categorical one by a per-block scalar, factorizing as `sA(a)·sB(b)·sC(c)`.

"""
    BlockSparseSigns{T}

The per-block scalars that turn a raw block-wise contraction into TensorKit's categorical
one, one entry per non-zero block of each of the three tensors, in the block order of
[`BlockSparseStructure`](@ref).

A field is `nothing` when every scalar of that tensor is one, so that the correction can be
skipped entirely. All three are `nothing` for bosonic sectors.

See also [`TensorKit.blocksparse_contract_signs`](@ref).
"""
struct BlockSparseSigns{T}
    A::Union{Nothing, Vector{T}}
    B::Union{Nothing, Vector{T}}
    C::Union{Nothing, Vector{T}}
end

istrivial(s::BlockSparseSigns) = isnothing(s.A) && isnothing(s.B) && isnothing(s.C)

# the uncoupled sector carried by leg `j` of a tree pair, with `N₁` codomain legs
function _legsector((f₁, f₂)::FusionTreePair, N₁::Int, j::Int)
    return j <= N₁ ? f₁.uncoupled[j] : f₂.uncoupled[j - N₁]
end

# a tensor whose scalars are all one needs no correction: the common case even for fermions
_nothing_if_trivial(signs) = all(isone, signs) ? nothing : signs

"""
    blocksparse_contract_signs(WC, WA, pA, conjA, WB, pB, conjB, pAB) -> BlockSparseSigns

Derive the per-block scalars that a block-sparse contraction of the raw block data must be
corrected by to agree with `tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β)`.

A pure function of the spaces and index tuples, and *not* cached: the backend derives it once
per contraction plan rather than once per call, so a cache here would only ever be hit on a
plan-cache miss. Callers repeating a contraction should hold on to the result the same way.

Assumes that every fusion tree transformation of `sectortype(WA)` is scalar, i.e. that
[`blocksparse_compatible`](@ref) holds; the scalars are derived from `permute` of a fusion
tree pair, which for unique fusion returns a single `(tree, coeff)` pair.

The correction to apply is: scale the blocks of `A` by `signs.A`, those of `B` by `signs.B`,
contract, and scale the blocks of the result by `signs.C` before accumulating into `C`.

See also [`TensorKit.BlockSparseSigns`](@ref).
""" blocksparse_contract_signs

function blocksparse_contract_signs(
        WC::HomSpace, WA::HomSpace, pA::Index2Tuple, conjA::Bool,
        WB::HomSpace, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple
    )
    I = sectortype(WA)
    T = sectorscalartype(I)
    # bosonic sectors have unit R-symbols and twists, so nothing to correct
    BraidingStyle(I) isa Bosonic && return BlockSparseSigns{T}(nothing, nothing, nothing)

    # the twist for a dual contracted leg is always attributable to `A`, since
    # `domain(Anew) == codomain(Bnew)`.
    sA = _blocksparse_operand_signs(T, WA, pA, conjA, true)
    sB = _blocksparse_operand_signs(T, WB, pB, conjB, false)
    sC = _blocksparse_output_signs(T, WC, length(pA[1]), pAB)
    return BlockSparseSigns{T}(sA, sB, sC)
end

function _blocksparse_operand_signs(
        ::Type{T}, W::HomSpace, p::Index2Tuple, isconj::Bool, twistcontracted::Bool
    ) where {T}
    # fold `conj` by moving to the adjoint space, exactly as `_generic_tensorcontract!` does
    W★ = isconj ? adjoint(W) : W
    p★ = isconj ? adjointtensorindices(W, p) : p
    N₁ = numout(W★)
    twistlegs = twistcontracted ? filter(j -> !isdual(W★[j]), p★[2]) : ()

    trees = fusiontrees(W)
    signs = Vector{T}(undef, length(trees))
    for (i, f) in enumerate(trees)
        f★ = isconj ? (f[2], f[1]) : f
        coeff = last(permute(f★, p★))
        for j in twistlegs
            coeff *= twist(_legsector(f★, N₁, j))
        end
        signs[i] = coeff
    end
    return _nothing_if_trivial(signs)
end

"scale each subblock of `t` by its entry of `signs`, or by its conjugate if `inverse`"
function _scale_subblocks!(t::AbstractTensorMap, signs, inverse::Bool = false)
    # `StridedView(t.data, sz, str, offset)` is what `subblock` builds, minus the lookup
    for (s, (sz, str, offset)) in zip(signs, values(subblockstructure(space(t))))
        isone(s) && continue
        sv = StridedView(t.data, sz, str, offset)
        sv .*= inverse ? conj(s) : s
    end
    return t
end

"write `signs ⊙ src` into `dst`, conjugating first if `doconj`; every subblock is written"
function _scale_subblocks!(
        dst::AbstractTensorMap, src::AbstractTensorMap, signs, doconj::Bool
    )
    for (s, (sz, str, offset)) in zip(signs, values(subblockstructure(space(src))))
        srcv = StridedView(src.data, sz, str, offset)
        dstv = StridedView(dst.data, sz, str, offset)
        dstv .= s .* (doconj ? conj(srcv) : srcv)
    end
    return dst
end

function _blocksparse_output_signs(
        ::Type{T}, WC::HomSpace, nopenA::Int, pAB::Index2Tuple
    ) where {T}
    # the coefficient of the final `Cnew -> C` permutation, which the kernel does not apply
    ipAB = TO.repartition(invperm(linearize(pAB)), nopenA)

    trees = fusiontrees(WC)
    signs = Vector{T}(undef, length(trees))
    for (i, f) in enumerate(trees)
        # permuting the other way gives the inverse, i.e. the conjugate for a unitary scalar
        signs[i] = conj(last(permute(f, ipAB)))
    end
    return _nothing_if_trivial(signs)
end

# convenience accessors on tensors
blocksparsestructure(t::AbstractTensorMap) = blocksparsestructure(space(t))
