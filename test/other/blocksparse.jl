using Test, TestExtras
using TensorOperations: TensorOperations as TO
using TensorKit
using TensorKit: blocksparsestructure,
    nmodes, nblocks, BlockSparseUnsupported, subblockstructure, fusiontrees,
    blocksparse_contract_signs, BlockSparseSigns, istrivial, subblock,
    _blocksparse_operand_signs, _blocksparse_output_signs, _scale_subblocks!,
    sectorscalartype

@isdefined(TestSetup) || include("../setup.jl")
using .TestSetup

# Host-side validation of the block-sparse description of a `HomSpace`: the sector-ordering,
# dual-space and stride reasoning the cuTENSOR backend relies on, pinned down without a GPU.

"per-mode cumulative section offsets, so section `k` of mode `i` spans `off[i][k]+1:off[i][k+1]`"
function section_offsets(s)
    N = nmodes(s)
    offsets = Vector{Vector{Int}}(undef, N)
    k = 0
    for i in 1:N
        cum = Int[0]
        for j in 1:s.numsections[i]
            push!(cum, cum[end] + s.extent[k + j])
        end
        k += s.numsections[i]
        offsets[i] = cum
    end
    return offsets
end

"read block `b` out of the flat data vector using only `(extent, strides, offset)`"
function read_block(t, s, b)
    N = nmodes(s)
    coords = view(s.coordinates, :, b)
    strides = view(s.strides, :, b)
    secoff = section_offsets(s)
    sz = ntuple(i -> secoff[i][coords[i] + 2] - secoff[i][coords[i] + 1], N)
    blk = Array{scalartype(t)}(undef, sz)
    for idx in CartesianIndices(sz)
        lin = s.offsets[b] + 1 + sum(i -> (idx[i] - 1) * strides[i], 1:N; init = 0)
        blk[idx] = t.data[lin]
    end
    return blk
end

"the dense index ranges block `b` occupies"
function block_ranges(s, b)
    N = nmodes(s)
    secoff = section_offsets(s)
    return ntuple(N) do i
        c = s.coordinates[i, b]
        return (secoff[i][c + 1] + 1):secoff[i][c + 2]
    end
end

const abelian_spaces = (
    "ℤ₂" => TestSetup.VRepℤ₂,
    "ℤ₃" => TestSetup.VRepℤ₃,
    "U₁" => TestSetup.VRepU₁,
)

# abelian fermionic spaces; `TestSetup.VfHubbard` has an `SU2Irrep` factor, so it cannot serve
const VfRepU₁ = (
    Vect[FermionNumber](0 => 2, 1 => 2, -1 => 1),
    Vect[FermionNumber](0 => 1, 1 => 1, -1 => 1)',
    Vect[FermionNumber](0 => 2, 1 => 1, -1 => 2),
    Vect[FermionNumber](0 => 1, 1 => 2, -1 => 1)',
    Vect[FermionNumber](0 => 1, 1 => 1, -1 => 2),
)
const fermionic_spaces = ("fℤ₂" => TestSetup.VfRepℤ₂, "fU₁" => VfRepU₁)

@testset "blocksparse_compatible trait" begin
    for I in (
            Z2Irrep, Z3Irrep, Z4Irrep, ZNIrrep{7}, U1Irrep, Z3Irrep ⊠ Z4Irrep,
            Z2Irrep ⊠ U1Irrep,
        )
        @test blocksparse_compatible(I)
    end
    # fermionic sectors qualify through the per-block sign correction
    for I in (FermionParity, FermionNumber, FermionParity ⊠ U1Irrep)
        @test blocksparse_compatible(I)
        @test BraidingStyle(I) isa Fermionic
    end
    # unique fusion is NOT sufficient: anyonic R-symbols are phases rather than signs
    for I in (Z3Element{1}, ZNElement{5, 2}, Z4Element{2})
        @test !blocksparse_compatible(I)
        @test FusionStyle(I) isa UniqueFusion  # ... and yet they do have unique fusion
    end
    for I in (
            Trivial, SU2Irrep, CU1Irrep, FibonacciAnyon, A4Irrep, IsingAnyon,
            FermionSpin, FermionParity ⊠ SU2Irrep ⊠ U1Irrep,
        )
        @test !blocksparse_compatible(I)
    end
end

@testset "layout for $name" for (name, V) in (abelian_spaces..., fermionic_spaces...)
    V1, V2, V3, V4, V5 = V
    homspaces = (
        V1 ← V2,
        V1 ⊗ V2 ← V3,
        V1 ⊗ V2 ← V3 ⊗ V4,
        V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5,
        V1 ⊗ V2 ← one(V1),          # N₂ == 0
        one(V1) ← V4 ⊗ V5,          # N₁ == 0
    )
    for W in homspaces
        s = blocksparsestructure(W)

        @testset "$W" begin
            # the constraint block-sparse libraries impose: one permutation -- here the
            # identity -- sorts every block's strides ascending
            for (_, st, _) in values(subblockstructure(W))
                @test issorted(st)
            end

            # one block per fusion tree pair, in that order
            @test nblocks(s) == length(fusiontrees(W))
            @test nmodes(s) == numind(W)
            @test length(s.numsections) == numind(W)
            @test length(s.extent) == sum(s.numsections)
            @test size(s.coordinates) == (nmodes(s), nblocks(s))
            @test size(s.strides) == (nmodes(s), nblocks(s))

            # sections of a mode tile that mode's dimension, even though `W[i]` duals the
            # domain factors while the section labels do not
            secoff = section_offsets(s)
            for i in 1:numind(W)
                @test last(secoff[i]) == dim(W[i])
            end

            # coordinates are zero-based and in range
            for b in 1:nblocks(s), i in 1:nmodes(s)
                c = s.coordinates[i, b]
                @test 0 <= c < s.numsections[i]
            end

            # strides and offsets are `subblockstructure`'s, verbatim
            substr = collect(values(subblockstructure(W)))
            for b in 1:nblocks(s)
                _, st, offset = substr[b]
                @test s.offsets[b] == offset
                @test Tuple(view(s.strides, :, b)) == st
            end

            # blocks exactly tile the flat data vector
            @test sum(1:nblocks(s); init = 0) do b
                prod(length, block_ranges(s, b))
            end == dim(W)
        end
    end
end

@testset "blocks equal dense slices for $name" for (name, V) in abelian_spaces
    V1, V2, V3, V4, V5 = V
    hasfusiontensor(sectortype(V1)) || continue
    for W in (V1 ⊗ V2 ← V3, V1 ⊗ V2 ← V3 ⊗ V4, V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5)
        for T in (Float64, ComplexF64)
            t = randn(T, W)
            s = blocksparsestructure(W)
            A = convert(Array, t)
            covered = falses(size(A))
            @testset "$T $W" begin
                for b in 1:nblocks(s)
                    r = block_ranges(s, b)
                    # the load-bearing claim: a subblock addressed only via
                    # (extent, strides, offset) is the corresponding dense slice
                    @test read_block(t, s, b) ≈ A[r...]
                    @test !any(view(covered, r...))   # blocks do not overlap
                    covered[r...] .= true
                end
                # everything the blocks do not cover is zero by charge conservation
                @test all(iszero, A[.!covered])
            end
        end
    end
end

@testset "fused scratch write for $name, $T" for (name, V) in (
            abelian_spaces..., fermionic_spaces...,
        ), T in (Float64, ComplexF64)

    # drive the extension's fused scratch write on the host, where a bug shows up without a GPU
    V1, V2, V3, V4, V5 = V
    for W in (V1 ⊗ V2 ← V3, V1 ⊗ V2 ← V3 ⊗ V4, V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5)
        t = randn(T, W)
        n = length(fusiontrees(W))
        # unit scalars, as the real thing gets: ±1 for fermions, phases in the complex case
        signs = T <: Complex ? cis.(2π .* rand(n)) : rand((-1.0, 1.0), n)
        for doconj in (false, true)
            # NaN-filled, so a subblock the fused write skips cannot go unnoticed
            dst = similar(t)
            fill!(dst.data, T(NaN))
            _scale_subblocks!(dst, t, signs, doconj)

            # the two-pass reference the extension used to run
            ref = copy(t)
            doconj && (ref.data .= conj.(t.data))
            _scale_subblocks!(ref, signs)

            @test !any(isnan, dst.data)
            @test dst.data ≈ ref.data
        end
    end
end

# Per-block sign correction
# -------------------------
# Reimplement on the host what the extension does -- scale the operand blocks, contract
# block-wise, scale the result blocks -- and compare against the default path.

"total extent of each mode, i.e. the shape of the dense array the blocks tile"
function dense_size(W)
    s = blocksparsestructure(W)
    secoff = section_offsets(s)
    return ntuple(i -> last(secoff[i]), nmodes(s))
end

"embed the stored blocks of `t` into a dense array; whatever no block covers stays zero"
function dense_embedding(t)
    s = blocksparsestructure(space(t))
    d = zeros(scalartype(t), dense_size(space(t)))
    for b in 1:nblocks(s)
        d[block_ranges(s, b)...] = read_block(t, s, b)
    end
    return d
end

"the inverse: read the stored blocks of a `WC`-shaped tensor back out of a dense array"
function from_dense_embedding(d, WC, ::Type{T}) where {T}
    C = zeros(T, WC)
    s = blocksparsestructure(WC)
    trees = collect(fusiontrees(WC))
    for b in 1:nblocks(s)
        copyto!(subblock(C, trees[b]), d[block_ranges(s, b)...])
    end
    return C
end

"""
    naive_blockwise(WC, A, pA, conjA, B, pB, conjB, pAB, T)

What a block-sparse kernel computes: the raw block data contracted with plain
tensor-contraction semantics, with no fusion tree transformation.

Realized by embedding the stored blocks densely and contracting *once*, rather than by matching
blocks pairwise: that avoids reimplementing cuTENSOR's block matching, and it only gives the
right answer if the dense axes of the contracted modes line up, so a violation of the
section-index invariant surfaces as a `DimensionMismatch` rather than as a wrong number.
"""
function naive_blockwise(WC, A, pA, conjA, B, pB, conjB, pAB, ::Type{T}) where {T}
    d = zeros(T, dense_size(WC))
    TO.tensorcontract!(
        d, dense_embedding(A), pA, conjA, dense_embedding(B), pB, conjB, pAB, true, false
    )
    return from_dense_embedding(d, WC, T)
end

# drive the shipping helper, so that a bug in it is caught here and not only on a GPU
scale_blocks!(t, ::Nothing) = t
scale_blocks!(t, signs) = _scale_subblocks!(t, signs)

"""
`tensorcontract!` via the block-sparse route, with the sign correction applied. `α`/`β` are
applied by a plain `add!` rather than by the extension's pre/post-scaling trick, which is
covered on the GPU side instead.
"""
function signed_blockwise(C0, A, pA, conjA, B, pB, conjB, pAB, α, β, ::Type{T}, signs) where {T}
    A′ = scale_blocks!(copy(A), signs.A)
    B′ = scale_blocks!(copy(B), signs.B)
    D = naive_blockwise(space(C0), A′, pA, conjA, B′, pB, conjB, pAB, T)
    scale_blocks!(D, signs.C)
    return add!(scale!(copy(C0), β), D, α)
end

"""
The contraction geometries the sign derivation has to cover. Conjugating *exactly one* operand
dualizes the composability requirement on the contracted legs, so both variants are listed and
the sweep skips whichever combinations do not typecheck.
"""
function sign_cases(V)
    V1, V2, V3, V4, V5 = V
    return (
        # contracted leg non-dual, so `blas_contract!` inserts no twist
        ("non-dual contracted leg", V1 ⊗ V2 ← V3, V3 ← V4 ⊗ V5, ((1, 2), (3,)), ((1,), (2, 3))),
        ("non-dual contracted leg, dualized", V1 ⊗ V2 ← V3, dual(V3) ← V4 ⊗ V5, ((1, 2), (3,)), ((1,), (2, 3))),
        # ... and dual (`V2` and `V4` are dual in every test space), so it does
        ("dual contracted leg", V1 ⊗ V2 ← V4, V4 ← V3 ⊗ V5, ((1, 2), (3,)), ((1,), (2, 3))),
        ("dual contracted leg, dualized", V1 ⊗ V2 ← V4, dual(V4) ← V3 ⊗ V5, ((1, 2), (3,)), ((1,), (2, 3))),
        # contracted leg in `A`'s codomain and in `B`'s domain
        ("contracted legs off-diagonal", V1 ⊗ V2 ← V3, V4 ← V1 ⊗ V5, ((2, 3), (1,)), ((2,), (1, 3))),
        ("contracted legs off-diagonal, dualized", V1 ⊗ V2 ← V3, V4 ← dual(V1) ⊗ V5, ((2, 3), (1,)), ((2,), (1, 3))),
        ("two contracted legs", V1 ← V2 ⊗ V3, V2 ⊗ V3 ← V4, ((1,), (2, 3)), ((1, 2), (3,))),
        ("two contracted legs, dualized", V1 ← V2 ⊗ V3, dual(V2) ⊗ dual(V3) ← V4, ((1,), (2, 3)), ((1, 2), (3,))),
        ("two contracted legs, swapped", V1 ← V2 ⊗ V3, V2 ⊗ V3 ← V4, ((1,), (3, 2)), ((2, 1), (3,))),
    )
end

"output partitions to sweep for an `n`-index result; the sign only shows up under these"
function sign_partitions(n)
    return n == 4 ?
        (
            ((1, 2), (3, 4)), ((1, 2, 3, 4), ()), ((), (1, 2, 3, 4)), ((2, 1), (4, 3)),
            ((3, 1), (2, 4)), ((4, 3, 2, 1), ()),
        ) :
        (((1,), (2,)), ((1, 2), ()), ((2, 1), ()), ((), (2, 1)))
end

"the output space of a contraction, or `nothing` if this combination is not well-formed"
function contract_space(WA, pA, conjA, WB, pB, conjB, pAB)
    return try
        TO.tensorcontract(WA, pA, conjA, WB, pB, conjB, pAB)
    catch e
        e isa Union{SpaceMismatch, ArgumentError} || rethrow()
        nothing
    end
end

const conjugations = ((false, false), (true, false), (false, true), (true, true))

@testset "signs are trivial for $name" for (name, V) in abelian_spaces
    for (_, WA, WB, pA, pB) in sign_cases(V)
        for pAB in sign_partitions(length(pA[1]) + length(pB[2])),
                (conjA, conjB) in conjugations

            WC = contract_space(WA, pA, conjA, WB, pB, conjB, pAB)
            isnothing(WC) && continue
            # bosonic sectors need no correction at all, whatever the permutations
            @test istrivial(
                blocksparse_contract_signs(WC, WA, pA, conjA, WB, pB, conjB, pAB)
            )
        end
    end
end

@testset "sign correction for $name, $T" for (name, V) in fermionic_spaces,
        T in (Float64, ComplexF64)

    covered = Dict(c => 0 for c in conjugations)
    for (case, WA, WB, pA, pB) in sign_cases(V)
        @testset "$case" begin
            for pAB in sign_partitions(length(pA[1]) + length(pB[2])),
                    (conjA, conjB) in conjugations

                WC = contract_space(WA, pA, conjA, WB, pB, conjB, pAB)
                isnothing(WC) && continue
                covered[(conjA, conjB)] += 1
                signs = blocksparse_contract_signs(WC, WA, pA, conjA, WB, pB, conjB, pAB)
                A, B, C0 = randn(T, WA), randn(T, WB), randn(T, WC)
                for (α, β) in ((one(T), zero(T)), (T(-1.5), T(0.5)))
                    ref = TO.tensorcontract!(
                        copy(C0), A, pA, conjA, B, pB, conjB, pAB, α, β,
                        TO.DefaultBackend(), TO.DefaultAllocator()
                    )
                    got = signed_blockwise(
                        C0, A, pA, conjA, B, pB, conjB, pAB, α, β, T, signs
                    )
                    @test got.data ≈ ref.data
                end
            end
        end
    end
    # not all `(conjA, conjB)` combinations compose, so guard against a silently skipped one
    @testset "coverage" begin
        for c in conjugations
            @test covered[c] > 0
        end
    end
end

@testset "every sign factor is load-bearing" begin
    # guards against a vacuous test above: with any factor dropped the results must disagree
    V = TestSetup.VfRepℤ₂
    T = ComplexF64
    Tc = sectorscalartype(sectortype(V[1]))
    drop_none(WC, WA, pA, cA, WB, pB, cB, pAB) =
        blocksparse_contract_signs(WC, WA, pA, cA, WB, pB, cB, pAB)
    drop_all(WC, WA, pA, cA, WB, pB, cB, pAB) =
        BlockSparseSigns{Tc}(nothing, nothing, nothing)
    drop_twist(WC, WA, pA, cA, WB, pB, cB, pAB) = BlockSparseSigns{Tc}(
        _blocksparse_operand_signs(Tc, WA, pA, cA, false),
        _blocksparse_operand_signs(Tc, WB, pB, cB, false),
        _blocksparse_output_signs(Tc, WC, length(pA[1]), pAB)
    )
    drop_output(WC, WA, pA, cA, WB, pB, cB, pAB) = BlockSparseSigns{Tc}(
        _blocksparse_operand_signs(Tc, WA, pA, cA, true),
        _blocksparse_operand_signs(Tc, WB, pB, cB, false),
        nothing
    )

    for variant in (drop_all, drop_twist, drop_output)
        agree = total = 0
        for (_, WA, WB, pA, pB) in sign_cases(V)
            for pAB in sign_partitions(length(pA[1]) + length(pB[2])),
                    (conjA, conjB) in ((false, false), (true, false))

                WC = contract_space(WA, pA, conjA, WB, pB, conjB, pAB)
                isnothing(WC) && continue
                A, B, C0 = randn(T, WA), randn(T, WB), randn(T, WC)
                ref = TO.tensorcontract!(
                    copy(C0), A, pA, conjA, B, pB, conjB, pAB, one(T), zero(T),
                    TO.DefaultBackend(), TO.DefaultAllocator()
                )
                got = signed_blockwise(
                    C0, A, pA, conjA, B, pB, conjB, pAB, one(T), zero(T), T,
                    variant(WC, WA, pA, conjA, WB, pB, conjB, pAB)
                )
                total += 1
                agree += got.data ≈ ref.data
            end
        end
        @test total > 0
        @test agree < total
    end
end

@testset "repartition alone needs a correction" begin
    # a natural index *order* is not enough: bending a leg between codomain and domain is not
    # free for fermions, so `pAB = ((1,2,3,4),())` carries a correction despite permuting nothing
    V1, V2, V3, V4, V5 = TestSetup.VfRepℤ₂
    WA, WB, pA, pB = V1 ⊗ V2 ← V3, V3 ← V4 ⊗ V5, ((1, 2), (3,)), ((1,), (2, 3))
    identity_order = ((1, 2, 3, 4), ())
    @test TO.linearize(identity_order) == (1, 2, 3, 4)   # ... it really is the identity
    WC = TO.tensorcontract(WA, pA, false, WB, pB, false, identity_order)
    signs = blocksparse_contract_signs(WC, WA, pA, false, WB, pB, false, identity_order)
    @test !isnothing(signs.C)
    # ... whereas the partition the contraction produces naturally does not
    natural = ((1, 2), (3, 4))
    WC′ = TO.tensorcontract(WA, pA, false, WB, pB, false, natural)
    @test isnothing(blocksparse_contract_signs(WC′, WA, pA, false, WB, pB, false, natural).C)
end

@testset "sign derivation is a pure function of the geometry" begin
    # derived once per plan rather than per call, so it must be deterministic and must depend
    # on the conjugation flags -- which is why a plan carries all four combinations
    V1, V2, V3, V4, V5 = TestSetup.VfRepℤ₂
    WA, WB = V1 ⊗ V2 ← V3, V3 ← V4 ⊗ V5
    pA, pB, pAB = ((1, 2), (3,)), ((1,), (2, 3)), ((2, 1), (4, 3))
    WC = TO.tensorcontract(WA, pA, false, WB, pB, false, pAB)
    s1 = blocksparse_contract_signs(WC, WA, pA, false, WB, pB, false, pAB)
    @test !istrivial(s1)                  # this geometry does need a correction
    s1′ = blocksparse_contract_signs(WC, WA, pA, false, WB, pB, false, pAB)
    @test (s1′.A, s1′.B, s1′.C) == (s1.A, s1.B, s1.C)
    s2 = blocksparse_contract_signs(WC, WA, pA, true, WB, pB, false, pAB)
    @test s2.A != s1.A
end

@testset "unsupported spaces" begin
    V = Vect[Z2Irrep](0 => 2, 1 => 2)
    # no non-zero blocks: incompatible charge sectors
    W = Vect[Z2Irrep](0 => 1) ← Vect[Z2Irrep](1 => 1)
    @test_throws BlockSparseUnsupported blocksparsestructure(W)
    # a scalar has no modes
    @test_throws BlockSparseUnsupported blocksparsestructure(one(V) ← one(V))
end

@testset "caching" begin
    V = Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1)
    W = V ⊗ V ← V
    s1 = blocksparsestructure(W)
    s2 = blocksparsestructure(W)
    @test s1 === s2                       # cached per HomSpace
    empty_globalcaches!()
    s3 = blocksparsestructure(W)
    @test s3 !== s1
    @test s3.offsets == s1.offsets        # ... but identical content
    @test s3.strides == s1.strides
end
