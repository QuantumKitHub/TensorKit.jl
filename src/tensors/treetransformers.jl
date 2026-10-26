"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.

The transformers only store how subblocks map onto each other in terms of their positions in
[`StridedSubblocks`](@ref) (the canonical order of [`fusiontrees`](@ref)), together with the recoupling
coefficients, and are therefore independent of the sectortype once constructed. The
transformation is that of `permutedims(conjsrc ? conj(tsrc) : tsrc, p)` where `p` indexes the legs
of `tsrc` itself; when `conjsrc` is `true` the fusion trees that are transformed are those of
`adjoint(space(tsrc))`, which read the subblocks of `tsrc` with the tree pair swapped.
"""
abstract type TreeTransformer end

# `StridedSubblocks` report their storage as the `StridedView` parent type, which is `Memory` there
@static if isdefined(Core, :Memory)
    const CPUStorage{T} = Union{Array{T}, Memory{T}}
else
    const CPUStorage{T} = Array{T}
end

"""
    recoupling_scalartype(A::Type{<:AbstractVector}, Tₛ::Type{<:Number}) -> Type{<:Number}

Scalar type used to store the recoupling coefficients with sector scalar type `Tₛ` in the
transformers for destination tensors with storagetype `A`. For storage with BLAS scalars, this is
the precision of the storage, where real coefficients are kept real also for complex storage, such
that they can be applied to the real and imaginary parts at once.
"""
function recoupling_scalartype(::Type{A}, ::Type{Tₛ}) where {A, Tₛ <: Number}
    T = scalartype(A)
    T <: BlasFloat || return Tₛ
    return Tₛ <: Real ? real(T) : complex(T)
end

"""
    SubblockGroup{T}

Small subblocks of equal size and strides, which are transformed by a single loop over all of them
rather than by one `tensoradd!` call per subblock. Entry `k` of a subblock lies at `rel_dst[k]`
and `rel_src[k]` relative to the offsets of the destination and source subblock, where the source
strides are permuted such that both refer to the same entry. Term `n` then computes
`dst[offset_dst[n] .+ rel_dst] = α * coeff[n] * src[offset_src[n] .+ rel_src] + b * dst[offset_dst[n] .+ rel_dst]`,
with `b = β` for the first term into every destination subblock and `b = 1` for the groups of
further recoupling terms, which have `accumulate` set and are applied after all others.
The terms are sorted by `offset_dst`.
"""
struct SubblockGroup{T}
    rel_dst::Vector{Int}
    rel_src::Vector{Int}
    offset_dst::Vector{Int}
    offset_src::Vector{Int}
    coeff::Vector{T}
    accumulate::Bool
end

# subblocks up to this length are transformed in `SubblockGroup`s
const SUBBLOCKGROUP_MAXLENGTH = 1024

# (coefficient, destination position, source position)
const UniqueTransformerData{T} = Tuple{T, Int, Int}

"""
    UniqueTreeTransformer{T, N} <: TreeTransformer

Tree transformation for `UniqueFusion` sectors, where every source subblock maps onto a single
destination subblock with a scalar coefficient, stored as `(coeff, idst, isrc)`. The subblock
structures of the destination and source spaces are kept alongside, such that the
[`StridedSubblocks`](@ref) of both tensors can be created without further lookups. The small
subblocks are also collected in [`SubblockGroup`](@ref)s, and `large` holds the positions in
`data` of the others.
"""
struct UniqueTreeTransformer{T, N} <: TreeTransformer
    data::Vector{UniqueTransformerData{T}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
    groups::Vector{SubblockGroup{T}}
    large::Vector{Int}
end

# (recoupling matrix, destination positions, source positions)
const GenericTransformerData{T} = Tuple{Matrix{T}, Vector{Int}, Vector{Int}}

"""
    GenericTreeTransformer{T, N} <: TreeTransformer

Tree transformation for sectors with multiple fusion channels, where the subblocks of a
[`FusionTreeBlock`](@ref) map onto the subblocks of the transformed block through a recoupling
matrix, stored as `(U, inds_dst, inds_src)`, where `U[j, i]` maps source `i` onto destination `j`.
The subblock structures of the destination and source spaces are kept alongside, such that the
[`StridedSubblocks`](@ref) of both tensors can be created without further lookups. The blocks of
small subblocks are also collected in [`SubblockGroup`](@ref)s, and `large` holds the positions in
`data` of the others.
"""
struct GenericTreeTransformer{T, N} <: TreeTransformer
    data::Vector{GenericTransformerData{T}}
    structure_dst::Vector{StridedStructure{N}}
    structure_src::Vector{StridedStructure{N}}
    groups::Vector{SubblockGroup{T}}
    large::Vector{Int}
end

function UniqueTreeTransformer(::Type{A}, transform, p, Vdst::HomSpace, Vsrc::HomSpace, conjsrc::Bool) where {A}
    t₀ = Base.time()

    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)

    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vdst)))
    data = Vector{UniqueTransformerData{T}}(undef, length(src_trees))

    @timeit_debug GLOBAL_TIMER "symmetry: tree transform" for (isrc, (f₁, f₂)) in enumerate(src_trees)
        f_dst, coeff = transform(conjsrc ? (f₂, f₁) : (f₁, f₂))
        _, (_, idst) = gettoken(dst_trees, f_dst)
        data[isrc] = (convert(T, coeff), idst, isrc)
    end

    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    groups, large = subblockgroups(T, data, structure_dst, structure_src, linearize(p))
    transformer = UniqueTreeTransformer(data, structure_dst, structure_src, groups, large)

    Δt = Base.time() - t₀
    @debug(lazy"Treetransformer for $Vsrc to $Vdst via $p", conjsrc, nblocks = length(data), Δt)

    return transformer
end

function GenericTreeTransformer(::Type{A}, transform, p, Vdst::HomSpace, Vsrc::HomSpace, conjsrc::Bool) where {A}
    t₀ = Base.time()
    spacecheck_transform(permute, Vdst, Vsrc, p, conjsrc)
    # the fusion blocks that are transformed are those of the adjoint space for a conjugated source
    Vsrc′ = conjsrc ? Vsrc' : Vsrc
    src_trees, dst_trees = fusiontrees(Vsrc), fusiontrees(Vdst)
    structure_dst = degeneracystructure(Vdst).subblockstructure
    structure_src = degeneracystructure(Vsrc).subblockstructure
    T = recoupling_scalartype(A, sectorscalartype(sectortype(Vsrc)))

    fblocks = @timeit_debug GLOBAL_TIMER "bookkeeping: fusionblocks" fusionblocks(Vsrc′)
    nblocks = length(fblocks)
    data = Vector{GenericTransformerData{T}}(undef, nblocks)
    weights = Vector{Int}(undef, nblocks)

    nthreads = get_num_manipulation_threads()
    @timeit_debug GLOBAL_TIMER "symmetry: recoupling matrices" begin
        taskforeach(1:nblocks, nthreads) do i
            fs_src = fblocks[i]
            fs_dst, U = transform(fs_src)
            @timeit_debug GLOBAL_TIMER "bookkeeping: subblock positions" begin
                # the token into the fusion tree `Indices` is the subblock position
                inds_src = map(fusiontrees(fs_src)) do (f₁, f₂)
                    _, (_, isrc) = gettoken(src_trees, conjsrc ? (f₂, f₁) : (f₁, f₂))
                    return isrc
                end
                inds_dst = map(fusiontrees(fs_dst)) do f
                    _, (_, idst) = gettoken(dst_trees, f)
                    return idst
                end
            end
            data[i] = (convert(Matrix{T}, U), inds_dst, inds_src)
            # cost model: L input blocks each going to L output blocks of a given length
            weights[i] = length(U) * prod(structure_dst[first(inds_dst)][1])

            @debug(
                lazy"Created recoupling block for uncoupled: $(fs_src.uncoupled)",
                sz = size(U), sparsity = count(!iszero, U) / length(U)
            )
        end
    end

    # sort by (approximate) weight to facilitate multi-threading strategies
    @timeit_debug GLOBAL_TIMER "bookkeeping: sort" Base.permute!(data, sortperm(weights; rev = true))
    groups, large = subblockgroups(T, data, structure_dst, structure_src, linearize(p))
    transformer = GenericTreeTransformer{T, numind(Vdst)}(data, structure_dst, structure_src, groups, large)

    Δt = Base.time() - t₀
    @debug(
        lazy"TreeTransformer for $Vsrc to $Vdst via $p", conjsrc,
        nblocks = nblocks,
        sz_median = nblocks > 0 ? size(data[cld(end, 2)][1], 1) : 0,
        sz_max = nblocks > 0 ? size(data[1][1], 1) : 0,
        Δt
    )

    return transformer
end

"""
    subblockgroups(T, data, structure_dst, structure_src, p) -> (groups, large)

Collect the terms of the transformer `data` that act on subblocks with at most
`SUBBLOCKGROUP_MAXLENGTH` entries into [`SubblockGroup`](@ref)s by size and strides, with the
accumulating groups last, and return them together with the positions in `data` of the other entries.
"""
function subblockgroups(::Type{T}, data, structure_dst::Vector{StridedStructure{N}}, structure_src, p) where {T, N}
    terms = Tuple{T, Int, Int, Bool}[]
    large = Int[]
    for (n, (U, inds_dst, inds_src)) in enumerate(data)
        if prod(first(structure_dst[first(inds_dst)])) > SUBBLOCKGROUP_MAXLENGTH
            push!(large, n)
        else
            _push_terms!(terms, U, inds_dst, inds_src)
        end
    end
    sort!(terms; by = term -> structure_dst[term[2]][3])

    # strides of dimensions of size 1 are irrelevant, and set to zero to obtain fewer groups
    groups = Dict{Tuple{Bool, NTuple{3, NTuple{N, Int}}}, SubblockGroup{T}}()
    for (coeff, idst, isrc, accumulate) in terms
        sz, strides_dst, offset_dst = structure_dst[idst]
        _, strides_src, offset_src = structure_src[isrc]
        strides_dst = map((n, s) -> n == 1 ? 0 : s, sz, strides_dst)
        strides_src′ = map((n, s) -> n == 1 ? 0 : s, sz, TupleTools.getindices(strides_src, p))
        group = get!(groups, (accumulate, (sz, strides_dst, strides_src′))) do
            rel_dst, rel_src = Int[], Int[]
            for I in CartesianIndices(sz)
                push!(rel_dst, 1 + sum((Tuple(I) .- 1) .* strides_dst; init = 0))
                push!(rel_src, 1 + sum((Tuple(I) .- 1) .* strides_src′; init = 0))
            end
            return SubblockGroup(rel_dst, rel_src, Int[], Int[], T[], accumulate)
        end
        push!(group.offset_dst, offset_dst)
        push!(group.offset_src, offset_src)
        push!(group.coeff, coeff)
    end
    return sort!(collect(values(groups)); by = group -> group.accumulate), large
end

_push_terms!(terms, coeff::Number, idst::Int, isrc::Int) = push!(terms, (coeff, idst, isrc, false))
function _push_terms!(terms, U::AbstractMatrix, inds_dst, inds_src)
    for (i, idst) in enumerate(inds_dst)
        js = findall(!iszero, view(U, i, :))
        isempty(js) && push!(js, 1) # such that β is still applied
        for j in js
            push!(terms, (U[i, j], idst, inds_src[j], j != first(js)))
        end
    end
    return terms
end

"""
    buffersize(transformer::TreeTransformer) -> Int

Compute the workspace size required to pack, recouple and unpack the largest multi-tree
block, i.e. `prod(sz_src) * (rows + cols)` where `(rows, cols) = size(U)` is the size of
the recoupling matrix.
"""
buffersize(::UniqueTreeTransformer) = 0
function buffersize(transformer::GenericTreeTransformer)
    structure_src = transformer.structure_src
    return maximum(transformer.data; init = 0) do (U, _, inds_src)
        return length(U) == 1 ? 0 : prod(structure_src[first(inds_src)][1]) * sum(size(U))
    end
end

function treetransformertype(::Type{A}, Vdst, Vsrc) where {A}
    I = sectortype(Vdst)
    T = recoupling_scalartype(A, sectorscalartype(I))
    N = numind(Vdst)
    return FusionStyle(I) == UniqueFusion() ? UniqueTreeTransformer{T, N} : GenericTreeTransformer{T, N}
end

function TreeTransformer(
        ::Type{A}, transform::Function, p, Vdst::HomSpace{S}, Vsrc::HomSpace{S}, conjsrc::Bool
    ) where {A, S}
    I = sectortype(Vdst)
    return FusionStyle(I) == UniqueFusion() ?
        UniqueTreeTransformer(A, transform, p, Vdst, Vsrc, conjsrc) :
        GenericTreeTransformer(A, transform, p, Vdst, Vsrc, conjsrc)
end

# braid is special because it has levels
@cached function treebraider(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool, levels::IndexTuple
    )::treetransformertype(A, Vdst, Vsrc) where {TA}
    Vsrc′, p′ = conjsrc ? (Vsrc', adjointtensorindices(Vsrc, p)) : (Vsrc, p)
    # levels are attached to the legs, so they follow the same relabeling as the permutation
    levels′ = conjsrc ? TupleTools.getindices(levels, adjointtensorindices(Vsrc′, allind(Vsrc′))) : levels
    levels″ = (TupleTools.getindices(levels′, codomainind(Vsrc′)), TupleTools.getindices(levels′, domainind(Vsrc′)))
    fusiontreebraider(f) = braid(f, p′, levels″)
    return TreeTransformer(A, fusiontreebraider, p, Vdst, Vsrc, conjsrc)
end

@cached function treetransposer(
        A::Type{TA}, Vdst::TensorMapSpace, Vsrc::TensorMapSpace, p::Index2Tuple, conjsrc::Bool
    )::treetransformertype(A, Vdst, Vsrc) where {TA}
    p′ = conjsrc ? adjointtensorindices(Vsrc, p) : p
    fusiontreetransform(f) = transpose(f, p′)
    return TreeTransformer(A, fusiontreetransform, p, Vdst, Vsrc, conjsrc)
end

# default cachestyle is GlobalLRUCache
