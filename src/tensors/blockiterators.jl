"""
    struct BlockIterator{T<:AbstractTensorMap,S}

Iterator over the blocks of type `T`, possibly holding some pre-computed data of type `S`
"""
struct BlockIterator{T <: AbstractTensorMap, S}
    t::T
    structure::S
end

Base.IteratorSize(::BlockIterator) = Base.HasLength()
Base.IteratorEltype(::BlockIterator) = Base.HasEltype()
Base.eltype(::Type{<:BlockIterator{T}}) where {T} = Pair{sectortype(T), blocktype(T)}
Base.length(iter::BlockIterator) = length(iter.structure)
Base.isdone(iter::BlockIterator, state...) = Base.isdone(iter.structure, state...)

# TODO: fast-path when structures are the same?
# TODO: implement scheduler
"""
    foreachblock(f, ts::AbstractTensorMap...; [scheduler])

Apply `f` to each block of `t` and the corresponding blocks of `ts`.
Optionally, `scheduler` can be used to parallelize the computation.
This function is equivalent to the following loop:

```julia
for c in union(blocksectors.(ts)...)
    bs = map(t -> block(t, c), ts)
    f(c, bs)
end
```
"""
function foreachblock(f, t, ts...; scheduler = nothing)
    tensors = (t, ts...)
    allsectors = union(blocksectors.(tensors)...)
    foreach(allsectors) do c
        return f(c, block.(tensors, Ref(c)))
    end
    return nothing
end
function foreachblock(f, t; scheduler = nothing)
    foreach(blocks(t)) do (c, b)
        return f(c, (b,))
    end
    return nothing
end

function show_blocks(io, mime::MIME"text/plain", iter; maytruncate::Bool = true)
    if maytruncate && get(io, :limit, false)
        numlinesleft, numcols = get(io, :displaysize, displaysize(io))::Tuple{Int, Int}
        numlinesleft -= 2 # lines of headers have already been subtracted, but not the 2 spare lines for old and new prompts
        minlinesperblock = 7 # aim to have at least this many lines per printed block (= 5 lines for the actual matrix)
        minnumberofblocks = clamp(length(iter), 1, 3) # aim to show at least this many blocks
        truncateblocks = sum(cb -> min(size(cb[2], 1) + 2, minlinesperblock), iter; init = 0) > numlinesleft
        maxnumlinesperblock = max(div(numlinesleft - 2 * truncateblocks, minnumberofblocks), minlinesperblock)
        # aim to show at least minnumberofblocks, but not if this means that there would be less than minlinesperblock
        # deduct two lines for a truncation message (and newline) if needed
        for (n, (c, b)) in enumerate(iter)
            n == 1 || print(io, "\n\n")
            numlinesneeded = min(size(b, 1) + 2, maxnumlinesperblock)
            if numlinesleft >= numlinesneeded + 2 * truncateblocks
                # we can still print at least this block, and have two lines for
                # the truncation message (and its newline) if it is required
                print(io, " * ", c, " => ")
                newio = IOContext(io, :displaysize => (maxnumlinesperblock - 1 + 3, numcols))
                # subtract 1 line for the newline, but add 3 because of how matrices are printed
                show(newio, mime, b)
                numlinesleft -= numlinesneeded
            else
                print(io, " * ", "  \u2026   [output of ", length(iter) - n + 1, " more block(s) truncated]")
                break
            end
        end
    else
        first = true
        for (c, b) in iter
            first || print(io, "\n\n")
            print(io, " * ", c, " => ")
            show(io, mime, b)
            first = false
        end
    end
    return nothing
end

function show_blocks(io, iter)
    print(io, "(")
    Base.join(io, iter, ", ")
    print(io, ")")
    return nothing
end

function Base.summary(io::IO, b::BlockIterator)
    print(io, "blocks(")
    Base.showarg(io, b.t, false)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", b::BlockIterator)
    summary(io, b)
    println(io, ":")
    (numlines, numcols) = get(io, :displaysize, displaysize(io))::Tuple{Int, Int}
    newio = IOContext(io, :displaysize => (numlines - 1, numcols))
    show_blocks(newio, mime, b; maytruncate = false)
    return nothing
end

# Positional subblock collections
# -------------------------------
# These address subblocks by their position in the canonical order of `fusiontrees(space(t))`,
# hoisting the space-level lookups out of the individual accesses.

"""
    struct SubblockIterator{T <: AbstractTensorMap, S}
    SubblockIterator(t::AbstractTensorMap)

Collection of the subblocks of a tensor of type `T`, indexable both by fusion tree pair and by
token, i.e. by the position in the canonical order of [`fusiontrees`](@ref), and iterating over
`(f₁, f₂) => subblock` pairs. This is what [`subblocks`](@ref) returns.

This object additionally has a `structure` field which can be used to precompute data that facilitates
fast indexing/iteration. By default this only holds the fusion tree pairs, but it can be any dictionary
mapping those onto the data needed to address the relevant subblocks, as `TensorMap` does.
"""
struct SubblockIterator{T <: AbstractTensorMap, S}
    t::T
    structure::S
end

# default just holds the set of fusiontrees for fast iteration and mapping index to fusiontree
SubblockIterator(t::AbstractTensorMap) = SubblockIterator(t, fusiontrees(t))

storagetype(::Type{<:SubblockIterator{T}}) where {T} = storagetype(T)

Base.IteratorSize(::SubblockIterator) = Base.HasLength()
Base.IteratorEltype(::SubblockIterator) = Base.HasEltype()
Base.eltype(::Type{<:SubblockIterator{T}}) where {T} = Pair{fusiontreetype(T), subblocktype(T)}
Base.length(iter::SubblockIterator) = length(iter.structure)
Base.firstindex(::SubblockIterator) = 1
Base.lastindex(iter::SubblockIterator) = length(iter)
Base.isdone(iter::SubblockIterator, i::Int = 1) = i > length(iter)

@propagate_inbounds Base.getindex(iter::SubblockIterator, i::Int) =
    subblock(iter.t, gettokenvalue(keys(iter.structure), i))
@propagate_inbounds Base.getindex(iter::SubblockIterator, f::FusionTreePair) = subblock(iter.t, f)

function Base.iterate(iter::SubblockIterator, i::Int = 1)
    i > length(iter) && return nothing
    @inbounds return gettokenvalue(keys(iter.structure), i) => iter[i], i + 1
end

function Base.showarg(io::IO, iter::SubblockIterator, toplevel::Bool)
    print(io, "subblocks(")
    Base.showarg(io, iter.t, false)
    print(io, ")")
    return nothing
end
function Base.summary(io::IO, iter::SubblockIterator)
    Base.showarg(io, iter, true)
    return nothing
end

function show_subblocks(io::IO, mime::MIME"text/plain", iter::SubblockIterator)
    if FusionStyle(sectortype(iter.t)) isa UniqueFusion
        first = true
        for ((f₁, f₂), b) in iter
            first || print(io, "\n\n")
            print(io, " * ", f₁.uncoupled, " ← ", f₂.uncoupled, " => ")
            show(io, mime, b)
            first = false
        end
    else
        first = true
        for ((f₁, f₂), b) in iter
            first || print(io, "\n\n")
            print(io, " * ", (f₁, f₂), " => ")
            show(io, mime, b)
            first = false
        end
    end
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", iter::SubblockIterator)
    summary(io, iter)
    println(io, ":")
    show_subblocks(io, mime, iter)
    return nothing
end

"""
    struct StridedSubblocks{A <: DenseVector, N}
    StridedSubblocks(t::TensorMap)

Sector-independent, integer-indexable collection of the subblocks of a `TensorMap`, as `StridedView`s into its flat data vector.
Subblock `i` corresponds to the `i`th fusion tree pair in the canonical order of [`fusiontrees`](@ref).

This is the data structure consumed by the index manipulation kernels, whose type does not depend on the sectortype of `t`:
it only carries the storage type `A` of the flat data vector and the number of indices `N` of the subblocks.
As a result, the kernels do not have to be recompiled for each new symmetry type.
"""
struct StridedSubblocks{A <: DenseVector, N}
    data::A
    structure::Vector{StridedStructure{N}}
    # store the data as `StridedView` parents it, so that `A` is also the parent type of the views
    function StridedSubblocks(data::DenseVector, structure::Vector{StridedStructure{N}}) where {N}
        data′ = parent(StridedView(data))
        return new{typeof(data′), N}(data′, structure)
    end
end

storagetype(::Type{StridedSubblocks{A, N}}) where {A, N} = A

Base.length(s::StridedSubblocks) = length(s.structure)
Base.firstindex(s::StridedSubblocks) = 1
Base.lastindex(s::StridedSubblocks) = length(s)
Base.eltype(::Type{StridedSubblocks{A, N}}) where {A, N} = StridedView{eltype(A), N, A, typeof(identity)}

Base.@propagate_inbounds function Base.getindex(s::StridedSubblocks, i::Int)
    sz, str, offset = s.structure[i]
    return StridedView(s.data, sz, str, offset)
end

function Base.iterate(s::StridedSubblocks, i::Int = 1)
    i > length(s) && return nothing
    return @inbounds(s[i]), i + 1
end
