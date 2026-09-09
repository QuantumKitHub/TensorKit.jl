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

"""
    struct SubblockIterator{T <: AbstractTensorMap, S}

Iterator over the subblocks of a tensor of type `T`, possibly holding some pre-computed data of type `S`.
This is typically constructed through of [`subblocks`](@ref).
"""
struct SubblockIterator{T <: AbstractTensorMap, S}
    t::T
    structure::S
end

Base.IteratorSize(::SubblockIterator) = Base.HasLength()
Base.IteratorEltype(::SubblockIterator) = Base.HasEltype()
Base.eltype(::Type{<:SubblockIterator{T}}) where {T} = Pair{fusiontreetype(T), subblocktype(T)}
Base.length(iter::SubblockIterator) = length(iter.structure)
Base.isdone(iter::SubblockIterator, state...) = Base.isdone(iter.structure, state...)

# default implementation assumes `structure = fusiontrees(t)`
function Base.iterate(iter::SubblockIterator, state...)
    next = Base.iterate(iter.structure, state...)
    isnothing(next) && return nothing
    (f₁, f₂), state = next
    @inbounds data = subblock(iter.t, (f₁, f₂))
    return (f₁, f₂) => data, state
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
    struct StridedSubblocks{A <: DenseVector, N, F}
    StridedSubblocks(t::TensorMap, [op = identity])

Sector-independent, integer-indexable collection of the subblocks of a `TensorMap`, as
`StridedView`s into its flat data vector. Subblock `i` corresponds to the `i`th fusion tree pair
in the canonical order of `fusiontrees(space(t))`, see also [`fusiontreeindices`](@ref).
The operation `op` (`identity` or `conj`) is applied lazily to every view, which allows
representing the subblocks of a conjugated tensor without materializing it.

This is the data structure consumed by the index manipulation kernels, whose type does not
depend on the sectortype of `t`.
"""
const SubblockOp = Union{typeof(identity), typeof(conj)}
struct StridedSubblocks{A <: DenseVector, N, F <: SubblockOp}
    data::A
    structure::Vector{StridedStructure{N}}
    op::F
end
Base.length(s::StridedSubblocks) = length(s.structure)
Base.firstindex(s::StridedSubblocks) = 1
Base.lastindex(s::StridedSubblocks) = length(s)
Base.eltype(::Type{S}) where {S <: StridedSubblocks} = Core.Compiler.return_type(getindex, Tuple{S, Int})

Base.@propagate_inbounds function Base.getindex(s::StridedSubblocks, i::Int)
    sz, str, offset = s.structure[i]
    return StridedView(s.data, sz, str, offset, s.op)
end

function Base.iterate(s::StridedSubblocks, i::Int = 1)
    i > length(s) && return nothing
    return @inbounds(s[i]), i + 1
end

storagetype(::Type{StridedSubblocks{A, N, F}}) where {A, N, F} = A

"""
    struct TreeSubblocks{TT <: AbstractTensorMap, I, F}
    TreeSubblocks(t::AbstractTensorMap, [op = identity])

Integer-indexable collection of the subblocks of an arbitrary tensor `t`, where position `i`
refers to the `i`th fusion tree pair of `fusiontrees(space(t))` and the data is retrieved through
[`subblock`](@ref), with `op` (`identity` or `conj`) applied. This is the generic counterpart of
[`StridedSubblocks`](@ref) for tensor types that do not store their data in a flat vector.
"""
struct TreeSubblocks{TT <: AbstractTensorMap, I, F <: SubblockOp}
    t::TT
    trees::I
    op::F
end
function TreeSubblocks(t::AbstractTensorMap, op::SubblockOp = identity)
    return TreeSubblocks(t, fusiontrees(t), scalartype(t) <: Real ? identity : op)
end

storagetype(::Type{TreeSubblocks{TT, I, F}}) where {TT, I, F} = storagetype(TT)

Base.length(s::TreeSubblocks) = length(s.trees)
Base.firstindex(s::TreeSubblocks) = 1
Base.lastindex(s::TreeSubblocks) = length(s)
Base.eltype(::Type{S}) where {S <: TreeSubblocks} = Core.Compiler.return_type(getindex, Tuple{S, Int})

Base.@propagate_inbounds function Base.getindex(s::TreeSubblocks, i::Int)
    return s.op(subblock(s.t, gettokenvalue(s.trees, i)))
end

function Base.iterate(s::TreeSubblocks, i::Int = 1)
    i > length(s) && return nothing
    return @inbounds(s[i]), i + 1
end
