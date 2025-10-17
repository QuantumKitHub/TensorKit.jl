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
function foreachblock(f, t::AbstractTensorMap, ts::AbstractTensorMap...; scheduler = nothing)
    tensors = (t, ts...)
    allsectors = union(blocksectors.(tensors)...)
    foreach(allsectors) do c
        return f(c, block.(tensors, Ref(c)))
    end
    return nothing
end
function foreachblock(f, t::AbstractTensorMap; scheduler = nothing)
    foreach(blocks(t)) do (c, b)
        return f(c, (b,))
    end
    return nothing
end

function show_blocks(io, mime::MIME"text/plain", iter)
    first = true
    for (c, b) in iter
        first || print(io, "\n\n")
        print(io, " * ", c, " => ")
        show(io, mime, b)
        first = false
    end
    return nothing
end

function show_blocks(io, iter)
    print(io, "(")
    join(io, iter, ", ")
    print(io, ")")
    return nothing
end
