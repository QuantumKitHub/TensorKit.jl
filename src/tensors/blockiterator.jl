"""
    struct BlockIterator{T<:AbstractTensorMap,S}

Iterator over the blocks of type `T`, possibly holding some pre-computed data of type `S`
"""
struct BlockIterator{T<:AbstractTensorMap,S}
    t::T
    structure::S
end

Base.IteratorSize(::BlockIterator) = Base.HasLength()
Base.IteratorEltype(::BlockIterator) = Base.HasEltype()
Base.eltype(::Type{<:BlockIterator{T}}) where {T} = Pair{sectortype(T),blocktype(T)}
Base.length(iter::BlockIterator) = length(iter.structure)
Base.isdone(iter::BlockIterator, state...) = Base.isdone(iter.structure, state...)

# TODO: fast-path when structures are the same?
# TODO: do we want f(c, bs...) or f(c, bs)?
# TODO: implement scheduler
# TODO: do we prefer `blocks(t, ts...)` instead or as well?
"""
    foreachblock(f, t::AbstractTensorMap, ts::AbstractTensorMap...; [scheduler])

Apply `f` to each block of `t` and the corresponding blocks of `ts`.
Optionally, `scheduler` can be used to parallelize the computation.
This function is equivalent to the following loop:

```julia
for (c, b) in blocks(t)
    bs = (b, block.(ts, c)...)
    f(c, bs)
end
```
"""
function foreachblock(f, t::AbstractTensorMap, ts::AbstractTensorMap...; scheduler=nothing)
    foreach(blocks(t)) do (c, b)
        return f(c, (b, map(Base.Fix2(block, c), ts)...))
    end
    return nothing
end
