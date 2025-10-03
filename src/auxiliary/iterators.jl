struct OneOrNoneIterator{T}
    cond::Bool
    first::T
end

function Base.iterate(it::OneOrNoneIterator, state = true)
    if state && it.cond
        return (it.first, false)
    else
        return nothing
    end
end

Base.IteratorEltype(::Type{<:OneOrNoneIterator}) = Base.HasEltype()
Base.IteratorSize(::Type{<:OneOrNoneIterator}) = Base.HasLength()

Base.isempty(it::OneOrNoneIterator) = !it.cond
Base.length(it::OneOrNoneIterator) = Int(it.cond)
Base.eltype(::OneOrNoneIterator{T}) where {T} = T
