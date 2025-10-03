"""
    fusiontrees(uncoupled::NTuple{N,I}[,
        coupled::I=one(I)[, isdual::NTuple{N,Bool}=ntuple(n -> false, length(uncoupled))]])
        where {N,I<:Sector} -> FusionTreeIterator{I,N,I}

Return an iterator over all fusion trees with a given coupled sector label `coupled` and
uncoupled sector labels and isomorphisms `uncoupled` and `isdual` respectively.
"""
function fusiontrees(
        uncoupled::NTuple{N, I}, coupled::I, isdual::NTuple{N, Bool}
    ) where {N, I <: Sector}
    uncouplediterators = map(tuple, uncoupled)
    return FusionTreeIterator(uncouplediterators, coupled, isdual)
end
function fusiontrees(uncoupled::Tuple{Vararg{I}}, coupled::I) where {I <: Sector}
    isdual = ntuple(n -> false, length(uncoupled))
    return fusiontrees(uncoupled, coupled, isdual)
end
function fusiontrees(uncoupled::Tuple{I, Vararg{I}}) where {I <: Sector}
    coupled = one(I)
    isdual = ntuple(n -> false, length(uncoupled))
    return fusiontrees(uncoupled, coupled, isdual)
end

# # make sectors iteratable in the same way as numbers (implementation from Base)
# Base.iterate(s::Sector) = (s, nothing)
# Base.iterate(s::Sector, ::Any) = nothing
# TODO: reconsider whether this is desirable; currently it conflicts with the iteration of `ProductSector`

struct FusionTreeIterator{I <: Sector, N, T <: NTuple{N}}
    uncouplediterators::T # iterators over uncoupled sectors
    coupled::I
    isdual::NTuple{N, Bool}
end

Base.IteratorSize(::FusionTreeIterator) = Base.SizeUnknown()
Base.IteratorEltype(::FusionTreeIterator) = Base.HasEltype()
Base.eltype(::Type{<:FusionTreeIterator{I, N}}) where {I <: Sector, N} = fusiontreetype(I, N)

Base.length(iter::FusionTreeIterator) = _fusiondim(iter.uncouplediterators, iter.coupled)
_fusiondim(::Tuple{}, c::I) where {I <: Sector} = Int(isone(c))
_fusiondim(iters::NTuple{1}, c::I) where {I <: Sector} = Int(c ∈ iters[1])
function _fusiondim(iters::NTuple{2}, c::I) where {I <: Sector}
    d = 0
    for a in iters[1], b in iters[2]
        d += Int(Nsymbol(a, b, c))
    end
    return d
end
function _fusiondim(iters, c::I) where {I <: Sector}
    d = 0
    N = length(iters)
    for b in iters[N]
        for a in c ⊗ dual(b)
            d += Nsymbol(a, b, c) * _fusiondim(Base.front(iters), a)
        end
    end
    return d
end

# * Iterator methods:
#   Start with special cases:
function Base.iterate(it::FusionTreeIterator{I, 0}, state = !isone(it.coupled)) where {I <: Sector}
    state && return nothing
    tree = FusionTree{I}((), it.coupled, (), (), ())
    return tree, true
end

function Base.iterate(
        it::FusionTreeIterator{I, 1}, state = !(it.coupled ∈ it.uncouplediterators[1])
    ) where {I <: Sector}
    state && return nothing
    tree = FusionTree{I}((it.coupled,), it.coupled, it.isdual, (), ())
    return tree, true
end

#   General case:
function Base.iterate(it::FusionTreeIterator{I}) where {I <: Sector}
    coupled = it.coupled
    next = _fusiontree_iterate(it.uncouplediterators, coupled)
    next === nothing && return nothing
    out, lines, vertices, states = next
    f = FusionTree{I}(out, coupled, it.isdual, lines, vertices)
    return f, (out, lines, vertices, states)
end
function Base.iterate(it::FusionTreeIterator{I}, (out, lines, vertices, states)) where {I <: Sector}
    coupled = it.coupled
    next = _fusiontree_iterate(it.uncouplediterators, coupled, out, lines, vertices, states)
    next === nothing && return nothing
    out, lines, vertices, states = next
    f = FusionTree{I}(out, coupled, it.isdual, lines, vertices)
    return f, (out, lines, vertices, states)
end

function _fusiontree_iterate(uncoupledsectors::NTuple{2}, c::I) where {I <: Sector}
    outiter1 = uncoupledsectors[1]
    outiter2 = uncoupledsectors[2]
    nextout2 = iterate(outiter2)
    nextout2 === nothing && return nothing
    b, outstate2 = nextout2
    nextout1 = iterate(outiter1)
    nextout1 === nothing && return nothing
    a, outstate1 = nextout1
    while Nsymbol(a, b, c) == 0
        nextout1 = iterate(outiter1, outstate1)
        if isnothing(nextout1)
            nextout2 = iterate(outiter2, outstate2)
            nextout2 === nothing && return nothing
            b, outstate2 = nextout2
            nextout1 = iterate(outiter1)
        end
        a, outstate1 = nextout1
    end
    n = 1
    return (a, b), (), (n,), (outstate1, outstate2)
end

function _fusiontree_iterate(uncoupledsectors::NTuple{2}, c::I, out, lines, vertices, states) where {I <: Sector}
    a, b = out
    n = vertices[1]
    n < Nsymbol(a, b, c) && return out, lines, (n + 1,), states
    outiter1 = uncoupledsectors[1]
    outiter2 = uncoupledsectors[2]
    outstate1, outstate2 = states
    nextout1 = iterate(outiter1, outstate1)
    if isnothing(nextout1)
        nextout2 = iterate(outiter2, outstate2)
        nextout2 === nothing && return nothing
        b, outstate2 = nextout2
        nextout1 = iterate(outiter1)
    end
    a, outstate1 = nextout1
    while Nsymbol(a, b, c) == 0
        nextout1 = iterate(outiter1, outstate1)
        if isnothing(nextout1)
            nextout2 = iterate(outiter2, outstate2)
            nextout2 === nothing && return nothing
            b, outstate2 = nextout2
            nextout1 = iterate(outiter1)
        end
        a, outstate1 = nextout1
    end
    n = 1
    return (a, b), (), (n,), (outstate1, outstate2)
end

function _fusiontree_iterate(uncoupledsectors::NTuple{N}, coupled::I) where {N, I <: Sector}
    outiterN = uncoupledsectors[N]
    nextout = iterate(outiterN)
    nextout === nothing && return nothing
    b, outstateN = nextout
    vertexiterN = coupled ⊗ dual(b)
    nextline = iterate(vertexiterN)
    while isnothing(nextline)
        nextout = iterate(outiterN, outstateN)
        nextout === nothing && return nothing
        b, outstateN = nextout
        vertexiterN = coupled ⊗ dual(b)
        nextline = iterate(vertexiterN)
    end
    a, vertexstateN = nextline
    rest = _fusiontree_iterate(Base.front(uncoupledsectors), a)
    while isnothing(rest)
        nextline = iterate(vertexiterN, vertexstateN)
        while isnothing(nextline)
            nextout = iterate(outiterN, outstateN)
            nextout === nothing && return nothing
            b, outstateN = nextout
            vertexiterN = coupled ⊗ dual(b)
            nextline = iterate(vertexiterN)
        end
        a, vertexstateN = nextline
        rest = _fusiontree_iterate(Base.front(uncoupledsectors), a)
    end
    n = 1
    restout, restlines, restvertices, reststates = rest
    out = (restout..., b)
    lines = (restlines..., a)
    vertices = (restvertices..., n)
    states = (reststates..., vertexstateN, outstateN)
    return out, lines, vertices, states
end

function _fusiontree_iterate(uncoupledsectors::NTuple{N}, coupled::I, out, lines, vertices, states) where {N, I <: Sector}
    a = lines[end]
    b = out[end]
    c = coupled
    restout = Base.front(out)
    restlines = Base.front(lines)
    restvertices = Base.front(vertices)
    reststates = Base.front(Base.front(states))
    rest = _fusiontree_iterate(
        Base.front(uncoupledsectors), a, restout, restlines, restvertices, reststates
    )
    outiterN = uncoupledsectors[N]
    vertexiterN = c ⊗ dual(b)
    outstateN = states[end]
    vertexstateN = states[end - 1]
    n = vertices[end]
    while isnothing(rest)
        if n < Nsymbol(a, b, c)
            n += 1
            # reset the first part of the fusion tree
            rest = _fusiontree_iterate(Base.front(uncoupledsectors), a)
        else
            nextline = iterate(vertexiterN, vertexstateN)
            while isnothing(nextline)
                nextout = iterate(outiterN, outstateN)
                nextout === nothing && return nothing
                b, outstateN = nextout
                vertexiterN = c ⊗ dual(b)
                nextline = iterate(vertexiterN)
            end
            a, vertexstateN = nextline
            n = 1
            rest = _fusiontree_iterate(Base.front(uncoupledsectors), a)
        end
    end
    restout, restlines, restvertices, reststates = rest
    out = (restout..., b)
    lines = (restlines..., a)
    vertices = (restvertices..., n)
    states = (reststates..., vertexstateN, outstateN)
    return out, lines, vertices, states
end
