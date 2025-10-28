# AdjointTensorMap: lazy adjoint
#==========================================================#
"""
    struct AdjointTensorMap{T, S, N₁, N₂, TT<:AbstractTensorMap} <: AbstractTensorMap{T, S, N₁, N₂}

Specific subtype of [`AbstractTensorMap`](@ref) that is a lazy wrapper for representing the
adjoint of an instance of [`AbstractTensorMap`](@ref).
"""
struct AdjointTensorMap{T, S, N₁, N₂, TT <: AbstractTensorMap{T, S, N₂, N₁}} <:
    AbstractTensorMap{T, S, N₁, N₂}
    parent::TT
end
Base.parent(t::AdjointTensorMap) = t.parent

# Constructor: construct from taking adjoint of a tensor
Base.adjoint(t::AdjointTensorMap) = parent(t)
Base.adjoint(t::AbstractTensorMap) = AdjointTensorMap(t)

# Properties
space(t::AdjointTensorMap) = adjoint(space(parent(t)))
dim(t::AdjointTensorMap) = dim(parent(t))
storagetype(::Type{AdjointTensorMap{T, S, N₁, N₂, TT}}) where {T, S, N₁, N₂, TT} = storagetype(TT)

# Blocks and subblocks
#----------------------
block(t::AdjointTensorMap, s::Sector) = block(parent(t), s)'

blocks(t::AdjointTensorMap) = BlockIterator(t, blocks(parent(t)))

function blocktype(::Type{AdjointTensorMap{T, S, N₁, N₂, TT}}) where {T, S, N₁, N₂, TT}
    return Base.promote_op(adjoint, blocktype(TT))
end

function Base.iterate(iter::BlockIterator{<:AdjointTensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, b), newstate = next
    return c => adjoint(b), newstate
end

function Base.getindex(iter::BlockIterator{<:AdjointTensorMap}, c::Sector)
    return adjoint(Base.getindex(iter.structure, c))
end

Base.@propagate_inbounds function subblock(t::AdjointTensorMap, (f₁, f₂)::Tuple{FusionTree, FusionTree})
    tp = parent(t)
    data = subblock(tp, (f₂, f₁))
    return permutedims(conj(data), (domainind(tp)..., codomainind(tp)...))
end

# Show
#------
function Base.showarg(io::IO, t::AdjointTensorMap, toplevel::Bool)
    print(io, "adjoint(")
    Base.showarg(io, parent(t), false)
    print(io, ")")
    return nothing
end
