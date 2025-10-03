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

function Base.getindex(
        t::AdjointTensorMap{T, S, N₁, N₂}, f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}
    ) where {T, S, N₁, N₂, I}
    tp = parent(t)
    subblock = getindex(tp, f₂, f₁)
    return permutedims(conj(subblock), (domainind(tp)..., codomainind(tp)...))
end
function Base.setindex!(
        t::AdjointTensorMap{T, S, N₁, N₂}, v, f₁::FusionTree{I, N₁}, f₂::FusionTree{I, N₂}
    ) where {T, S, N₁, N₂, I}
    return copy!(getindex(t, f₁, f₂), v)
end

# Show
#------
function Base.summary(io::IO, t::AdjointTensorMap)
    return print(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
end
function Base.show(io::IO, t::AdjointTensorMap)
    if get(io, :compact, false)
        print(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), "):")
    if sectortype(t) === Trivial
        Base.print_array(io, t[])
        println(io)
    elseif FusionStyle(sectortype(t)) isa UniqueFusion
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for sector ", f₁.uncoupled, " ← ", f₂.uncoupled, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    else
        for (f₁, f₂) in fusiontrees(t)
            println(io, "* Data for fusiontree ", f₁, " ← ", f₂, ":")
            Base.print_array(io, t[f₁, f₂])
            println(io)
        end
    end
    return nothing
end
