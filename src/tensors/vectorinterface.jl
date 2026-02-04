# scalartype
#------------
VectorInterface.scalartype(::Type{TT}) where {T, TT <: AbstractTensorMap{T}} = scalartype(T)

# zerovector & zerovector!!
#---------------------------
function VectorInterface.zerovector(t::AbstractTensorMap, ::Type{S}) where {S <: Number}
    return zerovector!(similar(t, S))
end
function VectorInterface.zerovector!(t::AbstractTensorMap)
    for (c, b) in blocks(t)
        zerovector!(b)
    end
    return t
end
function VectorInterface.zerovector!(t::TensorMap)
    zerovector!(t.data)
    return t
end
VectorInterface.zerovector!!(t::AbstractTensorMap) = zerovector!(t)

# scale, scale! & scale!!
#-------------------------
function VectorInterface.scale(t::AbstractTensorMap, α::Number)
    T = VectorInterface.promote_scale(t, α)
    return scale!(zerovector(t, T), t, α)
end
function VectorInterface.scale!(t::AbstractTensorMap, α::Number)
    for (c, b) in blocks(t)
        scale!(b, α)
    end
    return t
end
function VectorInterface.scale!(t::TensorMap, α::Number)
    scale!(t.data, α)
    return t
end
function VectorInterface.scale!!(t::AbstractTensorMap, α::Number)
    α === One() && return t
    T = VectorInterface.promote_scale(t, α)
    return T <: scalartype(t) ? scale!(t, α) : scale(t, α)
end
function VectorInterface.scale!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    for ((cy, by), (cx, bx)) in zip(blocks(ty), blocks(tx))
        scale!(by, bx, α)
    end
    return ty
end
function VectorInterface.scale!(ty::TensorMap, tx::TensorMap, α::Number)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    scale!(ty.data, tx.data, α)
    return ty
end
function VectorInterface.scale!!(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number)
    T = VectorInterface.promote_scale(tx, α)
    if T <: scalartype(ty)
        return scale!(ty, tx, α)
    else
        return scale(tx, α)
    end
end

# add, add! & add!!
#-------------------
# TODO: remove VectorInterface from calls to `add!` when `TensorKit.add!` is renamed
function VectorInterface.add(ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number, β::Number)
    S = check_spacetype(ty, tx)
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))

    # result type defaults to TensorMap if the types don't match to avoid assymmetric
    # implementation via zerovector(ty, T) vs zerovector(tx, T)
    # This would give issues for example with DiagonalTensorMap + TensorMap
    T = VectorInterface.promote_add(ty, tx, α, β)
    tdst = if typeof(ty) === typeof(tx)
        zerovector(ty, T)
    else
        M = promote_storagetype(similarstoragetype(ty, T), similarstoragetype(tx, T))
        tensormaptype(S, numout(ty), numin(ty), M)(undef, space(ty))
    end

    return add!(scale!(tdst, ty, β), tx, α)
end
function VectorInterface.add!(
        ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number, β::Number
    )
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    for ((cy, by), (cx, bx)) in zip(blocks(ty), blocks(tx))
        add!(by, bx, α, β)
    end
    return ty
end
function VectorInterface.add!(
        ty::TensorMap, tx::TensorMap, α::Number, β::Number
    )
    space(ty) == space(tx) || throw(SpaceMismatch("$(space(ty)) ≠ $(space(tx))"))
    add!(ty.data, tx.data, α, β)
    return ty
end
function VectorInterface.add!!(
        ty::AbstractTensorMap, tx::AbstractTensorMap, α::Number, β::Number
    )
    # spacecheck is done in add(!)
    T = VectorInterface.promote_add(ty, tx, α, β)
    if T <: scalartype(ty)
        return add!(ty, tx, α, β)
    else
        return add(ty, tx, α, β)
    end
end

# inner
#-------
function VectorInterface.inner(tx::AbstractTensorMap, ty::AbstractTensorMap)
    space(tx) == space(ty) || throw(SpaceMismatch("$(space(tx)) ≠ $(space(ty))"))
    InnerProductStyle(tx) === EuclideanInnerProduct() || throw_invalid_innerproduct(:inner)
    T = VectorInterface.promote_inner(tx, ty)
    s = zero(T)
    for ((cx, bx), (cy, by)) in zip(blocks(tx), blocks(ty))
        s += convert(T, dim(cx)) * inner(bx, by)
    end
    return s
end
function VectorInterface.inner(tx::TensorMap, ty::TensorMap)
    space(tx) == space(ty) || throw(SpaceMismatch("$(space(tx)) ≠ $(space(ty))"))
    InnerProductStyle(tx) === EuclideanInnerProduct() || throw_invalid_innerproduct(:inner)
    if FusionStyle(sectortype(tx)) isa UniqueFusion # all quantum dimensions are one
        return inner(tx.data, ty.data)
    else
        T = VectorInterface.promote_inner(tx, ty)
        s = zero(T)
        for c in blocksectors(tx)
            bx = parent(block(tx, c)) # matrix structure (reshape) does not matter
            by = parent(block(ty, c)) # but does lead to slower path in inner
            s += convert(T, dim(c)) * inner(bx, by)
        end
    end
    return s
end
# TODO: do we need a fast path for `AdjointTensorMap` instances?
