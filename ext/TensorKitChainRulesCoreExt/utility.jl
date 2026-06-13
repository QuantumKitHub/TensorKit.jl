TensorKit.block(t::ZeroTangent, c::Sector) = t

ChainRulesCore.ProjectTo(::T) where {T <: AbstractTensorMap} = ProjectTo{T}()
function (::ProjectTo{T1})(x::T2) where {
        S, N1, N2, T1 <: AbstractTensorMap{<:Any, S, N1, N2}, T2 <: AbstractTensorMap{<:Any, S, N1, N2},
    }
    T1 === T2 && return x
    y = similar(x, scalartype(T1))
    for (c, b) in blocks(y)
        p = ProjectTo(b)
        b .= p(block(x, c))
    end
    return y
end

function (::ProjectTo{DiagonalTensorMap{T, S, A}})(x::AbstractTensorMap) where {T, S, A}
    x isa DiagonalTensorMap{T, S, A} && return x
    V = space(x, 1)
    space(x) == (V ← V) || throw(SpaceMismatch())
    y = DiagonalTensorMap{T, S, A}(undef, V)
    for (c, b) in blocks(y)
        p = ProjectTo(b)
        b .= p(block(x, c))
    end
    return y
end
