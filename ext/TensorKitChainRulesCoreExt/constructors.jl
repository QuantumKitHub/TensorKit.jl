@non_differentiable TensorKit.TensorMap(f::Function, storagetype, cod, dom)
@non_differentiable TensorKit.id(args...)
@non_differentiable TensorKit.isomorphism(args...)
@non_differentiable TensorKit.isometry(args...)
@non_differentiable TensorKit.unitary(args...)

function ChainRulesCore.rrule(::Type{TensorMap}, d::DenseArray, args...; kwargs...)
    function TensorMap_pullback(Δt)
        ∂d = convert(Array, unthunk(Δt))
        return NoTangent(), ∂d, ntuple(_ -> NoTangent(), length(args))...
    end
    return TensorMap(d, args...; kwargs...), TensorMap_pullback
end

# these are not the conversion to/from array, but actually take in data parameters
# -- as a result, requires quantum dimensions to keep inner product the same:
#   ⟨Δdata, ∂data⟩ = ⟨Δtensor, ∂tensor⟩ = ∑_c d_c ⟨Δtensor_c, ∂tensor_c⟩
#   ⟹ Δdata = d_c Δtensor_c
function ChainRulesCore.rrule(::Type{TensorMap{T}}, data::DenseVector, V::TensorMapSpace) where {T}
    t = TensorMap{T}(data, V)
    P = ProjectTo(data)
    function TensorMap_pullback(Δt_)
        Δt = copy(unthunk(Δt_))
        for (c, b) in blocks(Δt)
            scale!(b, dim(c))
        end
        ∂data = P(Δt.data)
        return NoTangent(), ∂data, NoTangent()
    end
    return t, TensorMap_pullback
end

function ChainRulesCore.rrule(::Type{<:DiagonalTensorMap}, data::DenseVector, args...; kwargs...)
    D = DiagonalTensorMap(data, args...; kwargs...)
    P = ProjectTo(data)
    function DiagonalTensorMap_pullback(Δt_)
        # unclear if we're allowed to modify/take ownership of the input
        Δt = copy(unthunk(Δt_))
        for (c, b) in blocks(Δt)
            scale!(b, dim(c))
        end
        ∂data = P(Δt.data)
        return NoTangent(), ∂data, NoTangent()
    end
    return D, DiagonalTensorMap_pullback
end

function ChainRulesCore.rrule(::Type{DiagonalTensorMap}, t::AbstractTensorMap)
    d = DiagonalTensorMap(t)
    function DiagonalTensorMap_pullback(Δd_)
        Δt = similar(t) # no projector needed
        for (c, b) in blocks(unthunk(Δd_))
            copy!(block(Δt, c), Diagonal(b))
        end
        return NoTangent(), Δt
    end
    return d, DiagonalTensorMap_pullback
end

function ChainRulesCore.rrule(::typeof(Base.getproperty), t::TensorMap, prop::Symbol)
    if prop === :data
        function getdata_pullback(Δdata)
            # unclear if we're allowed to modify/take ownership of the input
            t′ = typeof(t)(copy(unthunk(Δdata)), t.space)
            for (c, b) in blocks(t′)
                scale!(b, inv(dim(c)))
            end
            return NoTangent(), t′, NoTangent()
        end
        return t.data, getdata_pullback
    elseif prop === :space
        return t.space, Returns((NoTangent(), ZeroTangent(), NoTangent()))
    else
        throw(ArgumentError("unknown property $prop"))
    end
end

function ChainRulesCore.rrule(::typeof(Base.getproperty), t::DiagonalTensorMap, prop::Symbol)
    if prop === :data
        function getdata_pullback(Δdata)
            # unclear if we're allowed to modify/take ownership of the input
            t′ = typeof(t)(copy(unthunk(Δdata)), t.domain)
            for (c, b) in blocks(t′)
                scale!(b, inv(dim(c)))
            end
            return NoTangent(), t′, NoTangent()
        end
        return t.data, getdata_pullback
    elseif prop === :domain
        return t.domain, Returns((NoTangent(), ZeroTangent(), NoTangent()))
    else
        throw(ArgumentError("unknown property $prop"))
    end
end

function ChainRulesCore.rrule(::typeof(Base.copy), t::AbstractTensorMap)
    copy_pullback(Δt) = NoTangent(), Δt
    return copy(t), copy_pullback
end

function ChainRulesCore.rrule(
        ::typeof(TensorKit.copy_oftype), t::AbstractTensorMap, T::Type{<:Number}
    )
    project = ProjectTo(t)
    copy_oftype_pullback(Δt) = NoTangent(), project(unthunk(Δt)), NoTangent()
    return TensorKit.copy_oftype(t, T), copy_oftype_pullback
end

function ChainRulesCore.rrule(::typeof(Base.convert), T::Type{<:Array}, t::AbstractTensorMap)
    A = convert(T, t)
    function convert_pullback(ΔA)
        # use constructor to (unconditionally) project back onto symmetric subspace
        ∂t = TensorMap(unthunk(ΔA), codomain(t), domain(t); tol = Inf)
        return NoTangent(), NoTangent(), ∂t
    end
    return A, convert_pullback
end

function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{Dict}, t::AbstractTensorMap)
    out = convert(Dict, t)
    function convert_pullback(c′)
        c = unthunk(c′)
        if haskey(c, :data) # :data is the only thing for which this dual makes sense
            dual = copy(out)
            dual[:data] = c[:data]
            return (NoTangent(), NoTangent(), convert(TensorMap, dual))
        else
            # instead of zero(t) you can also return ZeroTangent(), which is type unstable
            return (NoTangent(), NoTangent(), zero(t))
        end
    end
    return out, convert_pullback
end
function ChainRulesCore.rrule(::typeof(Base.convert), ::Type{TensorMap}, t::Dict{Symbol, Any})
    return convert(TensorMap, t), v -> (NoTangent(), NoTangent(), convert(Dict, v))
end

function ChainRulesCore.rrule(T::Type{<:TensorKit.AdjointTensorMap}, t::AbstractTensorMap)
    return T(t), Δt -> (NoTangent(), adjoint(unthunk(Δt)))
end
