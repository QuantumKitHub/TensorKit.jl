Mooncake.arrayify(A_dA::CoDual{<:TensorMap}) = arrayify(primal(A_dA), tangent(A_dA))
Mooncake.arrayify(A::TensorMap, dA::TensorMap) = (A, dA)

function Mooncake.arrayify(Aᴴ_ΔAᴴ::CoDual{<:TK.AdjointTensorMap})
    Aᴴ = Mooncake.primal(Aᴴ_ΔAᴴ)
    ΔAᴴ = Mooncake.tangent(Aᴴ_ΔAᴴ)
    A_ΔA = CoDual(Aᴴ', ΔAᴴ.data.parent)
    A, ΔA = arrayify(A_ΔA)
    return A', ΔA'
end

# Define the tangent type of a TensorMap to be TensorMap itself.
# This has a number of benefits, but also correctly alters the
# inner product when dealing with non-abelian symmetries.
#
# Note: this implementation is technically a little lazy, since we are
# assuming that the tangent type of the underlying storage is also given
# by that same type. This should in principle work out fine, and will only
# fail for types that would be self-referential, which we choose to not support
# for now.

Mooncake.@foldable Mooncake.tangent_type(::Type{T}, ::Type{NoRData}) where {T <: TensorMap} = T
Mooncake.@foldable Mooncake.tangent_type(::Type{TensorMap{T, S, N₁, N₂, A}}) where {T, S, N₁, N₂, A} =
    TK.tensormaptype(S, N₁, N₂, Mooncake.tangent_type(A))

Mooncake.@foldable Mooncake.fdata_type(::Type{T}) where {T <: TensorMap} = Mooncake.tangent_type(T)
Mooncake.@foldable Mooncake.rdata_type(::Type{T}) where {T <: TensorMap} = NoRData

Mooncake.tangent(t::TensorMap, ::NoRData) = t
Mooncake.zero_tangent_internal(t::TensorMap, c::Mooncake.MaybeCache) =
    TensorMap(Mooncake.zero_tangent_internal(t.data, c), space(t))

Mooncake.randn_tangent_internal(rng::AbstractRNG, p::TensorMap, c::Mooncake.MaybeCache) =
    TensorMap(Mooncake.randn_tangent_internal(rng, p.data, c), space(p))

Mooncake.set_to_zero_internal!!(::Mooncake.SetToZeroCache, t::TensorMap) = zerovector!(t)
function Mooncake.increment!!(x::TensorMap, y::TensorMap)
    data = Mooncake.increment!!(x.data, y.data)
    return x.data === data ? x : TensorMap(data, space(x))
end
function Mooncake.increment_internal!!(c::Mooncake.IncCache, x::TensorMap, y::TensorMap)
    data = Mooncake.increment_internal!!(c, x.data, y.data)
    return x.data === data ? x : TensorMap(data, space(x))
end

Mooncake._add_to_primal_internal(c::Mooncake.MaybeCache, p::TensorMap, t::TensorMap, unsafe::Bool) =
    TensorMap(Mooncake._add_to_primal_internal(c, p.data, t.data, unsafe), space(p))
function Mooncake.tangent_to_primal_internal!!(p::TensorMap, t::TensorMap, c::Mooncake.MaybeCache)
    data = Mooncake.tangent_to_primal_internal!!(p.data, t.data, c)
    data === p.data || copy!(p.data, data)
    return p
end
Mooncake.primal_to_tangent_internal!!(t::T, p::T, ::Mooncake.MaybeCache) where {T <: TensorMap} = copy!(t, p)

Mooncake._dot_internal(::Mooncake.MaybeCache, t::TensorMap, s::TensorMap) = Float64(real(inner(t, s)))
Mooncake._scale_internal(::Mooncake.MaybeCache, a::Float64, t::TensorMap) = scale(t, a)

Mooncake.TestUtils.populate_address_map_internal(m::Mooncake.TestUtils.AddressMap, primal::TensorMap, tangent::TensorMap) =
    Mooncake.populate_address_map_internal(m, primal.data, tangent.data)
@inline Mooncake.TestUtils.__get_data_field(t::TensorMap, n) = getfield(t, n)

function Mooncake.__verify_fdata_value(::IdDict{Any, Nothing}, p::TensorMap, f::TensorMap)
    space(p) == space(f) ||
        throw(Mooncake.InvalidFDataException(lazy"p has space $(space(p)) but f has size $(space(f))"))
    return nothing
end
function Mooncake.__verify_fdata_value(c::IdDict{Any, Nothing}, p::TensorMap, t::TensorMap)
    return Mooncake.__verify_fdata_value(c, p.data, t.data)
end

@is_primitive MinimalCtx Tuple{typeof(Mooncake.lgetfield), <:TensorMap, Val}

# TODO: double-check if this has to include quantum dimensinos for non-abelian?
function Mooncake.frule!!(
        ::Dual{typeof(Mooncake.lgetfield)}, t::Dual{<:TensorMap}, ::Dual{Val{FieldName}}
    ) where {FieldName}
    y = getfield(primal(t), FieldName)

    return if FieldName === 1 || FieldName === :data
        dval = tangent(t).data
        Dual(val, dval)
    elseif FieldName === 2 || FieldName === :space
        Dual(val, NoFData()), getfield_pullback
    else
        throw(ArgumentError(lazy"Invalid fieldname `$FieldName`"))
    end
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake.lgetfield)}, t::CoDual{<:TensorMap}, ::CoDual{Val{FieldName}}
    ) where {FieldName}
    val = getfield(primal(t), FieldName)
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 3))

    return if FieldName === 1 || FieldName === :data
        dval = Mooncake.tangent(t).data
        CoDual(val, dval), getfield_pullback
    elseif FieldName === 2 || FieldName === :space
        Mooncake.zero_fcodual(val), getfield_pullback
    else
        throw(ArgumentError(lazy"Invalid fieldname `$FieldName`"))
    end
end

@is_primitive MinimalCtx Tuple{typeof(getfield), <:TensorMap, Any, Vararg{Symbol}}

Base.@constprop :aggressive function Mooncake.frule!!(
        ::Dual{typeof(getfield)}, t::Dual{<:TensorMap}, name::Dual
    )
    y = getfield(primal(t), primal(name))

    return if primal(name) === 1 || primal(name) === :data
        dval = tangent(t).data
        Dual(val, dval)
    elseif primal(name) === 2 || primal(name) === :space
        Dual(val, NoFData())
    else
        throw(ArgumentError(lazy"Invalid fieldname `$(primal(name))`"))
    end
end

Base.@constprop :aggressive function Mooncake.rrule!!(
        ::CoDual{typeof(getfield)}, t::CoDual{<:TensorMap}, name::CoDual
    )
    val = getfield(primal(t), primal(name))
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 3))

    return if primal(name) === 1 || primal(name) === :data
        dval = Mooncake.tangent(t).data
        CoDual(val, dval), getfield_pullback
    elseif primal(name) === 2 || primal(name) === :space
        Mooncake.zero_fcodual(val), getfield_pullback
    else
        throw(ArgumentError(lazy"Invalid fieldname `$(primal(name))`"))
    end
end

Base.@constprop :aggressive function Mooncake.frule!!(
        ::Dual{typeof(getfield)}, t::Dual{<:TensorMap}, name::Dual, order::Dual
    )
    y = getfield(primal(t), primal(name), primal(order))

    return if primal(name) === 1 || primal(name) === :data
        dval = tangent(t).data
        Dual(val, dval)
    elseif primal(name) === 2 || primal(name) === :space
        Dual(val, NoFData())
    else
        throw(ArgumentError(lazy"Invalid fieldname `$(primal(name))`"))
    end
end

Base.@constprop :aggressive function Mooncake.rrule!!(
        ::CoDual{typeof(getfield)}, t::CoDual{<:TensorMap}, name::CoDual, order::CoDual
    )
    val = getfield(primal(t), primal(name), primal(order))
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 4))

    return if primal(name) === 1 || primal(name) === :data
        dval = Mooncake.tangent(t).data
        CoDual(val, dval), getfield_pullback
    elseif primal(name) === 2 || primal(name) === :space
        Mooncake.zero_fcodual(val), getfield_pullback
    else
        throw(ArgumentError(lazy"Invalid fieldname `$(primal(name))`"))
    end
end


@is_primitive MinimalCtx Tuple{typeof(Mooncake.lgetfield), <:TensorMap, Val, Val}

# TODO: double-check if this has to include quantum dimensinos for non-abelian?
function Mooncake.frule!!(
        ::Dual{typeof(Mooncake.lgetfield)}, t::Dual{<:TensorMap}, ::Dual{Val{FieldName}}, ::Dual{Val{Order}}
    ) where {FieldName, Order}
    y = getfield(primal(t), FieldName, Order)

    return if FieldName === 1 || FieldName === :data
        dval = tangent(t).data
        Dual(val, dval)
    elseif FieldName === 2 || FieldName === :space
        Dual(val, NoFData())
    else
        throw(ArgumentError(lazy"Invalid fieldname `$FieldName`"))
    end
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake.lgetfield)}, t::CoDual{<:TensorMap}, ::CoDual{Val{FieldName}}, ::CoDual{Val{Order}}
    ) where {FieldName, Order}
    val = getfield(primal(t), FieldName, Order)
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 4))

    return if FieldName === 1 || FieldName === :data
        dval = Mooncake.tangent(t).data
        CoDual(val, dval), getfield_pullback
    elseif FieldName === 2 || FieldName === :space
        Mooncake.zero_fcodual(val), getfield_pullback
    else
        throw(ArgumentError(lazy"Invalid fieldname `$FieldName`"))
    end
end


Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(Mooncake._new_), Type{TensorMap{T, S, N₁, N₂, A}}, UndefInitializer, TensorMapSpace{S, N₁, N₂}} where {T, S, N₁, N₂, A}
@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake._new_), Type{TensorMap{T, S, N₁, N₂, A}}, A, TensorMapSpace{S, N₁, N₂}} where {T, S, N₁, N₂, A}

function Mooncake.frule!!(
        ::Dual{typeof(Mooncake._new_)}, ::Dual{Type{TensorMap{T, S, N₁, N₂, A}}}, data::Dual{A}, space::Dual{TensorMapSpace{S, N₁, N₂}}
    ) where {T, S, N₁, N₂, A}
    t = Mooncake._new_(TensorMap{T, S, N₁, N₂, A}, primal(data), primal(space))
    dt = Mooncake._new_(TensorMap{T, S, N₁, N₂, A}, tangent(data), primal(space))
    return Dual(t, dt)
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake._new_)}, ::CoDual{Type{TensorMap{T, S, N₁, N₂, A}}}, data::CoDual{A}, space::CoDual{TensorMapSpace{S, N₁, N₂}}
    ) where {T, S, N₁, N₂, A}
    return Mooncake.zero_fcodual(Mooncake._new_(TensorMap{T, S, N₁, N₂, A}, primal(data), primal(space))),
        Returns(ntuple(Returns(NoRData()), 4))
end
