# Arrayify is needed to make MatrixAlgebraKit function properly -
# it turns coduals into argument types that MAK knows how to handle.
Mooncake.arrayify(A_dA::CoDual{<:TensorMap}) = arrayify(primal(A_dA), tangent(A_dA))
Mooncake.arrayify(A::TensorMap, dA::TensorMap) = (A, dA)

Mooncake.arrayify(A_dA::CoDual{<:DiagonalTensorMap}) = arrayify(primal(A_dA), tangent(A_dA))
Mooncake.arrayify(A::DiagonalTensorMap, dA::DiagonalTensorMap) = (A, dA)

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

# Define the tangent types
# ------------------------
const DiagOrTensorMap = Union{TensorMap, DiagonalTensorMap}

Mooncake.@foldable Mooncake.tangent_type(::Type{T}, ::Type{NoRData}) where {T <: TensorMap} = T
Mooncake.@foldable Mooncake.tangent_type(::Type{TensorMap{T, S, N₁, N₂, A}}) where {T, S, N₁, N₂, A} =
    TK.tensormaptype(S, N₁, N₂, Mooncake.tangent_type(A))
Mooncake.@foldable Mooncake.tangent_type(::Type{T}, ::Type{NoRData}) where {T <: DiagonalTensorMap} = T
Mooncake.@foldable Mooncake.tangent_type(::Type{DiagonalTensorMap{T, S, A}}) where {T, S, A} =
    DiagonalTensorMap{T, S, Mooncake.tangent_type(A)}

Mooncake.@foldable Mooncake.fdata_type(::Type{T}) where {T <: DiagOrTensorMap} = Mooncake.tangent_type(T)
Mooncake.@foldable Mooncake.rdata_type(::Type{T}) where {T <: DiagOrTensorMap} = NoRData

Mooncake.tangent(t::DiagOrTensorMap, ::NoRData) = t


# Required tangent methods
# ------------------------
# note that the internal functions have to be overloaded to make sure that tangents for types that share data are correctly handled.
# E.g. the tangent for (t, t) should have a zero tangent (dt, dt), and not (dt1, dt2)
# The cache objects are similar to how Base.deepcopy works

# generate new tangents for accumulation
Mooncake.zero_tangent_internal(t::TensorMap, c::Mooncake.MaybeCache) =
    TensorMap(Mooncake.zero_tangent_internal(t.data, c), space(t))
Mooncake.zero_tangent_internal(t::DiagonalTensorMap, c::Mooncake.MaybeCache) =
    DiagonalTensorMap(Mooncake.zero_tangent_internal(t.data, c), space(t, 1))

# generate random tangents for testing
Mooncake.randn_tangent_internal(rng::AbstractRNG, p::TensorMap, c::Mooncake.MaybeCache) =
    TensorMap(Mooncake.randn_tangent_internal(rng, p.data, c), space(p))
Mooncake.randn_tangent_internal(rng::AbstractRNG, p::DiagonalTensorMap, c::Mooncake.MaybeCache) =
    DiagonalTensorMap(Mooncake.randn_tangent_internal(rng, p.data, c), space(p, 1))


function Mooncake.set_to_zero_internal!!(c::Mooncake.SetToZeroCache, t::TensorMap)
    data = Mooncake.set_to_zero_internal!!(c, t.data)
    return data === t.data ? t : TensorMap(data, space(t))
end
function Mooncake.set_to_zero_internal!!(c::Mooncake.SetToZeroCache, d::DiagonalTensorMap)
    data = Mooncake.set_to_zero_internal!!(c, d.data)
    return data === d.data ? d : DiagonalTensorMap(data, space(d, 1))
end

function Mooncake.increment!!(x::TensorMap, y::TensorMap)
    data = Mooncake.increment!!(x.data, y.data)
    return x.data === data ? x : TensorMap(data, space(x))
end
function Mooncake.increment_internal!!(c::Mooncake.IncCache, x::TensorMap, y::TensorMap)
    data = Mooncake.increment_internal!!(c, x.data, y.data)
    return x.data === data ? x : TensorMap(data, space(x))
end
function Mooncake.increment!!(x::DiagonalTensorMap, y::DiagonalTensorMap)
    data = Mooncake.increment!!(x.data, y.data)
    return x.data === data ? x : DiagonalTensorMap(data, space(x, 1))
end
function Mooncake.increment_internal!!(c::Mooncake.IncCache, x::DiagonalTensorMap, y::DiagonalTensorMap)
    data = Mooncake.increment_internal!!(c, x.data, y.data)
    return x.data === data ? x : DiagonalTensorMap(data, space(x, 1))
end

# methods for converting between tangents and primals:
# fuels the `friendly_tangents` feature in Mooncake
Mooncake._add_to_primal_internal(c::Mooncake.MaybeCache, p::TensorMap, t::TensorMap, unsafe::Bool) =
    TensorMap(Mooncake._add_to_primal_internal(c, p.data, t.data, unsafe), space(p))
function Mooncake.tangent_to_primal_internal!!(p::TensorMap, t::TensorMap, c::Mooncake.MaybeCache)
    data = Mooncake.tangent_to_primal_internal!!(p.data, t.data, c)
    data === p.data || copy!(p.data, data)
    return p
end
function Mooncake.primal_to_tangent_internal!!(t::TensorMap, p::TensorMap, c::Mooncake.MaybeCache)
    data = Mooncake.primal_to_tangent_internal!!(t.data, p.data, c)
    data === t.data || copy!(t.data, data)
    return t
end
Mooncake._add_to_primal_internal(c::Mooncake.MaybeCache, p::DiagonalTensorMap, t::DiagonalTensorMap, unsafe::Bool) =
    DiagonalTensorMap(Mooncake._add_to_primal_internal(c, p.data, t.data, unsafe), space(p))
function Mooncake.tangent_to_primal_internal!!(p::DiagonalTensorMap, t::DiagonalTensorMap, c::Mooncake.MaybeCache)
    data = Mooncake.tangent_to_primal_internal!!(p.data, t.data, c)
    data === p.data || copy!(p.data, data)
    return p
end
function Mooncake.primal_to_tangent_internal!!(t::TensorMap, p::TensorMap, c::Mooncake.MaybeCache)
    data = Mooncake.primal_to_tangent_internal!!(t.data, p.data, c)
    data === t.data || copy!(t.data, data)
    return p
end
function Mooncake.primal_to_tangent_internal!!(t::DiagonalTensorMap, p::DiagonalTensorMap, c::Mooncake.MaybeCache)
    data = Mooncake.primal_to_tangent_internal!!(t.data, p.data, c)
    data === t.data || copy!(t.data, data)
    return p
end

# to convert from/to chainrules tangents
Mooncake.to_cr_tangent(x::DiagOrTensorMap) = x

# Test utilities
# --------------

# to work with finite differences
Mooncake._dot_internal(::Mooncake.MaybeCache, t::TensorMap, s::TensorMap) = Float64(real(inner(t, s)))
Mooncake._dot_internal(::Mooncake.MaybeCache, t::DiagonalTensorMap, s::DiagonalTensorMap) = Float64(real(inner(t, s)))
Mooncake._scale_internal(::Mooncake.MaybeCache, a::Float64, t::DiagOrTensorMap) = scale(t, a)

# To verify that shared data is handled correctly
Mooncake.TestUtils.populate_address_map_internal(m::Mooncake.TestUtils.AddressMap, primal::TensorMap, tangent::TensorMap) =
    Mooncake.populate_address_map_internal(m, primal.data, tangent.data)
Mooncake.TestUtils.populate_address_map_internal(m::Mooncake.TestUtils.AddressMap, primal::DiagonalTensorMap, tangent::DiagonalTensorMap) =
    Mooncake.populate_address_map_internal(m, primal.data, tangent.data)

@inline Mooncake.TestUtils.__get_data_field(t::DiagOrTensorMap, n) = getfield(t, n)

function Mooncake.__verify_fdata_value(c::IdDict{Any, Nothing}, p::TensorMap, t::TensorMap)
    space(p) == space(t) ||
        throw(Mooncake.InvalidFDataException(lazy"p has space $(space(p)) but t has size $(space(t))"))
    return Mooncake.__verify_fdata_value(c, p.data, t.data)
end
function Mooncake.__verify_fdata_value(c::IdDict{Any, Nothing}, p::DiagonalTensorMap, t::DiagonalTensorMap)
    space(p) == space(t) ||
        throw(Mooncake.InvalidFDataException(lazy"p has space $(space(p)) but t has size $(space(t))"))
    return Mooncake.__verify_fdata_value(c, p.data, t.data)
end


# Custom rules for constructors/getters/setters
# ---------------------------------------------
@is_primitive MinimalCtx Tuple{typeof(Mooncake.lgetfield), <:DiagOrTensorMap, Val}

function Mooncake.frule!!(
        ::Dual{typeof(Mooncake.lgetfield)}, t::Dual{<:DiagOrTensorMap}, ::Dual{Val{FieldName}}
    ) where {FieldName}
    val = getfield(primal(t), FieldName)
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 3))

    return if FieldName === 1 || FieldName === :data
        dval = tangent(t).data
        Dual(val, dval)
    else # cannot be invalid fieldname since already called `getfield`
        Dual(val, NoFData()), getfield_pullback
    end
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake.lgetfield)}, t::CoDual{<:DiagOrTensorMap}, ::CoDual{Val{FieldName}}
    ) where {FieldName}
    val = getfield(primal(t), FieldName)
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 3))

    return if FieldName === 1 || FieldName === :data
        dval = tangent(t).data
        CoDual(val, dval), getfield_pullback
    else # cannot be invalid fieldname since already called `getfield`
        zero_fcodual(val), getfield_pullback
    end
end

@is_primitive MinimalCtx Tuple{typeof(getfield), <:DiagOrTensorMap, Any, Vararg{Symbol}}

Base.@constprop :aggressive function Mooncake.frule!!(
        ::Dual{typeof(getfield)}, t::Dual{<:DiagOrTensorMap}, name::Dual
    )
    val = getfield(primal(t), primal(name))

    return if primal(name) === 1 || primal(name) === :data
        dval = tangent(t).data
        Dual(val, dval)
    else # cannot be invalid fieldname since already called `getfield`
        Dual(val, NoFData())
    end
end

Base.@constprop :aggressive function Mooncake.rrule!!(
        ::CoDual{typeof(getfield)}, t::CoDual{<:DiagOrTensorMap}, name::CoDual
    )
    val = getfield(primal(t), primal(name))
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 3))

    return if primal(name) === 1 || primal(name) === :data
        dval = tangent(t).data
        CoDual(val, dval), getfield_pullback
    else # cannot be invalid fieldname since already called `getfield`
        zero_fcodual(val), getfield_pullback
    end
end

Base.@constprop :aggressive function Mooncake.frule!!(
        ::Dual{typeof(getfield)}, t::Dual{<:DiagOrTensorMap}, name::Dual, order::Dual
    )
    y = getfield(primal(t), primal(name), primal(order))

    return if primal(name) === 1 || primal(name) === :data
        dval = tangent(t).data
        Dual(val, dval)
    else # cannot be invalid fieldname since already called `getfield`
        Dual(val, NoFData())
    end
end

Base.@constprop :aggressive function Mooncake.rrule!!(
        ::CoDual{typeof(getfield)}, t::CoDual{<:DiagOrTensorMap}, name::CoDual, order::CoDual
    )
    val = getfield(primal(t), primal(name), primal(order))
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 4))

    return if primal(name) === 1 || primal(name) === :data
        dval = tangent(t).data
        CoDual(val, dval), getfield_pullback
    else # cannot be invalid fieldname since already called `getfield`
        zero_fcodual(val), getfield_pullback
    end
end


@is_primitive MinimalCtx Tuple{typeof(Mooncake.lgetfield), <:DiagOrTensorMap, Val, Val}

# TODO: double-check if this has to include quantum dimensinos for non-abelian?
function Mooncake.frule!!(
        ::Dual{typeof(Mooncake.lgetfield)}, t::Dual{<:DiagOrTensorMap}, ::Dual{Val{FieldName}}, ::Dual{Val{Order}}
    ) where {FieldName, Order}
    y = getfield(primal(t), FieldName, Order)

    return if FieldName === 1 || FieldName === :data
        dval = tangent(t).data
        Dual(val, dval)
    else # cannot be invalid fieldname since already called `getfield`
        Dual(val, NoFData())
    end
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake.lgetfield)}, t::CoDual{<:DiagOrTensorMap}, ::CoDual{Val{FieldName}}, ::CoDual{Val{Order}}
    ) where {FieldName, Order}
    val = getfield(primal(t), FieldName, Order)
    getfield_pullback = Mooncake.NoPullback(ntuple(Returns(NoRData()), 4))

    return if FieldName === 1 || FieldName === :data
        dval = tangent(t).data
        CoDual(val, dval), getfield_pullback
    else # cannot be invalid fieldname since already called `getfield`
        zero_fcodual(val), getfield_pullback
    end
end


Mooncake.@zero_derivative MinimalCtx Tuple{typeof(Mooncake._new_), Type{TensorMap{T, S, N₁, N₂, A}}, UndefInitializer, TensorMapSpace{S, N₁, N₂}} where {T, S, N₁, N₂, A}
@is_primitive MinimalCtx Tuple{typeof(Mooncake._new_), Type{TensorMap{T, S, N₁, N₂, A}}, A, TensorMapSpace{S, N₁, N₂}} where {T, S, N₁, N₂, A}

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
    return zero_fcodual(Mooncake._new_(TensorMap{T, S, N₁, N₂, A}, primal(data), primal(space))),
        Returns(ntuple(Returns(NoRData()), 4))
end


Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(Mooncake._new_), Type{DiagonalTensorMap{T, S, A}}, UndefInitializer, S} where {T, S, A}
@is_primitive Mooncake.MinimalCtx Tuple{typeof(Mooncake._new_), Type{DiagonalTensorMap{T, S, A}}, A, S} where {T, S, A}

function Mooncake.frule!!(
        ::Dual{typeof(Mooncake._new_)}, ::Dual{Type{DiagonalTensorMap{T, S, A}}}, data::Dual{A}, space::Dual{S}
    ) where {T, S, A}
    t = Mooncake._new_(DiagonalTensorMap{T, S, A}, primal(data), primal(space))
    dt = Mooncake._new_(DiagonalTensorMap{T, S, A}, tangent(data), primal(space))
    return Dual(t, dt)
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake._new_)}, ::CoDual{Type{DiagonalTensorMap{T, S, A}}}, data::CoDual{A}, space::CoDual{S}
    ) where {T, S, A}
    return zero_fcodual(Mooncake._new_(DiagonalTensorMap{T, S, A}, primal(data), primal(space))),
        Returns(ntuple(Returns(NoRData()), 4))
end
