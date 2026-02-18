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
Mooncake.@foldable function Mooncake.tangent_type(::Type{TensorMap{T, S, N₁, N₂, A}}) where {T, S, N₁, N₂, A}
    Mooncake.tangent_type(T) isa NoTangent && return NoTangent
    TA = Mooncake.tangent_type(A)
    Mooncake.rdata_type(TA) === NoRData ||
        throw(ArgumentError("Mooncake support with storagetype `$A` is currently not implemented"))
    return TK.tensormaptype(S, N₁, N₂, TA)
end
Mooncake.@foldable Mooncake.tangent_type(::Type{T}, ::Type{NoRData}) where {T <: DiagonalTensorMap} = T
Mooncake.@foldable function Mooncake.tangent_type(::Type{DiagonalTensorMap{T, S, A}}) where {T, S, A}
    Mooncake.tangent_type(T) isa NoTangent && return NoTangent
    TA = Mooncake.tangent_type(A)
    Mooncake.rdata_type(TA) === NoRData ||
        throw(ArgumentError("Mooncake support with storagetype `$A` is currently not implemented"))
    return DiagonalTensorMap{scalartype(TA), S, TA}
end

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


# Custom rules for getters/setters
# --------------------------------
# both getfield and lgetfield are needed for cases where the field name is and isn't constant propagated
# no setfield because not a mutable struct

@is_primitive MinimalCtx Tuple{typeof(Mooncake.lgetfield), <:DiagOrTensorMap, Val}
@is_primitive MinimalCtx Tuple{typeof(Mooncake.lgetfield), <:DiagOrTensorMap, Val, Val}
@is_primitive MinimalCtx Tuple{typeof(getfield), <:DiagOrTensorMap, Symbol}
@is_primitive MinimalCtx Tuple{typeof(getfield), <:DiagOrTensorMap, Symbol, Symbol}
@is_primitive MinimalCtx Tuple{typeof(getfield), <:DiagOrTensorMap, Int, Symbol}

_field_symbol(f::Symbol) = f
_field_symbol(i::Int) = i == 1 ? :x : i == 2 ? :a : throw(ArgumentError("Invalid field index '$i' for type A."))
_field_symbol(::Type{Val{F}}) where {F} = _field_symbol(F)
_field_symbol(::Val{F}) where {F} = _field_symbol(F)

# frules
_frule_getfield_common(t_dt::Dual{<:DiagOrTensorMap}, field_sym::Symbol) =
    Dual(getfield(primal(t), field_sym), field_sym === :data ? tangent(t).data : NoFData())

Mooncake.frule!!(::Dual{typeof(Mooncake.lgetfield)}, t_dt::Dual{<:DiagOrTensorMap}, f_df::Dual) =
    _frule_getfield_common(t_dt, _field_symbol(primal(f_df)))
Mooncake.frule!!(::Dual{typeof(Mooncake.lgetfield)}, t_dt::Dual{<:DiagOrTensorMap}, f_df::Dual, o_do::Dual) =
    _frule_getfield_common(t_dt, _field_symbol(primal(f_df)))
Mooncake.frule!!(::Dual{typeof(getfield)}, t_dt::Dual{<:DiagOrTensorMap}, f_df::Dual) =
    _frule_getfield_common(t_dt, _field_symbol(primal(f_df)))
Mooncake.frule!!(::Dual{typeof(getfield)}, t_dt::Dual{<:DiagOrTensorMap}, f_df::Dual, o_do::Dual) =
    _frule_getfield_common(t_dt, _field_symbol(primal(f_df)))

# rrules
function _rrule_getfield_common(t_dt::CoDual{<:DiagOrTensorMap}, field_sym::Symbol, n_args::Int)
    t = primal(t)
    dt = tangent(t)

    value_primal = getfield(t, field_sym)
    value_dvalue = Mooncake.CoDual(
        value_primal,
        # fieldname is definitely valid here because `getfield` is called
        Mooncake.fdata(field_sym === :data ? dt.data : NoTangent())
    )

    function getfield_pullback(Δvalue_rdata)
        if field_sym === :data
            if !(Δvalue_rdata isa Mooncake.NoRData)
                data′ = Mooncake.increment_rdata!!(dt.data, Δvalue_rdata)
                data′ === dt.data || copy!(dt.data, data′)
            end
        else
            @assert Δvalue_rdata isa Mooncake.NoRData
        end
        return ntuple(Returns(Mooncake.NoRData()), n_args)
    end

    return value_dvalue, getfield_pullback
end

Mooncake.rrule!!(::CoDual{typeof(Mooncake.lgetfield)}, t_dt::CoDual{<:DiagOrTensorMap}, f_df::CoDual) =
    _rrule_getfield_common(t_dt, _field_symbol(primal(f_df)), 3)
Mooncake.rrule!!(::CoDual{typeof(Mooncake.lgetfield)}, t_dt::CoDual{<:DiagOrTensorMap}, f_df::CoDual, o_do::CoDual) =
    _rrule_getfield_common(t_dt, _field_symbol(primal(f_df)), 4)
Mooncake.rrule!!(::CoDual{typeof(getfield)}, t_dt::CoDual{<:DiagOrTensorMap}, f_df::CoDual) =
    _rrule_getfield_common(t_dt, _field_symbol(primal(f_df)), 3)
Mooncake.rrule!!(::CoDual{typeof(getfield)}, t_dt::CoDual{<:DiagOrTensorMap}, f_df::CoDual, o_do::CoDual) =
    _rrule_getfield_common(t_dt, _field_symbol(primal(f_df)), 4)


# Custom rules for constructors
# -----------------------------
# undef has zero derivative
@zero_derivative(
    MinimalCtx,
    Tuple{
        typeof(Mooncake._new_), Type{TensorMap{T, S, N₁, N₂, A}},
        UndefInitializer, TensorMapSpace{S, N₁, N₂},
    } where {T, S, N₁, N₂, A}
)
@zero_derivative(
    MinimalCtx,
    Tuple{
        typeof(Mooncake._new_), Type{DiagonalTensorMap{T, S, A}},
        UndefInitializer, S,
    } where {T, S, A}
)

@is_primitive(
    MinimalCtx, Tuple{
        typeof(Mooncake._new_), Type{TensorMap{T, S, N₁, N₂, A}},
        A, TensorMapSpace{S, N₁, N₂},
    } where {T, S, N₁, N₂, A}
)
@is_primitive(
    MinimalCtx, Tuple{
        typeof(Mooncake._new_), Type{DiagonalTensorMap{T, S, A}},
        A, S,
    } where {T, S, A}
)

function Mooncake.frule!!(
        ::Dual{typeof(Mooncake._new_)}, ::Dual{Type{TensorMap{T, S, N₁, N₂, A}}}, data::Dual{A}, space::Dual{TensorMapSpace{S, N₁, N₂}}
    ) where {T, S, N₁, N₂, A}
    t = TensorMap(primal(data), primal(space))
    dt = TensorMap(tangent(data), primal(space))
    return Dual(t, dt)
end
function Mooncake.frule!!(
        ::Dual{typeof(Mooncake._new_)}, ::Dual{Type{DiagonalTensorMap{T, S, A}}}, data::Dual{A}, space::Dual{S}
    ) where {T, S, A}
    t = DiagonalTensorMap(primal(data), primal(space))
    dt = DiagonalTensorMap(tangent(data), primal(space))
    return Dual(t, dt)
end

# rrules are trivial here because the magic is already happening in the construction of the
# `t_dt`, which already contains the `dt` correctly.
function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake._new_)}, ::CoDual{Type{TensorMap{T, S, N₁, N₂, A}}},
        data_ddata::CoDual{A}, space::CoDual{TensorMapSpace{S, N₁, N₂}}
    ) where {T, S, N₁, N₂, A}
    data = primal(data_ddata)
    ddata = tangent(data_ddata)
    t = TensorMap(data, primal(space))
    dt = TensorMap(ddata, primal(space))
    t_dt = CoDual(t, Mooncake.fdata(dt))
    TensorMap_pullback(Δt_rdata) = ntuple(Returns(NoRData()), 4)
    return t_dt, TensorMap_pullback
end
function Mooncake.rrule!!(
        ::CoDual{typeof(Mooncake._new_)}, ::CoDual{Type{DiagonalTensorMap{T, S, A}}},
        data_ddata::CoDual{A}, space::CoDual{S}
    ) where {T, S, A}
    data = primal(data_ddata)
    ddata = tangent(data_ddata)
    t = DiagonalTensorMap(data, primal(space))
    dt = DiagonalTensorMap(ddata, primal(space))
    t_dt = CoDual(t, Mooncake.fdata(dt))
    DiagonalTensorMap_pullback(Δt_rdata) = ntuple(Returns(NoRData()), 4)
    return t_dt, DiagonalTensorMap_pullback
end
