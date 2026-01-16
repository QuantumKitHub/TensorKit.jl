module TensorKitMooncakeExt

using Mooncake
using Mooncake: @zero_derivative, DefaultCtx, ReverseMode, NoRData, CoDual, arrayify, primal
using TensorKit
using TensorOperations: TensorOperations, tensorcontract!, IndexTuple, Index2Tuple, linearize
import TensorOperations as TO
using VectorInterface: One, Zero
using TupleTools

# Ignore derivatives
# ------------------
@zero_derivative DefaultCtx Tuple{typeof(TensorKit.fusionblockstructure), Any}

_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{<:Number}) = true
_needs_tangent(::Type{<:Integer}) = false
_needs_tangent(::Type{<:Union{One, Zero}}) = false


function Mooncake.arrayify(A_dA::CoDual{<:TensorMap})
    A = Mooncake.primal(A_dA)
    dA_fw = Mooncake.tangent(A_dA)
    data = dA_fw.data.data
    dA = typeof(A)(data, A.space)
    return A, dA
end
trivtuple(N) = ntuple(identity, N)
Base.@constprop :aggressive function _repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ ||
        throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    return TupleTools.getindices(p, trivtuple(N₁)),
        TupleTools.getindices(p, trivtuple(length(p) - N₁) .+ N₁)
end
Base.@constprop :aggressive function _repartition(p::Index2Tuple, N₁::Int)
    return _repartition(linearize(p), N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, ::Index2Tuple{N₁}) where {N₁}
    return _repartition(p, N₁)
end
function _repartition(p::Union{IndexTuple, Index2Tuple}, t::AbstractTensorMap)
    return _repartition(p, TensorKit.numout(t))
end

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(tensorcontract!), AbstractTensorMap, AbstractTensorMap, Index2Tuple, Bool, AbstractTensorMap, Index2Tuple, Bool, Index2Tuple, Number, Number, Vararg{Any}}
function Mooncake.rrule!!(
        ::CoDual{typeof(tensorcontract!)},
        C_dC::CoDual{<:AbstractTensorMap{TC}},
        A_dA::CoDual{<:AbstractTensorMap{TA}}, pA_dpA::CoDual{<:Index2Tuple}, conjA_dconjA::CoDual{Bool},
        B_dB::CoDual{<:AbstractTensorMap{TB}}, pB_dpB::CoDual{<:Index2Tuple}, conjB_dconjB::CoDual{Bool},
        pAB_dpAB::CoDual{<:Index2Tuple},
        α_dα::CoDual{Tα}, β_dβ::CoDual{Tβ},
        ba_dba::CoDual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TB <: Number, TC <: Number}
    C, ΔC = arrayify(C_dC)
    A, ΔA = arrayify(A_dA)
    B, ΔB = arrayify(B_dB)
    pA = primal(pA_dpA)
    pB = primal(pB_dpB)
    pAB = primal(pAB_dpAB)
    conjA = primal(conjA_dconjA)
    conjB = primal(conjB_dconjB)
    α = primal(α_dα)
    β = primal(β_dβ)
    ba = primal.(ba_dba)
    C_cache = copy(C)
    TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

    function tensorcontract_pullback(::NoRData)
        copy!(C, C_cache)
        if Tα == Zero && Tβ == Zero
            scale!(ΔC, zero(TC))
            return ntuple(i -> NoRData(), 11 + length(ba))
        end
        ipAB = invperm(linearize(pAB))
        pΔC = _repartition(ipAB, TO.numout(pA))

        # dC
        if β === Zero()
            scale!(ΔC, β)
        else
            scale!(ΔC, conj(β))
        end

        # dA
        ipA = _repartition(invperm(linearize(pA)), A)
        conjΔC = conjA
        conjB′ = conjA ? conjB : !conjB
        # TODO: allocator
        tB = twist(
            B,
            TupleTools.vcat(
                filter(x -> !isdual(space(B, x)), pB[1]),
                filter(x -> isdual(space(B, x)), pB[2])
            ); copy = false
        )
        tensorcontract!(
            ΔA,
            ΔC, pΔC, conjΔC,
            tB, reverse(pB), conjB′,
            ipA,
            conjA ? α : conj(α), Zero(), ba...
        )

        # dB
        ipB = _repartition(invperm(linearize(pB)), B)
        conjΔC = conjB
        conjA′ = conjB ? conjA : !conjA
        # TODO: allocator
        tA = twist(
            A,
            TupleTools.vcat(
                filter(x -> isdual(space(A, x)), pA[1]),
                filter(x -> !isdual(space(A, x)), pA[2])
            ); copy = false
        )
        tensorcontract!(
            ΔB,
            tA, reverse(pA), conjA′,
            ΔC, pΔC, conjΔC,
            ipB,
            conjB ? α : conj(α), Zero(), ba...
        )

        dα = if _needs_tangent(Tα)
            AB = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
            Mooncake._rdata(inner(AB, ΔC))
        else
            NoRData()
        end
        dβ = if _needs_tangent(Tβ)
            # TODO: consider using `inner`
            Mooncake._rdata(inner(C, ΔC))
        else
            NoRData()
        end

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, tensorcontract_pullback
end


Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(norm), AbstractTensorMap, Real}
function Mooncake.rrule!!(::CoDual{typeof(norm)}, tΔt::CoDual{<:AbstractTensorMap}, pdp::CoDual{<:Real})
    t, Δt = arrayify(tΔt)
    p = primal(pdp)
    p == 2 || error("currently only implemented for p = 2")
    n = norm(t, p)
    function norm_pullback(Δn)
        x = (Δn' + Δn) / 2 / hypot(n, eps(one(n)))
        add!(Δt, t, x)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(n, Mooncake.NoFData()), norm_pullback
end

end
