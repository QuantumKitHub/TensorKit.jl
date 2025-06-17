# Factorizations rules
# --------------------
function ChainRulesCore.rrule(::typeof(TensorKit.tsvd!), t::AbstractTensorMap;
                              trunc::TruncationStrategy=TensorKit.notrunc(),
                              kwargs...)
    # TODO: I think we can use tsvd! here without issues because we don't actually require
    # the data of `t` anymore.
    USVᴴ = tsvd(t; trunc=TensorKit.notrunc(), kwargs...)

    if trunc != TensorKit.notrunc() && !isempty(blocksectors(t))
        USVᴴ′ = MatrixAlgebraKit.truncate!(svd_trunc!, USVᴴ, trunc)
    else
        USVᴴ′ = USVᴴ
    end

    function tsvd!_pullback(ΔUSVᴴ′)
        ΔUSVᴴ = unthunk.(ΔUSVᴴ′)
        Δt = zerovector(t)
        foreachblock(Δt) do c, (b,)
            USVᴴc = block.(USVᴴ, Ref(c))
            ΔUSVᴴc = block.(ΔUSVᴴ, Ref(c))
            svd_compact_pullback!(b, USVᴴc, ΔUSVᴴc)
            return nothing
        end
        return NoTangent(), Δt
    end
    tsvd!_pullback(::NTuple{3,ZeroTangent}) = NoTangent(), ZeroTangent()

    return USVᴴ′, tsvd!_pullback
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.svdvals!), t::AbstractTensorMap)
    U, S, V⁺ = tsvd(t)
    s = diag(S)
    project_t = ProjectTo(t)

    function svdvals_pullback(Δs′)
        Δs = unthunk(Δs′)
        ΔS = diagm(codomain(S), domain(S), Δs)
        return NoTangent(), project_t(U * ΔS * V⁺)
    end

    return s, svdvals_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.eig!), t::AbstractTensorMap; kwargs...)
    DV = eig(t; kwargs...)

    function eig!_pullback(ΔDV′)
        ΔDV = unthunk.(ΔDV′)
        Δt = zerovector(t)
        foreachblock(Δt) do c, (b,)
            DVc = block.(DV, Ref(c))
            ΔDVc = block.(ΔDV, Ref(c))
            eig_full_pullback!(b, DVc, ΔDVc)
            return nothing
        end
        return NoTangent(), Δt
    end
    eig!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()

    return DV, eig!_pullback
end

function ChainRulesCore.rrule(::typeof(TensorKit.eigh!), t::AbstractTensorMap; kwargs...)
    DV = eigh(t; kwargs...)

    function eigh!_pullback(ΔDV′)
        ΔDV = unthunk.(ΔDV′)
        Δt = zerovector(t)
        foreachblock(Δt) do c, (b,)
            DVc = block.(DV, Ref(c))
            ΔDVc = block.(ΔDV, Ref(c))
            eigh_full_pullback!(b, DVc, ΔDVc)
            return nothing
        end
        return NoTangent(), Δt
    end
    eigh!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()

    return DV, eigh!_pullback
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.eigvals!), t::AbstractTensorMap;
                              sortby=nothing, kwargs...)
    @assert sortby === nothing "only `sortby=nothing` is supported"
    (D, _), eig_pullback = rrule(TensorKit.eig!, t; kwargs...)
    d = diag(D)
    project_t = ProjectTo(t)
    function eigvals_pullback(Δd′)
        Δd = unthunk(Δd′)
        ΔD = diagm(codomain(D), domain(D), Δd)
        return NoTangent(), project_t(eig_pullback((ΔD, ZeroTangent()))[2])
    end

    return d, eigvals_pullback
end

function ChainRulesCore.rrule(::typeof(leftorth!), t::AbstractTensorMap; alg=QRpos())
    alg isa TensorKit.QR || alg isa TensorKit.QRpos ||
        error("only `alg=QR()` and `alg=QRpos()` are supported")
    QR = leftorth(t; alg)
    function leftorth!_pullback(ΔQR′)
        ΔQR = unthunk.(ΔQR′)
        Δt = zerovector(t)
        foreachblock(Δt) do c, (b,)
            QRc = block.(QR, Ref(c))
            ΔQRc = block.(ΔQR, Ref(c))
            qr_compact_pullback!(b, QRc, ΔQRc)
            return nothing
        end
        return NoTangent(), Δt
    end
    leftorth!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()

    return QR, leftorth!_pullback
end

function ChainRulesCore.rrule(::typeof(rightorth!), t::AbstractTensorMap; alg=LQpos())
    alg isa TensorKit.LQ || alg isa TensorKit.LQpos ||
        error("only `alg=LQ()` and `alg=LQpos()` are supported")
    LQ = rightorth(t; alg)
    function rightorth!_pullback(ΔLQ′)
        ΔLQ = unthunk(ΔLQ′)
        Δt = zerovector(t)
        foreachblock(Δt) do c, (b,)
            LQc = block.(LQ, Ref(c))
            ΔLQc = block.(ΔLQ, Ref(c))
            lq_compact_pullback!(b, LQc, ΔLQc)
            return nothing
        end
        return NoTangent(), Δt
    end
    rightorth!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()
    return LQ, rightorth!_pullback
end
