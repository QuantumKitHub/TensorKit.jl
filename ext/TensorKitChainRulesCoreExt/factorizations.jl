# Factorizations rules
# --------------------
for f in (:tsvd, :eig, :eigh)
    f! = Symbol(f, :!)
    f_trunc! = f == :tsvd ? :svd_trunc! : Symbol(f, :_trunc!)
    f_pullback = Symbol(f, :_pullback)
    f_pullback! = f == :tsvd ? :svd_compact_pullback! : Symbol(f, :_full_pullback!)
    @eval function ChainRulesCore.rrule(::typeof(TensorKit.$f!), t::AbstractTensorMap;
                                        trunc::TruncationStrategy=TensorKit.notrunc(),
                                        kwargs...)
        # TODO: I think we can use f! here without issues because we don't actually require
        # the data of `t` anymore.
        F = $f(t; trunc=TensorKit.notrunc(), kwargs...)

        if trunc != TensorKit.notrunc() && !isempty(blocksectors(t))
            F′ = MatrixAlgebraKit.truncate!($f_trunc!, F, trunc)
        else
            F′ = F
        end

        function $f_pullback(ΔF′)
            ΔF = unthunk.(ΔF′)
            Δt = zerovector(t)
            foreachblock(Δt) do c, (b,)
                Fc = block.(F, Ref(c))
                ΔFc = block.(ΔF, Ref(c))
                $f_pullback!(b, Fc, ΔFc)
                return nothing
            end
            return NoTangent(), Δt
        end
        $f_pullback(::Tuple{ZeroTangent,Vararg{ZeroTangent}}) = NoTangent(), ZeroTangent()

        return F′, $f_pullback
    end
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
    alg isa MatrixAlgebraKit.LAPACK_HouseholderQR ||
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
    alg isa MatrixAlgebraKit.LAPACK_HouseholderLQ ||
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
