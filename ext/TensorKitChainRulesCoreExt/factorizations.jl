# Factorizations rules
# --------------------
function ChainRulesCore.rrule(::typeof(MatrixAlgebraKit.copy_input), f,
                              t::AbstractTensorMap)
    project = ProjectTo(t)
    copy_input_pullback(Δt) = (NoTangent(), NoTangent(), project(unthunk(Δt)))
    return MatrixAlgebraKit.copy_input(f, t), copy_input_pullback
end

@non_differentiable MatrixAlgebraKit.initialize_output(f, t::AbstractTensorMap, args...)
@non_differentiable MatrixAlgebraKit.check_input(f, t::AbstractTensorMap, args...)

for qr_f in (:qr_compact, :qr_full)
    qr_f! = Symbol(qr_f, '!')
    @eval function ChainRulesCore.rrule(::typeof($qr_f!), t::AbstractTensorMap, QR, alg)
        tc = MatrixAlgebraKit.copy_input($qr_f, t)
        QR = $(qr_f!)(tc, QR, alg)
        function qr_pullback(ΔQR′)
            ΔQR = unthunk.(ΔQR′)
            Δt = zerovector(t)
            MatrixAlgebraKit.qr_compact_pullback!(Δt, t, QR, ΔQR)
            return NoTangent(), Δt, ZeroTangent(), NoTangent()
        end
        function qr_pullback(::Tuple{ZeroTangent,ZeroTangent})
            return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
        end
        return QR, qr_pullback
    end
end
function ChainRulesCore.rrule(::typeof(qr_null!), t::AbstractTensorMap, N, alg)
    Q, R = qr_full(t, alg)
    for (c, b) in blocks(t)
        m, n = size(b)
        copy!(block(N, c), view(block(Q, c), 1:m, (n + 1):m))
    end

    function qr_null_pullback(ΔN′)
        ΔN = unthunk(ΔN′)
        Δt = zerovector(t)
        ΔQ = zerovector!(similar(Q, codomain(Q) ← fuse(codomain(Q))))
        foreachblock(ΔN) do c, (b,)
            n = size(b, 2)
            ΔQc = block(ΔQ, c)
            return copy!(@view(ΔQc[:, (end - n + 1):end]), b)
        end
        ΔR = ZeroTangent()
        MatrixAlgebraKit.qr_compact_pullback!(Δt, t, (Q, R), (ΔQ, ΔR))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    qr_null_pullback(::ZeroTangent) = NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()

    return N, qr_null_pullback
end

for lq_f in (:lq_compact, :lq_full)
    lq_f! = Symbol(lq_f, '!')
    @eval function ChainRulesCore.rrule(::typeof($lq_f!), t::AbstractTensorMap, LQ, alg)
        tc = MatrixAlgebraKit.copy_input($lq_f, t)
        LQ = $(lq_f!)(tc, LQ, alg)
        function lq_pullback(ΔLQ′)
            ΔLQ = unthunk.(ΔLQ′)
            Δt = zerovector(t)
            MatrixAlgebraKit.lq_compact_pullback!(Δt, t, LQ, ΔLQ)
            return NoTangent(), Δt, ZeroTangent(), NoTangent()
        end
        function lq_pullback(::Tuple{ZeroTangent,ZeroTangent})
            return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
        end
        return LQ, lq_pullback
    end
end
function ChainRulesCore.rrule(::typeof(lq_null!), t::AbstractTensorMap, Nᴴ, alg)
    L, Q = lq_full(t, alg)
    for (c, b) in blocks(t)
        m, n = size(b)
        copy!(block(Nᴴ, c), view(block(Q, c), (m + 1):n, 1:n))
    end

    function lq_null_pullback(ΔNᴴ′)
        ΔNᴴ = unthunk(ΔNᴴ′)
        Δt = zerovector(t)
        ΔQ = zerovector!(similar(Q, codomain(Q) ← fuse(codomain(Q))))
        foreachblock(ΔNᴴ) do c, (b,)
            m = size(b, 1)
            ΔQc = block(ΔQ, c)
            return copy!(@view(ΔQc[(end - m + 1):end, :]), b)
        end
        ΔL = ZeroTangent()
        MatrixAlgebraKit.lq_compact_pullback!(Δt, t, (L, Q), (ΔL, ΔQ))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    lq_null_pullback(::ZeroTangent) = NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()

    return Nᴴ, lq_null_pullback
end

for eig in (:eig, :eigh)
    eig_f = Symbol(eig, "_full")
    eig_f! = Symbol(eig_f, "!")
    eig_f_pb! = Symbol(eig, "_pullback!")
    eig_pb = Symbol(eig, "_pullback")
    @eval function ChainRulesCore.rrule(::typeof($eig_f!), t::AbstractTensorMap, DV, alg)
        tc = MatrixAlgebraKit.copy_input($eig_f, t)
        DV = $(eig_f!)(tc, DV, alg)
        function $eig_pb(ΔDV)
            Δt = zerovector(t)
            MatrixAlgebraKit.$eig_f_pb!(Δt, t, DV, unthunk.(ΔDV))
            return NoTangent(), Δt, ZeroTangent(), NoTangent()
        end
        function $eig_pb(::Tuple{ZeroTangent,ZeroTangent})
            return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
        end
        return DV, $eig_pb
    end
end

for svd_f in (:svd_compact, :svd_full)
    svd_f! = Symbol(svd_f, "!")
    @eval begin
        function ChainRulesCore.rrule(::typeof($svd_f!), t::AbstractTensorMap, USVᴴ, alg)
            tc = MatrixAlgebraKit.copy_input($svd_f, t)
            USVᴴ = $(svd_f!)(tc, USVᴴ, alg)
            function svd_pullback(ΔUSVᴴ)
                Δt = zerovector(t)
                MatrixAlgebraKit.svd_pullback!(Δt, t, USVᴴ, unthunk.(ΔUSVᴴ))
                return NoTangent(), Δt, ZeroTangent(), NoTangent()
            end
            function svd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
                return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
            end
            return USVᴴ, svd_pullback
        end
    end
end

function ChainRulesCore.rrule(::typeof(svd_trunc!), t::AbstractTensorMap, USVᴴ,
                              alg::TruncatedAlgorithm)
    tc = MatrixAlgebraKit.copy_input(svd_compact, t)
    USVᴴ = svd_compact!(tc, USVᴴ, alg.alg)
    USVᴴ_trunc, ind = TensorKit.Factorizations.truncate(svd_trunc!, USVᴴ, alg.trunc)
    svd_trunc_pullback = _make_svd_trunc_pullback(t, USVᴴ, ind)
    return USVᴴ_trunc, svd_trunc_pullback
end
function _make_svd_trunc_pullback(t::AbstractTensorMap, USVᴴ, ind)
    function svd_trunc_pullback(ΔUSVᴴ)
        Δt = zerovector(t)
        MatrixAlgebraKit.svd_pullback!(Δt, t, USVᴴ, unthunk.(ΔUSVᴴ), ind)
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    function svd_trunc_pullback(::NTuple{3,ZeroTangent})
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return svd_trunc_pullback
end

function ChainRulesCore.rrule(::typeof(left_polar!), t::AbstractTensorMap, WP, alg)
    tc = MatrixAlgebraKit.copy_input(left_polar, t)
    WP = left_polar!(tc, WP, alg)
    function left_polar_pullback(ΔWP)
        Δt = zerovector(t)
        MatrixAlgebraKit.left_polar_pullback!(Δt, t, WP, unthunk.(ΔWP))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    function left_polar_pullback(::Tuple{ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return WP, left_polar_pullback
end

function ChainRulesCore.rrule(::typeof(right_polar!), t::AbstractTensorMap, PWᴴ, alg)
    tc = MatrixAlgebraKit.copy_input(left_polar, t)
    PWᴴ = right_polar!(tc, PWᴴ, alg)
    function right_polar_pullback(ΔPWᴴ)
        Δt = zerovector(t)
        MatrixAlgebraKit.right_polar_pullback!(Δt, t, PWᴴ, unthunk.(ΔPWᴴ))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    function right_polar_pullback(::Tuple{ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return PWᴴ, right_polar_pullback
end

# function ChainRulesCore.rrule(::typeof(LinearAlgebra.svdvals!), t::AbstractTensorMap)
#     U, S, V⁺ = tsvd(t)
#     s = diag(S)
#     project_t = ProjectTo(t)

#     function svdvals_pullback(Δs′)
#         Δs = unthunk(Δs′)
#         ΔS = diagm(codomain(S), domain(S), Δs)
#         return NoTangent(), project_t(U * ΔS * V⁺)
#     end

#     return s, svdvals_pullback
# end

# function ChainRulesCore.rrule(::typeof(LinearAlgebra.eigvals!), t::AbstractTensorMap;
#                               sortby=nothing, kwargs...)
#     @assert sortby === nothing "only `sortby=nothing` is supported"
#     (D, _), eig_pullback = rrule(TensorKit.eig!, t; kwargs...)
#     d = diag(D)
#     project_t = ProjectTo(t)
#     function eigvals_pullback(Δd′)
#         Δd = unthunk(Δd′)
#         ΔD = diagm(codomain(D), domain(D), Δd)
#         return NoTangent(), project_t(eig_pullback((ΔD, ZeroTangent()))[2])
#     end

#     return d, eigvals_pullback
# end
