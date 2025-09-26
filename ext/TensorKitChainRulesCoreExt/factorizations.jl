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
            MatrixAlgebraKit.qr_compact_pullback!(Δt, QR, ΔQR)
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
        MatrixAlgebraKit.qr_compact_pullback!(Δt, (Q, R), (ΔQ, ΔR))
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
            MatrixAlgebraKit.lq_compact_pullback!(Δt, LQ, ΔLQ)
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
        MatrixAlgebraKit.lq_compact_pullback!(Δt, (L, Q), (ΔL, ΔQ))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    lq_null_pullback(::ZeroTangent) = NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()

    return Nᴴ, lq_null_pullback
end

for eig in (:eig, :eigh)
    eig_f = Symbol(eig, "_full")
    eig_f! = Symbol(eig_f, "!")
    eig_f_pb! = Symbol(eig, "_full_pullback!")
    eig_pb = Symbol(eig, "_pullback")
    @eval function ChainRulesCore.rrule(::typeof($eig_f!), t::AbstractTensorMap, DV, alg)
        tc = copy_input($eig_f, t)
        DV = $(eig_f!)(tc, DV, alg)
        function $eig_pb(ΔDV)
            Δt = zerovector(t)
            MatrixAlgebraKit.$eig_f_pb!(Δt, DV, unthunk.(ΔDV))
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
            tc = copy_input($svd_f, t)
            USVᴴ = $(svd_f!)(tc, USVᴴ, alg)
            function svd_pullback(ΔUSVᴴ)
                Δt = zerovector(t)
                MatrixAlgebraKit.svd_compact_pullback!(Δt, USVᴴ, unthunk.(ΔUSVᴴ))
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
    function svd_trunc_pullback(ΔUSVᴴ)
        Δt = zerovector(t)
        MatrixAlgebraKit.svd_compact_pullback!(Δt, USVᴴ, unthunk.(ΔUSVᴴ))
        return NoTangent(), ΔA, ZeroTangent(), NoTangent()
    end
    function svd_trunc_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return MatrixAlgebraKit.truncate!(svd_trunc!, USVᴴ, alg.trunc), svd_trunc_pullback
end

function ChainRulesCore.rrule(::typeof(left_polar!), t::AbstractTensorMap, WP, alg)
    tc = copy_input(left_polar, t)
    WP = left_polar!(tc, WP, alg)
    function left_polar_pullback(ΔWP)
        Δt = zerovector(t)
        MatrixAlgebraKit.left_polar_pullback!(Δt, WP, unthunk.(ΔWP))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    function left_polar_pullback(::Tuple{ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return WP, left_polar_pullback
end

function ChainRulesCore.rrule(::typeof(right_polar!), t::AbstractTensorMap, PWᴴ, alg)
    tc = copy_input(left_polar, t)
    PWᴴ = right_polar!(Ac, PWᴴ, alg)
    function right_polar_pullback(ΔPWᴴ)
        Δt = zerovector(t)
        MatrixAlgebraKit.right_polar_pullback!(Δt, PWᴴ, unthunk.(ΔPWᴴ))
        return NoTangent(), Δt, ZeroTangent(), NoTangent()
    end
    function right_polar_pullback(::Tuple{ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
    end
    return PWᴴ, right_polar_pullback
end

# for f in (:tsvd, :eig, :eigh)
#     f! = Symbol(f, :!)
#     f_trunc! = f == :tsvd ? :svd_trunc! : Symbol(f, :_trunc!)
#     f_pullback = Symbol(f, :_pullback)
#     f_pullback! = f == :tsvd ? :svd_compact_pullback! : Symbol(f, :_full_pullback!)
#     @eval function ChainRulesCore.rrule(::typeof(TensorKit.$f!), t::AbstractTensorMap;
#                                         trunc::TruncationStrategy=TensorKit.notrunc(),
#                                         kwargs...)
#         # TODO: I think we can use f! here without issues because we don't actually require
#         # the data of `t` anymore.
#         F = $f(t; trunc=TensorKit.notrunc(), kwargs...)

#         if trunc != TensorKit.notrunc() && !isempty(blocksectors(t))
#             F′ = MatrixAlgebraKit.truncate!($f_trunc!, F, trunc)
#         else
#             F′ = F
#         end

#         function $f_pullback(ΔF′)
#             ΔF = unthunk.(ΔF′)
#             Δt = zerovector(t)
#             foreachblock(Δt) do c, (b,)
#                 Fc = block.(F, Ref(c))
#                 ΔFc = block.(ΔF, Ref(c))
#                 $f_pullback!(b, Fc, ΔFc)
#                 return nothing
#             end
#             return NoTangent(), Δt
#         end
#         $f_pullback(::Tuple{ZeroTangent,Vararg{ZeroTangent}}) = NoTangent(), ZeroTangent()

#         return F′, $f_pullback
#     end
# end

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

# function ChainRulesCore.rrule(::typeof(leftorth!), t::AbstractTensorMap; alg=QRpos())
#     alg isa MatrixAlgebraKit.LAPACK_HouseholderQR ||
#         error("only `alg=QR()` and `alg=QRpos()` are supported")
#     QR = leftorth(t; alg)
#     function leftorth!_pullback(ΔQR′)
#         ΔQR = unthunk.(ΔQR′)
#         Δt = zerovector(t)
#         foreachblock(Δt) do c, (b,)
#             QRc = block.(QR, Ref(c))
#             ΔQRc = block.(ΔQR, Ref(c))
#             qr_compact_pullback!(b, QRc, ΔQRc)
#             return nothing
#         end
#         return NoTangent(), Δt
#     end
#     leftorth!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()

#     return QR, leftorth!_pullback
# end

# function ChainRulesCore.rrule(::typeof(rightorth!), t::AbstractTensorMap; alg=LQpos())
#     alg isa MatrixAlgebraKit.LAPACK_HouseholderLQ ||
#         error("only `alg=LQ()` and `alg=LQpos()` are supported")
#     LQ = rightorth(t; alg)
#     function rightorth!_pullback(ΔLQ′)
#         ΔLQ = unthunk(ΔLQ′)
#         Δt = zerovector(t)
#         foreachblock(Δt) do c, (b,)
#             LQc = block.(LQ, Ref(c))
#             ΔLQc = block.(ΔLQ, Ref(c))
#             lq_compact_pullback!(b, LQc, ΔLQc)
#             return nothing
#         end
#         return NoTangent(), Δt
#     end
#     rightorth!_pullback(::NTuple{2,ZeroTangent}) = NoTangent(), ZeroTangent()
#     return LQ, rightorth!_pullback
# end
