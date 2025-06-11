# convenience to set default
macro check_space(x, V)
    return esc(:($MatrixAlgebraKit.@check_size($x, $V, $space)))
end
macro check_scalar(x, y, op=:identity, eltype=:scalartype)
    return esc(:($MatrixAlgebraKit.@check_scalar($x, $y, $op, $eltype)))
end

# Generic
# -------
for f in (:eig_full, :eig_vals, :eig_trunc, :eigh_full, :eigh_vals, :eigh_trunc, :svd_full,
          :svd_compact, :svd_vals, :svd_trunc)
    @eval function MatrixAlgebraKit.copy_input(::typeof($f),
                                               t::AbstractTensorMap{<:BlasFloat})
        T = factorisation_scalartype($f, t)
        return copy_oftype(t, T)
    end
    f! = Symbol(f, :!)
    @eval function MatrixAlgebraKit.select_algorithm(::typeof($f!), t::AbstractTensorMap,
                                                     alg::Alg=nothing;
                                                     kwargs...) where {Alg}
        return MatrixAlgebraKit.select_algorithm($f!, typeof(t), alg; kwargs...)
    end
    @eval function MatrixAlgebraKit.select_algorithm(::typeof($f!), ::Type{T},
                                                     alg::Alg=nothing;
                                                     scheduler=default_blockscheduler(T),
                                                     kwargs...) where {T<:AbstractTensorMap,
                                                                       Alg}
        mat_alg = MatrixAlgebraKit.select_algorithm($f!, blocktype(T), alg; kwargs...)
        return BlockAlgorithm(mat_alg, scheduler)
    end
end

for f in (:qr, :lq, :svd, :eig, :eigh, :polar)
    default_f_algorithm = Symbol(:default_, f, :_algorithm)
    @eval function MatrixAlgebraKit.$default_f_algorithm(::Type{T};
                                                         scheduler=default_blockscheduler(T),
                                                         kwargs...) where {T<:AbstractTensorMap}
        return BlockAlgorithm(MatrixAlgebraKit.$default_f_algorithm(blocktype(T);
                                                                    kwargs...),
                              scheduler)
    end
end

# TODO: move to MatrixAlgebraKit?
macro check_eltype(x, y, f=:identity, g=:eltype)
    msg = "unexpected scalar type: "
    msg *= string(g) * "(" * string(x) * ") != "
    if f == :identity
        msg *= string(g) * "(" * string(y) * ")"
    else
        msg *= string(f) * "(" * string(y) * ")"
    end
    return esc(:($g($x) == $f($g($y)) || throw(ArgumentError($msg))))
end

function _select_truncation(f, ::AbstractTensorMap,
                            trunc::MatrixAlgebraKit.TruncationStrategy)
    return trunc
end
function _select_truncation(::typeof(left_null!), ::AbstractTensorMap, trunc::NamedTuple)
    return MatrixAlgebraKit.null_truncation_strategy(; trunc...)
end

function MatrixAlgebraKit.diagview(t::AbstractTensorMap)
    return SectorDict(c => MatrixAlgebraKit.diagview(b) for (c, b) in blocks(t))
end

# Singular value decomposition
# ----------------------------
const _T_USVᴴ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap,<:AbstractTensorMap}
const _T_USVᴴ_diag = Tuple{<:AbstractTensorMap,<:DiagonalTensorMap,<:AbstractTensorMap}

function MatrixAlgebraKit.check_input(::typeof(svd_full!), t::AbstractTensorMap,
                                      (U, S, Vᴴ)::_T_USVᴴ)
    # scalartype checks
    @check_scalar U t
    @check_scalar S t real
    @check_scalar Vᴴ t

    # space checks
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    @check_space(U, codomain(t) ← V_cod)
    @check_space(S, V_cod ← V_dom)
    @check_space(Vᴴ, V_dom ← domain(t))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(svd_compact!), t::AbstractTensorMap,
                                      (U, S, Vᴴ)::_T_USVᴴ_diag)
    # scalartype checks
    @check_eltype U t
    @check_eltype S t real
    @check_eltype Vᴴ t

    # space checks
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(U, codomain(t) ← V_cod)
    @check_space(S, V_cod ← V_dom)
    @check_space(Vᴴ, V_dom ← domain(t))

    return nothing
end

# TODO: svd_vals

function MatrixAlgebraKit.initialize_output(::typeof(svd_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(::typeof(svd_compact!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(::typeof(svd_trunc!), t::AbstractTensorMap,
                                            alg::MatrixAlgebraKit.AbstractAlgorithm)
    return MatrixAlgebraKit.initialize_output(svd_compact!, t, alg)
end

# TODO: svd_vals

function MatrixAlgebraKit.svd_full!(t::AbstractTensorMap, (U, S, Vᴴ),
                                    alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(svd_full!, t, (U, S, Vᴴ))

    foreachblock(t, U, S, Vᴴ; alg.scheduler) do _, (b, u, s, vᴴ)
        if isempty(b) # TODO: remove once MatrixAlgebraKit supports empty matrices
            MatrixAlgebraKit.one!(length(u) > 0 ? u : vᴴ)
            zerovector!(s)
        else
            u′, s′, vᴴ′ = MatrixAlgebraKit.svd_full!(b, (u, s, vᴴ), alg.alg)
            # deal with the case where the output is not the same as the input
            u === u′ || copyto!(u, u′)
            s === s′ || copyto!(s, s′)
            vᴴ === vᴴ′ || copyto!(vᴴ, vᴴ′)
        end
        return nothing
    end

    return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_compact!(t::AbstractTensorMap, (U, S, Vᴴ),
                                       alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(svd_compact!, t, (U, S, Vᴴ))

    foreachblock(t, U, S, Vᴴ; alg.scheduler) do _, (b, u, s, vᴴ)
        u′, s′, vᴴ′ = svd_compact!(b, (u, s, vᴴ), alg.alg)
        # deal with the case where the output is not the same as the input
        u === u′ || copyto!(u, u′)
        s === s′ || copyto!(s, s′)
        vᴴ === vᴴ′ || copyto!(vᴴ, vᴴ′)
        return nothing
    end

    return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_trunc!(t::AbstractTensorMap, USVᴴ,
                                     alg::MatrixAlgebraKit.TruncatedAlgorithm)
    USVᴴ′ = svd_compact!(t, USVᴴ, alg.alg)
    return MatrixAlgebraKit.truncate!(svd_trunc!, USVᴴ′, alg.trunc)
end

# Eigenvalue decomposition
# ------------------------
const _T_DV = Tuple{<:DiagonalTensorMap,<:AbstractTensorMap}
function MatrixAlgebraKit.check_input(::typeof(eigh_full!), t::AbstractTensorMap,
                                      (D, V)::_T_DV)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    # scalartype checks
    @check_scalar D t real
    @check_scalar V t

    # space checks
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    @check_space(V, codomain(t) ← V_D)

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(eig_full!), t::AbstractTensorMap,
                                      (D, V)::_T_DV)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    # scalartype checks
    @check_scalar D t complex
    @check_scalar V t complex

    # space checks
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    @check_space(V, codomain(t) ← V_D)

    return nothing
end

function MatrixAlgebraKit.initialize_output(::typeof(eigh_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function MatrixAlgebraKit.initialize_output(::typeof(eig_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

for f in (:eigh_full!, :eig_full!)
    @eval function MatrixAlgebraKit.$f(t::AbstractTensorMap, (D, V),
                                       alg::BlockAlgorithm)
        MatrixAlgebraKit.check_input($f, t, (D, V))

        foreachblock(t, D, V; alg.scheduler) do _, (b, d, v)
            d′, v′ = $f(b, (d, v), alg.alg)
            # deal with the case where the output is not the same as the input
            d === d′ || copyto!(d, d′)
            v === v′ || copyto!(v, v′)
            return nothing
        end

        return D, V
    end
end

# QR decomposition
# ----------------
function MatrixAlgebraKit.check_input(::typeof(qr_full!), t::AbstractTensorMap,
                                      (Q,
                                       R)::Tuple{<:AbstractTensorMap,<:AbstractTensorMap})
    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = fuse(codomain(t))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(qr_compact!), t::AbstractTensorMap, (Q, R))
    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(qr_null!), t::AbstractTensorMap,
                                      N::AbstractTensorMap)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    @check_space(N, codomain(t) ← V_N)

    return nothing
end

function MatrixAlgebraKit.initialize_output(::typeof(qr_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_Q = fuse(codomain(t))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

function MatrixAlgebraKit.initialize_output(::typeof(qr_compact!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

function MatrixAlgebraKit.initialize_output(::typeof(qr_null!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = similar(t, codomain(t) ← V_N)
    return N
end

function MatrixAlgebraKit.qr_full!(t::AbstractTensorMap, (Q, R),
                                   alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(qr_full!, t, (Q, R))

    foreachblock(t, Q, R; alg.scheduler) do _, (b, q, r)
        q′, r′ = qr_full!(b, (q, r), alg.alg)
        # deal with the case where the output is not the same as the input
        q === q′ || copyto!(q, q′)
        r === r′ || copyto!(r, r′)
        return nothing
    end

    return Q, R
end

function MatrixAlgebraKit.qr_compact!(t::AbstractTensorMap, (Q, R),
                                      alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(qr_compact!, t, (Q, R))

    foreachblock(t, Q, R; alg.scheduler) do _, (b, q, r)
        q′, r′ = qr_compact!(b, (q, r), alg.alg)
        # deal with the case where the output is not the same as the input
        q === q′ || copyto!(q, q′)
        r === r′ || copyto!(r, r′)
        return nothing
    end

    return Q, R
end

function MatrixAlgebraKit.qr_null!(t::AbstractTensorMap, N, alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(qr_null!, t, N)

    foreachblock(t, N; alg.scheduler) do _, (b, n)
        n′ = qr_null!(b, n, alg.alg)
        # deal with the case where the output is not the same as the input
        n === n′ || copyto!(n, n′)
        return nothing
    end

    return N
end


# LQ decomposition
# ----------------
function MatrixAlgebraKit.check_input(::typeof(lq_full!), t::AbstractTensorMap, (L, Q))
    # scalartype checks
    @check_eltype L t
    @check_eltype Q t

    # space checks
    V_Q = fuse(domain(t))
    space(L) == (codomain(t) ← V_Q) ||
        throw(SpaceMismatch("`lq_full!(t, (L, Q))` requires `space(L) == (codomain(t) ← fuse(domain(t)))`"))
    space(Q) == (V_Q ← domain(t)) ||
        throw(SpaceMismatch("`lq_full!(t, (L, Q))` requires `space(Q) == (fuse(domain(t)) ← domain(t))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(lq_compact!), t::AbstractTensorMap, (L, Q))
    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(lq_null!), t::AbstractTensorMap, N)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    @check_space(N, V_N ← domain(t))

    return nothing
end

function MatrixAlgebraKit.initialize_output(::typeof(lq_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_Q = fuse(domain(t))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end

function MatrixAlgebraKit.initialize_output(::typeof(lq_compact!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end

function MatrixAlgebraKit.initialize_output(::typeof(lq_null!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = similar(t, V_N ← domain(t))
    return N
end

function MatrixAlgebraKit.lq_full!(t::AbstractTensorMap, (L, Q),
                                   alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(lq_full!, t, (L, Q))

    foreachblock(t, L, Q; alg.scheduler) do _, (b, l, q)
        l′, q′ = lq_full!(b, (l, q), alg.alg)
        # deal with the case where the output is not the same as the input
        l === l′ || copyto!(l, l′)
        q === q′ || copyto!(q, q′)
        return nothing
    end

    return L, Q
end

function MatrixAlgebraKit.lq_compact!(t::AbstractTensorMap, (L, Q),
                                      alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(lq_compact!, t, (L, Q))

    foreachblock(t, L, Q; alg.scheduler) do _, (b, l, q)
        l′, q′ = lq_compact!(b, (l, q), alg.alg)
        # deal with the case where the output is not the same as the input
        l === l′ || copyto!(l, l′)
        q === q′ || copyto!(q, q′)
        return nothing
    end

    return L, Q
end

function MatrixAlgebraKit.lq_null!(t::AbstractTensorMap, N, alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(lq_null!, t, N)

    foreachblock(t, N; alg.scheduler) do _, (b, n)
        n′ = lq_null!(b, n, alg.alg)
        # deal with the case where the output is not the same as the input
        n === n′ || copyto!(n, n′)
        return nothing
    end

    return N
end

# Polar decomposition
# -------------------
using MatrixAlgebraKit: PolarViaSVD

function MatrixAlgebraKit.check_input(::typeof(left_polar!), t, (W, P))
    codomain(t) ≿ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `codomain(t) ≿ domain(t)`"))

    # scalartype checks
    @check_scalar W t
    @check_scalar P t

    # space checks
    @check_space(W, space(t))
    @check_space(P, domain(t) ← domain(t))

    return nothing
end

# TODO: do we really not want to fuse the spaces?
function MatrixAlgebraKit.initialize_output(::typeof(left_polar!), t::AbstractTensorMap,
                                            ::BlockAlgorithm)
    W = similar(t, space(t))
    P = similar(t, domain(t) ← domain(t))
    return W, P
end

function MatrixAlgebraKit.left_polar!(t::AbstractTensorMap, WP, alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(left_polar!, t, WP)

    foreachblock(t, WP...; alg.scheduler) do _, (b, w, p)
        w′, p′ = left_polar!(b, (w, p), alg.alg)
        # deal with the case where the output is not the same as the input
        w === w′ || copyto!(w, w′)
        p === p′ || copyto!(p, p′)
        return nothing
    end

    return WP
end

# Trick to relax the checks of "square" if coming from left_orth
function MatrixAlgebraKit.left_orth_polar!(t::AbstractTensorMap, VC, alg)
    alg′ = MatrixAlgebraKit.select_algorithm(left_polar!, t, alg)
    return MatrixAlgebraKit.left_orth_polar!(t, VC, alg′)
end
function MatrixAlgebraKit.left_orth_polar!(t::AbstractTensorMap, WP, alg::BlockAlgorithm)
    foreachblock(t, WP...; alg.scheduler) do _, (b, w, p)
        w′, p′ = left_polar!(b, (w, p), alg.alg)
        # deal with the case where the output is not the same as the input
        w === w′ || copyto!(w, w′)
        p === p′ || copyto!(p, p′)
        return nothing
    end
    return WP
end

# Orthogonalization
# -----------------
function MatrixAlgebraKit.check_input(::typeof(left_orth!), t::AbstractTensorMap, (V, C))
    # scalartype checks
    @check_scalar V t
    isnothing(C) || @check_scalar C t

    # space checks
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(V, codomain(t) ← V_C)
    isnothing(C) || @check_space(CV_C ← domain(t))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(right_orth!), t::AbstractTensorMap, (C, Vᴴ))
    # scalartype checks
    isnothing(C) || @check_scalar C t
    @check_scalar Vᴴ t

    # space checks
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    isnothing(C) || @check_space(C, codomain(t) ← V_C)
    @check_space(Vᴴ, V_dom ← domain(t))

    return nothing
end

function MatrixAlgebraKit.initialize_output(::typeof(left_orth!), t::AbstractTensorMap)
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    V = similar(t, codomain(t) ← V_C)
    C = similar(t, V_C ← domain(t))
    return V, C
end

function MatrixAlgebraKit.initialize_output(::typeof(right_orth!), t::AbstractTensorMap)
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    C = similar(t, codomain(t) ← V_C)
    Vᴴ = similar(t, V_C ← domain(t))
    return C, Vᴴ
end

# Nullspace
# ---------
function MatrixAlgebraKit.check_input(::typeof(left_null!), t::AbstractTensorMap, N)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    @check_space(N, codomain(t) ← V_N)

    return nothing
end

function MatrixAlgebraKit.initialize_output(::typeof(left_null!), t::AbstractTensorMap)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = similar(t, codomain(t) ← V_N)
    return N
end

# TODO: the following functions shouldn't be necessary if the AbstractArray restrictions are
# removed
function MatrixAlgebraKit.left_null(t::AbstractTensorMap; kwargs...)
    return left_null!(MatrixAlgebraKit.copy_input(left_null, t); kwargs...)
end
function MatrixAlgebraKit.left_null!(t::AbstractTensorMap; kwargs...)
    N = MatrixAlgebraKit.initialize_output(left_null!, t)
    return left_null!(t, N; kwargs...)
end

function MatrixAlgebraKit.left_null!(t::AbstractTensorMap, N;
                                     trunc=nothing,
                                     kind=isnothing(trunc) ? :qr : :svd,
                                     alg_qr=(; positive=true),
                                     alg_svd=(;))
    MatrixAlgebraKit.check_input(left_null!, t, N)

    if !isnothing(trunc) && kind != :svd
        throw(ArgumentError("truncation not supported for left_null with kind=$kind"))
    end

    if kind == :qr
        @info "qr"
        alg_qr′ = MatrixAlgebraKit._select_algorithm(qr_null!, t, alg_qr)
        return qr_null!(t, N, alg_qr′)
    elseif kind == :svd && isnothing(trunc)
        @info "svd"
        alg_svd′ = MatrixAlgebraKit._select_algorithm(svd_full!, t, alg_svd)
        # TODO: refactor into separate function
        U, _, _ = svd_full!(t, alg_svd′)
        for (c, b) in blocks(N)
            bU = block(U, c)
            m, n = size(bU)
            copy!(b, @view(bU[1:m, (n + 1):m]))
        end
        return N
    elseif kind == :svd
        @info "svd2"
        alg_svd′ = MatrixAlgebraKit._select_algorithm(svd_full!, t, alg_svd)
        @show U, S, _ = svd_full!(t, alg_svd′)
        @show trunc′ = _select_truncation(left_null!, t, @show trunc)
        return MatrixAlgebraKit.truncate!(left_null!, (U, S), trunc′)
    else
        throw(ArgumentError("`left_null!` received unknown value `kind = $kind`"))
    end
end

# Truncation
# ----------
# TODO: technically we could do this truncation in-place, but this might not be worth it
function MatrixAlgebraKit.truncate!(::typeof(svd_trunc!), (U, S, Vᴴ),
                                    trunc::MatrixAlgebraKit.TruncationKeepAbove)
    atol = max(trunc.atol, norm(S) * trunc.rtol)
    V_truncated = spacetype(S)(c => findlast(>=(atol), b.diag) for (c, b) in blocks(S))

    Ũ = similar(U, codomain(U) ← V_truncated)
    for (c, b) in blocks(Ũ)
        copy!(b, @view(block(U, c)[:, 1:size(b, 2)]))
    end

    S̃ = DiagonalTensorMap{scalartype(S)}(undef, V_truncated)
    for (c, b) in blocks(S̃)
        copy!(b.diag, @view(block(S, c).diag[1:size(b, 1)]))
    end

    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    for (c, b) in blocks(Ṽᴴ)
        copy!(b, @view(block(Vᴴ, c)[1:size(b, 1), :]))
    end

    return Ũ, S̃, Ṽᴴ
end

function MatrixAlgebraKit.truncate!(::typeof(left_null!),
                                    (U, S)::Tuple{<:AbstractTensorMap,
                                                  <:AbstractTensorMap},
                                    strategy::MatrixAlgebraKit.TruncationStrategy)
    extended_S = SectorDict(c => vcat(MatrixAlgebraKit.diagview(b),
                                      zeros(eltype(b), max(0, size(b, 2) - size(b, 1))))
                            for (c, b) in blocks(S))
    ind = MatrixAlgebraKit.findtruncated(extended_S, strategy)
    V_truncated = spacetype(S)(c => length(axes(b, 1)[ind[c]]) for (c, b) in blocks(S))
    Ũ = similar(U, codomain(U) ← V_truncated)
    for (c, b) in blocks(Ũ)
        copy!(b, @view(block(U, c)[:, ind[c]]))
    end
    return Ũ
end

const BlockWiseTruncations = Union{MatrixAlgebraKit.TruncationKeepAbove,
                                   MatrixAlgebraKit.TruncationKeepBelow,
                                   MatrixAlgebraKit.TruncationKeepFiltered}

# TODO: relative tolerances should be global
function MatrixAlgebraKit.findtruncated(values::SectorDict, strategy::BlockWiseTruncations)
    return SectorDict(c => MatrixAlgebraKit.findtruncated(v, strategy) for (c, v) in values)
end
function MatrixAlgebraKit.findtruncated(vals::SectorDict,
                                        strategy::MatrixAlgebraKit.TruncationKeepSorted)
    allpairs = mapreduce(vcat, vals) do (c, v)
        return map(Base.Fix1(=>, c), axes(v, 1))
    end
    by((c, i)) = strategy.sortby(vals[c][i])
    sort!(allpairs; by, strategy.rev)

    howmany = zero(Base.promote_op(dim, valtype(values)))
    i = 1
    while i ≤ length(allpairs)
        howmany += dim(first(allpairs[i]))

        howmany == strategy.howmany && break

        if howmany > strategy.howmany
            i -= 1
            break
        end

        i += 1
    end

    ind = SectorDict(c => allpairs[findall(==(c) ∘ first, view(allpairs, 1:i))]
                     for c in keys(vals))
    filter!(!isempty ∘ last, ind) # TODO: this might not be necessary

    return ind
end
