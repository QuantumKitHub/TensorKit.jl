# Generic
# -------
for f in (:eig_full, :eig_vals, :eig_trunc, :eigh_full, :eigh_vals, :eigh_trunc, :svd_full,
          :svd_compact, :svd_vals, :svd_trunc)
    @eval function MatrixAlgebraKit.copy_input(::typeof($f),
                                               t::AbstractTensorMap{<:BlasFloat})
        T = factorisation_scalartype($f, t)
        return copy_oftype(t, T)
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

# function factorisation_scalartype(::typeof(MAK.eig_full!), t::AbstractTensorMap)
#     T = scalartype(t)
#     return promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))
# end

# Singular value decomposition
# ----------------------------
const _T_USVᴴ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap,<:AbstractTensorMap}
const _T_USVᴴ_diag = Tuple{<:AbstractTensorMap,<:DiagonalTensorMap,<:AbstractTensorMap}

function MatrixAlgebraKit.check_input(::typeof(svd_full!), t::AbstractTensorMap,
                                      (U, S, Vᴴ)::_T_USVᴴ)
    # scalartype checks
    @check_eltype U t
    @check_eltype S t real
    @check_eltype Vᴴ t

    # space checks
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    space(U) == (codomain(t) ← V_cod) ||
        throw(SpaceMismatch("`svd_full!(t, (U, S, Vᴴ))` requires `space(U) == (codomain(t) ← fuse(domain(t)))`"))
    space(S) == (V_cod ← V_dom) ||
        throw(SpaceMismatch("`svd_full!(t, (U, S, Vᴴ))` requires `space(S) == (fuse(codomain(t)) ← fuse(domain(t))`"))
    space(Vᴴ) == (V_dom ← domain(t)) ||
        throw(SpaceMismatch("`svd_full!(t, (U, S, Vᴴ))` requires `space(Vᴴ) == (fuse(domain(t)) ← domain(t))`"))

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
    space(U) == (codomain(t) ← V_cod) ||
        throw(SpaceMismatch("`svd_compact!(t, (U, S, Vᴴ))` requires `space(U) == (codomain(t) ← infimum(fuse(domain(t)), fuse(codomain(t)))`"))
    space(S) == (V_cod ← V_dom) ||
        throw(SpaceMismatch("`svd_compact!(t, (U, S, Vᴴ))` requires diagonal `S` with `domain(S) == (infimum(fuse(codomain(t)), fuse(domain(t)))`"))
    space(Vᴴ) == (V_dom ← domain(t)) ||
        throw(SpaceMismatch("`svd_compact!(t, (U, S, Vᴴ))` requires `space(Vᴴ) == (infimum(fuse(domain(t)), fuse(codomain(t))) ← domain(t))`"))

    return nothing
end

# TODO: svd_vals

function MatrixAlgebraKit.initialize_output(::typeof(svd_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, domain(t) ← V_cod)
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

# TODO: svd_vals

function MatrixAlgebraKit.svd_full!(t::AbstractTensorMap, (U, S, Vᴴ),
                                    alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(svd_full!, t, (U, S, Vᴴ))

    foreachblock(t, U, S, Vᴴ; alg.scheduler) do _, (b, u, s, vᴴ)
        if isempty(b) # TODO: remove once MatrixAlgebraKit supports empty matrices
            one!(length(u) > 0 ? u : vᴴ)
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

function MatrixAlgebraKit.default_svd_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                scheduler=default_blockscheduler(t),
                                                kwargs...)
    return BlockAlgorithm(LAPACK_DivideAndConquer(; kwargs...), scheduler)
end

# Eigenvalue decomposition
# ------------------------
const _T_DV = Tuple{<:DiagonalTensorMap,<:AbstractTensorMap}
function MatrixAlgebraKit.check_input(::typeof(eigh_full!), t::AbstractTensorMap,
                                      (D, V)::_T_DV)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    # scalartype checks
    @check_eltype D t real
    @check_eltype V t

    # space checks
    V_D = fuse(domain(t))
    V_D == space(D, 1) ||
        throw(SpaceMismatch("`eigh_full!(t, (D, V))` requires diagonal `D` with `domain(D) == fuse(domain(t))`"))
    space(V) == (codomain(t) ← V_D) ||
        throw(SpaceMismatch("`eigh_full!(t, (D, V))` requires `space(V) == (codomain(t) ← fuse(domain(t)))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(eig_full!), t::AbstractTensorMap,
                                      (D, V)::_T_DV)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    # scalartype checks
    @check_eltype D t complex
    @check_eltype V t complex

    # space checks
    V_D = fuse(domain(t))
    V_D == space(D, 1) ||
        throw(SpaceMismatch("`eig_full!(t, (D, V))` requires diagonal `D` with `domain(D) == fuse(domain(t))`"))
    space(V) == (codomain(t) ← V_D) ||
        throw(SpaceMismatch("`eig_full!(t, (D, V))` requires `space(V) == (codomain(t) ← fuse(domain(t)))`"))

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

function MatrixAlgebraKit.default_eig_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                scheduler=default_blockscheduler(t),
                                                kwargs...)
    return BlockAlgorithm(LAPACK_Expert(; kwargs...), scheduler)
end
function MatrixAlgebraKit.default_eigh_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                 scheduler=default_blockscheduler(t),
                                                 kwargs...)
    return BlockAlgorithm(LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...),
                          scheduler)
end

# QR decomposition
# ----------------
function MatrixAlgebraKit.check_input(::typeof(qr_full!), t::AbstractTensorMap,
                                      (Q,
                                       R)::Tuple{<:AbstractTensorMap,<:AbstractTensorMap})
    # scalartype checks
    @check_eltype Q t
    @check_eltype R t

    # space checks
    V_Q = fuse(codomain(t))
    space(Q) == (codomain(t) ← V_Q) ||
        throw(SpaceMismatch("`qr_full!(t, (Q, R))` requires `space(Q) == (codomain(t) ← fuse(codomain(t)))`"))
    space(R) == (V_Q ← domain(t)) ||
        throw(SpaceMismatch("`qr_full!(t, (Q, R))` requires `space(R) == (fuse(codomain(t)) ← domain(t)`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(qr_compact!), t::AbstractTensorMap, (Q, R))
    # scalartype checks
    @check_eltype Q t
    @check_eltype R t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    space(Q) == (codomain(t) ← V_Q) ||
        throw(SpaceMismatch("`qr_compact!(t, (Q, R))` requires `space(Q) == (codomain(t) ← infimum(fuse(codomain(t)), fuse(domain(t)))`"))
    space(R) == (V_Q ← domain(t)) ||
        throw(SpaceMismatch("`qr_compact!(t, (Q, R))` requires `space(R) == (infimum(fuse(codomain(t)), fuse(domain(t))) ← domain(t))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(qr_null!), t::AbstractTensorMap,
                                      N::AbstractTensorMap)
    # scalartype checks
    @check_eltype N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = setdiff(fuse(codomain(t)), V_Q)
    space(N) == (codomain(t) ← V_N) ||
        throw(SpaceMismatch("`qr_null!(t, N)` requires `space(N) == (codomain(t) ← setdiff(fuse(codomain(t)), infimum(fuse(codomain(t)), fuse(domain(t))))`"))

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
    V_N = setdiff(fuse(codomain(t)), V_Q)
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

function MatrixAlgebraKit.default_qr_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                               scheduler=default_blockscheduler(t),
                                               kwargs...)
    return BlockAlgorithm(LAPACK_HouseholderQR(; kwargs...), scheduler)
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
    @check_eltype L t
    @check_eltype Q t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    space(L) == (codomain(t) ← V_Q) ||
        throw(SpaceMismatch("`lq_compact!(t, (L, Q))` requires `space(L) == infimum(fuse(codomain(t)), fuse(domain(t)))`"))
    space(Q) == (V_Q ← domain(t)) ||
        throw(SpaceMismatch("`lq_compact!(t, (L, Q))` requires `space(Q) == infimum(fuse(codomain(t)), fuse(domain(t)))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(lq_null!), t::AbstractTensorMap, N)
    # scalartype checks
    @check_eltype N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = setdiff(fuse(domain(t)), V_Q)
    space(N) == (V_N ← domain(t)) ||
        throw(SpaceMismatch("`lq_null!(t, N)` requires `space(N) == setdiff(fuse(domain(t)), infimum(fuse(codomain(t)), fuse(domain(t)))`"))

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
    V_N = setdiff(fuse(domain(t)), V_Q)
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
    @check_eltype W t
    @check_eltype P t

    # space checks
    space(W) == (codomain(t) ← fuse(domain(t))) ||
        throw(SpaceMismatch("`left_polar!(t, (W, P))` requires `space(W) == (codomain(t) ← domain(t))`"))
    space(P) == (fuse(domain(t)) ← domain(t)) ||
        throw(SpaceMismatch("`left_polar!(t, (W, P))` requires `space(P) == (domain(t) ← domain(t))`"))

    return nothing
end

# TODO: do we really not want to fuse the spaces?
function MatrixAlgebraKit.initialize_output(::typeof(left_polar!), t::AbstractTensorMap)
    W = similar(t, codomain(t) ← fuse(domain(t)))
    P = similar(t, fuse(domain(t)) ← domain(t))
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

function MatrixAlgebraKit.default_polar_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                  scheduler=default_blockscheduler(t),
                                                  kwargs...)
    return BlockAlgorithm(PolarViaSVD(LAPACK_DivideAndConquer(; kwargs...)),
                          scheduler)
end

# Orthogonalization
# -----------------
function MatrixAlgebraKit.check_input(::typeof(left_orth!), t::AbstractTensorMap, (V, C))
    # scalartype checks
    @check_eltype V t
    isnothing(C) || @check_eltype C t

    # space checks
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    space(V) == (codomain(t) ← V_C) ||
        throw(SpaceMismatch("`left_orth!(t, (V, C))` requires `space(V) == (codomain(t) ← infimum(fuse(codomain(t)), fuse(domain(t))))`"))
    isnothing(C) || space(C) == (V_C ← domain(t)) ||
        throw(SpaceMismatch("`left_orth!(t, (V, C))` requires `space(C) == (infimum(fuse(codomain(t)), fuse(domain(t))) ← domain(t))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(right_orth!), t::AbstractTensorMap, (C, Vᴴ))
    # scalartype checks
    isnothing(C) || @check_eltype C t
    @check_eltype Vᴴ t

    # space checks
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    isnothing(C) || space(C) == (codomain(t) ← V_C) ||
        throw(SpaceMismatch("`right_orth!(t, (C, Vᴴ))` requires `space(C) == (codomain(t) ← infimum(fuse(codomain(t)), fuse(domain(t)))`"))
    space(Vᴴ) == (V_dom ← domain(t)) ||
        throw(SpaceMismatch("`right_orth!(t, (C, Vᴴ))` requires `space(Vᴴ) == (infimum(fuse(codomain(t)), fuse(domain(t))) ← domain(t))`"))

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

function MatrixAlgebraKit.left_orth!(t::AbstractTensorMap, VC; kwargs...)
    MatrixAlgebraKit.check_input(left_orth!, t, VC)
    atol = get(kwargs, :atol, 0)
    rtol = get(kwargs, :rtol, 0)
    kind = get(kwargs, :kind, iszero(atol) && iszero(rtol) ? :qrpos : :svd)

    if !(iszero(atol) && iszero(rtol)) && kind != :svd
        throw(ArgumentError("nonzero tolerance not supported for left_orth with kind=$kind"))
    end

    if kind == :qr
        alg = get(kwargs, :alg, MatrixAlgebraKit.select_algorithm(qr_compact!, t))
        return qr_compact!(t, VC, alg)
    elseif kind == :qrpos
        alg = get(kwargs, :alg,
                  MatrixAlgebraKit.select_algorithm(qr_compact!, t; positive=true))
        return qr_compact!(t, VC, alg)
    elseif kind == :polar
        alg = get(kwargs, :alg, MatrixAlgebraKit.select_algorithm(left_polar!, t))
        return left_polar!(t, VC, alg)
    elseif kind == :svd && iszero(atol) && iszero(rtol)
        alg = get(kwargs, :alg, MatrixAlgebraKit.select_algorithm(svd_compact!, t))
        V, C = VC
        S = DiagonalTensorMap{real(scalartype(t))}(undef, domain(V) ← codomain(C))
        U, S, Vᴴ = svd_compact!(t, (V, S, C), alg)
        return U, lmul!(S, Vᴴ)
    elseif kind == :svd
        alg_svd = MatrixAlgebraKit.select_algorithm(svd_compact!, t)
        trunc = MatrixAlgebraKit.TruncationKeepAbove(atol, rtol)
        alg = get(kwargs, :alg, MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trunc))
        V, C = VC
        S = DiagonalTensorMap{real(scalartype(t))}(undef, domain(V) ← codomain(C))
        U, S, Vᴴ = svd_trunc!(t, (V, S, C), alg)
        return U, lmul!(S, Vᴴ)
    else
        throw(ArgumentError("`left_orth!` received unknown value `kind = $kind`"))
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
