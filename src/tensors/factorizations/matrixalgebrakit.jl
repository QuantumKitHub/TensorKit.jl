# Algorithm selection
# -------------------
for f! in
    [:svd_compact!, :svd_full!, :svd_trunc!, :svd_vals!, :qr_compact!, :qr_full!, :qr_null!,
     :lq_compact!, :lq_full!, :lq_null!, :eig_full!, :eig_trunc!, :eig_vals!, :eigh_full!,
     :eigh_trunc!, :eigh_vals!, :left_polar!, :right_polar!, :left_orth!, :right_orth!]
    @eval function default_algorithm(::typeof($f!), ::Type{T};
                                     kwargs...) where {T<:AbstractTensorMap}
        return default_algorithm($f!, blocktype(T); kwargs...)
    end
end

function _select_truncation(f, ::AbstractTensorMap,
                            trunc::MatrixAlgebraKit.TruncationStrategy)
    return trunc
end
function _select_truncation(::typeof(left_null!), ::AbstractTensorMap, trunc::NamedTuple)
    return MatrixAlgebraKit.null_truncation_strategy(; trunc...)
end

# Generic Implementations
# ----------------------_
for f! in (:qr_compact!, :qr_full!,
           :lq_compact!, :lq_full!,
           :eig_full!, :eigh_full!,
           :svd_compact!, :svd_full!,
           :left_polar!, :left_orth_polar!,
           :right_polar!, :right_orth_polar!,
           :left_orth!, :right_orth!)
    @eval function $f!(t::AbstractTensorMap, F, alg::AbstractAlgorithm)
        check_input($f!, t, F, alg)

        foreachblock(t, F...) do _, bs
            factors = Base.tail(bs)
            factors′ = $f!(first(bs), factors, alg)
            # deal with the case where the output is not in-place
            for (f′, f) in zip(factors′, factors)
                f′ === f || copyto!(f, f′)
            end
            return nothing
        end

        return F
    end
end

# Handle these separately because single output instead of tuple
for f! in (:qr_null!, :lq_null!, :svd_vals!, :eig_vals!, :eigh_vals!)
    @eval function $f!(t::AbstractTensorMap, N, alg::AbstractAlgorithm)
        check_input($f!, t, N, alg)

        foreachblock(t, N) do _, (b, n)
            n′ = $f!(b, n, alg)
            # deal with the case where the output is not the same as the input
            n === n′ || copyto!(n, n′)
            return nothing
        end

        return N
    end
end

# Singular value decomposition
# ----------------------------
const _T_USVᴴ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap,<:AbstractTensorMap}
const _T_USVᴴ_diag = Tuple{<:AbstractTensorMap,<:DiagonalTensorMap,<:AbstractTensorMap}

function check_input(::typeof(svd_full!), t::AbstractTensorMap, (U, S, Vᴴ)::_T_USVᴴ, ::AbstractAlgorithm)
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

function check_input(::typeof(svd_compact!), t::AbstractTensorMap, (U, S, Vᴴ)::_T_USVᴴ_diag, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar U t
    @check_scalar S t real
    @check_scalar Vᴴ t

    # space checks
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(U, codomain(t) ← V_cod)
    @check_space(S, V_cod ← V_dom)
    @check_space(Vᴴ, V_dom ← domain(t))

    return nothing
end

function check_input(::typeof(svd_vals!), t::AbstractTensorMap, S::SectorDict, ::AbstractAlgorithm)
    @check_scalar S t real
    V_cod = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(S, V_cod ← V_dom)
    return nothing
end

function initialize_output(::typeof(svd_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function initialize_output(::typeof(svd_compact!), t::AbstractTensorMap,
                           ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function initialize_output(::typeof(svd_trunc!), t::AbstractTensorMap,
                           alg::TruncatedAlgorithm)
    return initialize_output(svd_compact!, t, alg.alg)
end

function initialize_output(::typeof(svd_vals!), t::AbstractTensorMap,
                           alg::AbstractAlgorithm)
    V_cod = infimum(fuse(codomain(t)), fuse(domain(t)))
    return DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
end

function svd_trunc!(t::AbstractTensorMap, USVᴴ, alg::TruncatedAlgorithm)
    USVᴴ′ = svd_compact!(t, USVᴴ, alg.alg)
    return truncate!(svd_trunc!, USVᴴ′, alg.trunc)
end

# Eigenvalue decomposition
# ------------------------
const _T_DV = Tuple{<:DiagonalTensorMap,<:AbstractTensorMap}

function check_input(::typeof(eigh_full!), t::AbstractTensorMap, (D, V)::_T_DV, ::AbstractAlgorithm)
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

function check_input(::typeof(eig_full!), t::AbstractTensorMap, (D, V)::_T_DV, ::AbstractAlgorithm)
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

function check_input(::typeof(eigh_vals!), t::AbstractTensorMap, D::DiagonalTensorMap, ::AbstractAlgorithm)
    @check_scalar D t real
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    return nothing
end

function check_input(::typeof(eig_vals!), t::AbstractTensorMap, D::DiagonalTensorMap, ::AbstractAlgorithm)
    @check_scalar D t complex
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    return nothing
end

function initialize_output(::typeof(eigh_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function initialize_output(::typeof(eig_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function initialize_output(::typeof(eigh_vals!), t::AbstractTensorMap,
                           alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    return D = DiagonalTensorMap{Tc}(undef, V_D)
end

function initialize_output(::typeof(eig_vals!), t::AbstractTensorMap,
                           alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    return D = DiagonalTensorMap{Tc}(undef, V_D)
end

function initialize_output(::typeof(eigh_trunc!), t::AbstractTensorMap,
                           alg::TruncatedAlgorithm)
    return initialize_output(eigh_full!, t, alg.alg)
end

function initialize_output(::typeof(eig_trunc!), t::AbstractTensorMap,
                           alg::TruncatedAlgorithm)
    return initialize_output(eig_full!, t, alg.alg)
end

function eigh_trunc!(t::AbstractTensorMap, DV, alg::TruncatedAlgorithm)
    DV′ = eigh_full!(t, DV, alg.alg)
    return truncate!(eigh_trunc!, DV′, alg.trunc)
end

function eig_trunc!(t::AbstractTensorMap, DV, alg::TruncatedAlgorithm)
    DV′ = eig_full!(t, DV, alg.alg)
    return truncate!(eig_trunc!, DV′, alg.trunc)
end

# QR decomposition
# ----------------
const _T_QR = Tuple{<:AbstractTensorMap,<:AbstractTensorMap}

function check_input(::typeof(qr_full!), t::AbstractTensorMap, (Q, R)::_T_QR, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = fuse(codomain(t))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function check_input(::typeof(qr_compact!), t::AbstractTensorMap, (Q, R)::_T_QR, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function check_input(::typeof(qr_null!), t::AbstractTensorMap, N::AbstractTensorMap, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    @check_space(N, codomain(t) ← V_N)

    return nothing
end

function initialize_output(::typeof(qr_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(codomain(t))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

function initialize_output(::typeof(qr_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

function initialize_output(::typeof(qr_null!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = similar(t, codomain(t) ← V_N)
    return N
end

# LQ decomposition
# ----------------
const _T_LQ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap}

function check_input(::typeof(lq_full!), t::AbstractTensorMap, (L, Q)::_T_LQ, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = fuse(domain(t))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end

function check_input(::typeof(lq_compact!), t::AbstractTensorMap, (L, Q)::_T_LQ, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end

function check_input(::typeof(lq_null!), t::AbstractTensorMap, N, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    @check_space(N, V_N ← domain(t))

    return nothing
end

function initialize_output(::typeof(lq_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(domain(t))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end

function initialize_output(::typeof(lq_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end

function initialize_output(::typeof(lq_null!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = similar(t, V_N ← domain(t))
    return N
end

# Polar decomposition
# -------------------
const _T_WP = Tuple{<:AbstractTensorMap,<:AbstractTensorMap}
const _T_PWᴴ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap}
using MatrixAlgebraKit: PolarViaSVD

function check_input(::typeof(left_polar!), t::AbstractTensorMap, (W, P)::_T_WP, ::AbstractAlgorithm)
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

function check_input(::typeof(left_orth_polar!), t::AbstractTensorMap, (W, P)::_T_WP, ::AbstractAlgorithm)
    codomain(t) ≿ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `codomain(t) ≿ domain(t)`"))

    # scalartype checks
    @check_scalar W t
    @check_scalar P t

    # space checks
    VW = fuse(domain(t))
    @check_space(W, codomain(t) ← VW)
    @check_space(P, VW ← domain(t))

    return nothing
end

function initialize_output(::typeof(left_polar!), t::AbstractTensorMap, ::AbstractAlgorithm)
    W = similar(t, space(t))
    P = similar(t, domain(t) ← domain(t))
    return W, P
end

function check_input(::typeof(right_polar!), t::AbstractTensorMap, (P, Wᴴ)::_T_PWᴴ, ::AbstractAlgorithm)
    codomain(t) ≾ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `domain(t) ≿ codomain(t)`"))

    # scalartype checks
    @check_scalar P t
    @check_scalar Wᴴ t

    # space checks
    @check_space(P, codomain(t) ← codomain(t))
    @check_space(Wᴴ, space(t))

    return nothing
end

function check_input(::typeof(right_orth_polar!), t::AbstractTensorMap, (P, Wᴴ)::_T_PWᴴ, ::AbstractAlgorithm)
    codomain(t) ≾ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `domain(t) ≿ codomain(t)`"))

    # scalartype checks
    @check_scalar P t
    @check_scalar Wᴴ t

    # space checks
    VW = fuse(codomain(t))
    @check_space(P, codomain(t) ← VW)
    @check_space(Wᴴ, VW ← domain(t))

    return nothing
end

function initialize_output(::typeof(right_polar!), t::AbstractTensorMap,
                           ::AbstractAlgorithm)
    P = similar(t, codomain(t) ← codomain(t))
    Wᴴ = similar(t, space(t))
    return Wᴴ, P
end

# Needed to get algorithm selection to behave
function left_orth_polar!(t::AbstractTensorMap, VC, alg)
    alg′ = MatrixAlgebraKit.select_algorithm(left_polar!, t, alg)
    return left_orth_polar!(t, VC, alg′)
end
function right_orth_polar!(t::AbstractTensorMap, CVᴴ, alg)
    alg′ = MatrixAlgebraKit.select_algorithm(right_polar!, t, alg)
    return right_orth_polar!(t, CVᴴ, alg′)
end

# Orthogonalization
# -----------------
const _T_VC = Tuple{<:AbstractTensorMap,<:AbstractTensorMap}
const _T_CVᴴ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap}

function check_input(::typeof(left_orth!), t::AbstractTensorMap, (V, C)::_T_VC, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar V t
    isnothing(C) || @check_scalar C t

    # space checks
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(V, codomain(t) ← V_C)
    isnothing(C) || @check_space(C, V_C ← domain(t))

    return nothing
end

function check_input(::typeof(right_orth!), t::AbstractTensorMap, (C, Vᴴ)::_T_CVᴴ, ::AbstractAlgorithm)
    # scalartype checks
    isnothing(C) || @check_scalar C t
    @check_scalar Vᴴ t

    # space checks
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    isnothing(C) || @check_space(C, codomain(t) ← V_C)
    @check_space(Vᴴ, V_C ← domain(t))

    return nothing
end

function initialize_output(::typeof(left_orth!), t::AbstractTensorMap)
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    V = similar(t, codomain(t) ← V_C)
    C = similar(t, V_C ← domain(t))
    return V, C
end

function initialize_output(::typeof(right_orth!), t::AbstractTensorMap)
    V_C = infimum(fuse(codomain(t)), fuse(domain(t)))
    C = similar(t, codomain(t) ← V_C)
    Vᴴ = similar(t, V_C ← domain(t))
    return C, Vᴴ
end

# Nullspace
# ---------
function check_input(::typeof(left_null!), t::AbstractTensorMap, N, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    @check_space(N, codomain(t) ← V_N)

    return nothing
end

function check_input(::typeof(right_null!), t::AbstractTensorMap, N, ::AbstractAlgorithm)
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    @check_space(N, V_N ← domain(t))

    return nothing
end

function initialize_output(::typeof(left_null!), t::AbstractTensorMap)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = similar(t, codomain(t) ← V_N)
    return N
end

function initialize_output(::typeof(right_null!), t::AbstractTensorMap)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = similar(t, V_N ← domain(t))
    return N
end

for f! in (:left_null_svd!, :right_null_svd!)
    @eval function $f!(t::AbstractTensorMap, N, alg, ::Nothing=nothing)
        foreachblock(t, N) do _, (b, n)
            n′ = $f!(b, n, alg)
            # deal with the case where the output is not the same as the input
            n === n′ || copyto!(n, n′)
            return nothing
        end

        return N
    end
end
