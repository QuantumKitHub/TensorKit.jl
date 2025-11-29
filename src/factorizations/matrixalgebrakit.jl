# Algorithm selection
# -------------------
for f in
    [
        :svd_compact, :svd_full, :svd_vals,
        :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null,
        :eig_full, :eig_vals, :eigh_full, :eigh_vals,
        :left_polar, :right_polar,
        :project_hermitian, :project_antihermitian, :project_isometric,
    ]
    f! = Symbol(f, :!)
    @eval function MAK.default_algorithm(::typeof($f!), ::Type{T}; kwargs...) where {T <: AbstractTensorMap}
        return MAK.default_algorithm($f!, blocktype(T); kwargs...)
    end
    @eval function MAK.copy_input(::typeof($f), t::AbstractTensorMap)
        return copy_oftype(t, factorisation_scalartype($f, t))
    end
end

_select_truncation(f, ::AbstractTensorMap, trunc::TruncationStrategy) = trunc
function _select_truncation(::typeof(left_null!), ::AbstractTensorMap, trunc::NamedTuple)
    return MAK.null_truncation_strategy(; trunc...)
end

# Generic Implementations
# -----------------------
for f! in (
        :qr_compact!, :qr_full!, :lq_compact!, :lq_full!,
        :eig_full!, :eigh_full!, :svd_compact!, :svd_full!,
        :left_polar!, :right_polar!,
    )
    @eval function MAK.$f!(t::AbstractTensorMap, F, alg::AbstractAlgorithm)
        MAK.check_input($f!, t, F, alg)

        foreachblock(t, F...) do _, (tblock, Fblocks...)
            Fblocks′ = $f!(tblock, Fblocks, alg)
            # deal with the case where the output is not in-place
            for (b′, b) in zip(Fblocks′, Fblocks)
                b === b′ || copy!(b, b′)
            end
            return nothing
        end

        return F
    end
end

# Handle these separately because single output instead of tuple
for f! in (:qr_null!, :lq_null!, :project_hermitian!, :project_antihermitian!, :project_isometric!)
    @eval function MAK.$f!(t::AbstractTensorMap, N, alg::AbstractAlgorithm)
        MAK.check_input($f!, t, N, alg)

        foreachblock(t, N) do _, (tblock, Nblock)
            Nblock′ = $f!(tblock, Nblock, alg)
            # deal with the case where the output is not the same as the input
            Nblock === Nblock′ || copy!(Nblock, Nblock′)
            return nothing
        end

        return N
    end
end

# Handle these separately because single output instead of tuple
for f! in (:svd_vals!, :eig_vals!, :eigh_vals!)
    @eval function MAK.$f!(t::AbstractTensorMap, N, alg::AbstractAlgorithm)
        MAK.check_input($f!, t, N, alg)

        foreachblock(t, N) do _, (tblock, Nblock)
            Nblock′ = $f!(tblock, diagview(Nblock), alg)
            # deal with the case where the output is not the same as the input
            diagview(Nblock) === Nblock′ || copy!(diagview(Nblock), Nblock′)
            return nothing
        end

        return N
    end
end

# Singular value decomposition
# ----------------------------
function MAK.check_input(::typeof(svd_full!), t::AbstractTensorMap, USVᴴ, ::AbstractAlgorithm)
    U, S, Vᴴ = USVᴴ

    # type checks
    @assert U isa AbstractTensorMap
    @assert S isa AbstractTensorMap
    @assert Vᴴ isa AbstractTensorMap

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

function MAK.check_input(::typeof(svd_compact!), t::AbstractTensorMap, USVᴴ, ::AbstractAlgorithm)
    U, S, Vᴴ = USVᴴ

    # type checks
    @assert U isa AbstractTensorMap
    @assert S isa DiagonalTensorMap
    @assert Vᴴ isa AbstractTensorMap

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

function MAK.check_input(::typeof(svd_vals!), t::AbstractTensorMap, D, ::AbstractAlgorithm)
    @check_scalar D t real
    @assert D isa DiagonalTensorMap
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(D, V_cod ← V_dom)
    return nothing
end

function MAK.initialize_output(::typeof(svd_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, codomain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function MAK.initialize_output(::typeof(svd_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, codomain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function MAK.initialize_output(::typeof(svd_vals!), t::AbstractTensorMap, alg::AbstractAlgorithm)
    V_cod = infimum(fuse(codomain(t)), fuse(domain(t)))
    return DiagonalTensorMap{real(scalartype(t))}(undef, V_cod)
end

# Eigenvalue decomposition
# ------------------------
function MAK.check_input(::typeof(eigh_full!), t::AbstractTensorMap, DV, ::AbstractAlgorithm)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    D, V = DV

    # type checks
    @assert D isa DiagonalTensorMap
    @assert V isa AbstractTensorMap

    # scalartype checks
    @check_scalar D t real
    @check_scalar V t

    # space checks
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    @check_space(V, codomain(t) ← V_D)

    return nothing
end

function MAK.check_input(::typeof(eig_full!), t::AbstractTensorMap, DV, ::AbstractAlgorithm)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    D, V = DV

    # type checks
    @assert D isa DiagonalTensorMap
    @assert V isa AbstractTensorMap

    # scalartype checks
    @check_scalar D t complex
    @check_scalar V t complex

    # space checks
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    @check_space(V, codomain(t) ← V_D)

    return nothing
end

function MAK.check_input(::typeof(eigh_vals!), t::AbstractTensorMap, D, ::AbstractAlgorithm)
    @check_scalar D t real
    @assert D isa DiagonalTensorMap
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    return nothing
end

function MAK.check_input(::typeof(eig_vals!), t::AbstractTensorMap, D, ::AbstractAlgorithm)
    @check_scalar D t complex
    @assert D isa DiagonalTensorMap
    V_D = fuse(domain(t))
    @check_space(D, V_D ← V_D)
    return nothing
end

function MAK.initialize_output(::typeof(eigh_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(eig_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(eigh_vals!), t::AbstractTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    return D = DiagonalTensorMap{Tc}(undef, V_D)
end

function MAK.initialize_output(::typeof(eig_vals!), t::AbstractTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    return D = DiagonalTensorMap{Tc}(undef, V_D)
end

# QR decomposition
# ----------------
function MAK.check_input(::typeof(qr_full!), t::AbstractTensorMap, QR, ::AbstractAlgorithm)
    Q, R = QR

    # type checks
    @assert Q isa AbstractTensorMap
    @assert R isa AbstractTensorMap

    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = fuse(codomain(t))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function MAK.check_input(::typeof(qr_compact!), t::AbstractTensorMap, QR, ::AbstractAlgorithm)
    Q, R = QR

    # type checks
    @assert Q isa AbstractTensorMap
    @assert R isa AbstractTensorMap

    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function MAK.check_input(::typeof(qr_null!), t::AbstractTensorMap, N, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    @check_space(N, codomain(t) ← V_N)

    return nothing
end

function MAK.initialize_output(::typeof(qr_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(codomain(t))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

function MAK.initialize_output(::typeof(qr_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

function MAK.initialize_output(::typeof(qr_null!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(codomain(t)), V_Q)
    N = similar(t, codomain(t) ← V_N)
    return N
end

# LQ decomposition
# ----------------
function MAK.check_input(::typeof(lq_full!), t::AbstractTensorMap, LQ, ::AbstractAlgorithm)
    L, Q = LQ

    # type checks
    @assert L isa AbstractTensorMap
    @assert Q isa AbstractTensorMap

    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = fuse(domain(t))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end

function MAK.check_input(::typeof(lq_compact!), t::AbstractTensorMap, LQ, ::AbstractAlgorithm)
    L, Q = LQ

    # type checks
    @assert L isa AbstractTensorMap
    @assert Q isa AbstractTensorMap

    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end

function MAK.check_input(::typeof(lq_null!), t::AbstractTensorMap, N, ::AbstractAlgorithm)
    # scalartype checks
    @check_scalar N t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    @check_space(N, V_N ← domain(t))

    return nothing
end

function MAK.initialize_output(::typeof(lq_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = fuse(domain(t))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end

function MAK.initialize_output(::typeof(lq_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end

function MAK.initialize_output(::typeof(lq_null!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    V_N = ⊖(fuse(domain(t)), V_Q)
    N = similar(t, V_N ← domain(t))
    return N
end

# Polar decomposition
# -------------------
function MAK.check_input(::typeof(left_polar!), t::AbstractTensorMap, WP, ::AbstractAlgorithm)
    codomain(t) ≿ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `codomain(t) ≿ domain(t)`"))

    W, P = WP
    @assert W isa AbstractTensorMap
    @assert P isa AbstractTensorMap

    # scalartype checks
    @check_scalar W t
    @check_scalar P t

    # space checks
    @check_space(W, space(t))
    @check_space(P, domain(t) ← domain(t))

    return nothing
end

function MAK.initialize_output(::typeof(left_polar!), t::AbstractTensorMap, ::AbstractAlgorithm)
    W = similar(t, space(t))
    P = similar(t, domain(t) ← domain(t))
    return W, P
end

function MAK.check_input(::typeof(right_polar!), t::AbstractTensorMap, PWᴴ, ::AbstractAlgorithm)
    codomain(t) ≾ domain(t) ||
        throw(ArgumentError("Polar decomposition requires `domain(t) ≿ codomain(t)`"))

    P, Wᴴ = PWᴴ
    @assert P isa AbstractTensorMap
    @assert Wᴴ isa AbstractTensorMap

    # scalartype checks
    @check_scalar P t
    @check_scalar Wᴴ t

    # space checks
    @check_space(P, codomain(t) ← codomain(t))
    @check_space(Wᴴ, space(t))

    return nothing
end

function MAK.initialize_output(::typeof(right_polar!), t::AbstractTensorMap, ::AbstractAlgorithm)
    P = similar(t, codomain(t) ← codomain(t))
    Wᴴ = similar(t, space(t))
    return P, Wᴴ
end

# Projections
# -----------
function MAK.check_input(::typeof(project_hermitian!), tsrc::AbstractTensorMap, tdst::AbstractTensorMap, ::AbstractAlgorithm)
    domain(tsrc) == codomain(tsrc) || throw(ArgumentError("(Anti-)Hermitian projection requires square input tensor"))
    tsrc === tdst || @check_space(tdst, space(tsrc))
    return nothing
end

MAK.check_input(::typeof(project_antihermitian!), tsrc::AbstractTensorMap, tdst::AbstractTensorMap, alg::AbstractAlgorithm) =
    MAK.check_input(project_hermitian!, tsrc, tdst, alg)

function MAK.check_input(::typeof(project_isometric!), t::AbstractTensorMap, W::AbstractTensorMap, alg::AbstractAlgorithm)
    codomain(t) ≿ domain(t) || throw(ArgumentError("Isometric projection requires `codomain(t) ≿ domain(t)`"))
    @check_space W space(t)
    @check_scalar(W, t)

    return nothing
end


MAK.initialize_output(::typeof(project_hermitian!), tsrc::AbstractTensorMap, ::AbstractAlgorithm) = tsrc
MAK.initialize_output(::typeof(project_antihermitian!), tsrc::AbstractTensorMap, ::AbstractAlgorithm) = tsrc
MAK.initialize_output(::typeof(project_isometric!), tsrc::AbstractTensorMap, ::AbstractAlgorithm) = similar(tsrc)
