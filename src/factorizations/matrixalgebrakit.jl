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
    output_type = similarstoragetype(t, real(scalartype(t)))
    S = DiagonalTensorMap{real(scalartype(t)), typeof(V_cod), output_type}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end

function MAK.initialize_output(::typeof(svd_vals!), t::AbstractTensorMap, alg::AbstractAlgorithm)
    V_cod = infimum(fuse(codomain(t)), fuse(domain(t)))
    output_storage_type = similarstoragetype(t, real(scalartype(t)))
    return DiagonalTensorMap{real(scalartype(t)), typeof(V_cod), output_storage_type}(undef, V_cod)
end

# Eigenvalue decomposition
# ------------------------
function MAK.initialize_output(::typeof(eigh_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T, typeof(V_D), similarstoragetype(t, T)}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(eig_full!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc, typeof(V_D), similarstoragetype(t, Tc)}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

function MAK.initialize_output(::typeof(eigh_vals!), t::AbstractTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    return D = DiagonalTensorMap{Tc, typeof(V_D), storagetype(t)}(undef, V_D)
end

function MAK.initialize_output(::typeof(eig_vals!), t::AbstractTensorMap, alg::AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    return D = DiagonalTensorMap{Tc, typeof(V_D), similarstoragetype(t, Tc)}(undef, V_D)
end

# QR decomposition
# ----------------
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
function MAK.initialize_output(::typeof(left_polar!), t::AbstractTensorMap, ::AbstractAlgorithm)
    W = similar(t, space(t))
    P = similar(t, domain(t) ← domain(t))
    return W, P
end

function MAK.initialize_output(::typeof(right_polar!), t::AbstractTensorMap, ::AbstractAlgorithm)
    P = similar(t, codomain(t) ← codomain(t))
    Wᴴ = similar(t, space(t))
    return P, Wᴴ
end

# Projections
# -----------
MAK.initialize_output(::typeof(project_hermitian!), tsrc::AbstractTensorMap, ::AbstractAlgorithm) =
    tsrc
MAK.initialize_output(::typeof(project_antihermitian!), tsrc::AbstractTensorMap, ::AbstractAlgorithm) =
    tsrc
MAK.initialize_output(::typeof(project_isometric!), tsrc::AbstractTensorMap, ::AbstractAlgorithm) =
    similar(tsrc)
