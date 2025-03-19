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

# function factorisation_scalartype(::typeof(MAK.eig_full!), t::AbstractTensorMap)
#     T = scalartype(t)
#     return promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))
# end

# Singular value decomposition
# ----------------------------
function MatrixAlgebraKit.check_input(::typeof(svd_full!), t::AbstractTensorMap, (U, S, Vᴴ))
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))

    (U isa AbstractTensorMap &&
     scalartype(U) == scalartype(t) &&
     space(U) == (codomain(t) ← V_cod)) ||
        throw(ArgumentError("`svd_full!` requires unitary tensor U with same `scalartype`"))
    (S isa AbstractTensorMap &&
     scalartype(S) == real(scalartype(t)) &&
     space(S) == (V_cod ← V_dom)) ||
        throw(ArgumentError("`svd_full!` requires rectangular tensor S with real `scalartype`"))
    (Vᴴ isa AbstractTensorMap &&
     scalartype(Vᴴ) == scalartype(t) &&
     space(Vᴴ) == (V_dom ← domain(t))) ||
        throw(ArgumentError("`svd_full!` requires unitary tensor Vᴴ with same `scalartype`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(svd_compact!), t::AbstractTensorMap,
                                      (U, S, Vᴴ))
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))

    (U isa AbstractTensorMap &&
     scalartype(U) == scalartype(t) &&
     space(U) == (codomain(t) ← V_cod)) ||
        throw(ArgumentError("`svd_compact!` requires isometric tensor U with same `scalartype`"))
    (S isa DiagonalTensorMap &&
     scalartype(S) == real(scalartype(t)) &&
     space(S) == (V_cod ← V_dom)) ||
        throw(ArgumentError("`svd_compact!` requires diagonal tensor S with real `scalartype`"))
    (Vᴴ isa AbstractTensorMap &&
     scalartype(Vᴴ) == scalartype(t) &&
     space(Vᴴ) == (V_dom ← domain(t))) ||
        throw(ArgumentError("`svd_compact!` requires isometric tensor Vᴴ with same `scalartype`"))

    return nothing
end

# TODO: svd_vals

function MatrixAlgebraKit.initialize_output(::typeof(svd_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, domain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = similar(t, domain(t) ← V_dom)
    return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(::typeof(svd_compact!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, domain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod ← V_dom)
    Vᴴ = similar(t, domain(t) ← V_dom)
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
function MatrixAlgebraKit.check_input(::typeof(eigh_full!), t::AbstractTensorMap, (D, V))
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    V_D = fuse(domain(t))

    (D isa DiagonalTensorMap &&
     scalartype(D) == real(scalartype(t)) &&
     V_D == space(D, 1)) ||
        throw(ArgumentError("`eigh_full!` requires diagonal tensor D with isomorphic domain and real `scalartype`"))

    V isa AbstractTensorMap &&
        scalartype(V) == scalartype(t) &&
        space(V) == (codomain(t) ← V_D) ||
        throw(ArgumentError("`eigh_full!` requires square tensor V with isomorphic domain and equal `scalartype`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(eig_full!), t::AbstractTensorMap, (D, V))
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))
    Tc = complex(scalartype(t))
    V_D = fuse(domain(t))

    (D isa DiagonalTensorMap &&
     scalartype(D) == Tc &&
     V_D == space(D, 1)) ||
        throw(ArgumentError("`eig_full!` requires diagonal tensor D with isomorphic domain and complex `scalartype`"))

    V isa AbstractTensorMap &&
        scalartype(V) == Tc &&
        space(V) == (codomain(t) ← V_D) ||
        throw(ArgumentError("`eig_full!` requires square tensor V with isomorphic domain and complex `scalartype`"))

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
