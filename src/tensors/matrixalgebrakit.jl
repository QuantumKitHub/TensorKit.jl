function MAK.copy_input(::typeof(MAK.eig_full), t::AbstractTensorMap)
    return copy_oftype(t, factorisation_scalartype(MAK.eig_full!, t))
end

function factorisation_scalartype(::typeof(MAK.eig_full!), t::AbstractTensorMap)
    T = complex(scalartype(t))
    return promote_type(ComplexF32, typeof(zero(T) / sqrt(abs2(one(T)))))
end

function MAK.check_input(::typeof(MAK.eig_full!), t::AbstractTensorMap, (D, V))
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))
    Tc = complex(scalartype(t))

    (D isa DiagonalTensorMap &&
     scalartype(D) == Tc &&
     fuse(domain(t)) == space(D, 1)) ||
        throw(ArgumentError("`eig_full!` requires diagonal tensor D with isomorphic domain and complex `scalartype`"))

    V isa AbstractTensorMap &&
        scalartype(V) == Tc &&
        space(V) == (codomain(t) ← codomain(D)) ||
        throw(ArgumentError("`eig_full!` requires square tensor V with isomorphic domain and complex `scalartype`"))

    return nothing
end

function MAK.initialize_output(::typeof(MAK.eig_full!), t::AbstractTensorMap,
                               ::MAK.LAPACK_EigAlgorithm)
    Tc = complex(scalartype(t))
    V_diag = fuse(domain(t))
    return DiagonalTensorMap{Tc}(undef, V_diag), similar(t, Tc, domain(t) ← V_diag)
end

function MAK.eig_full!(t::AbstractTensorMap, (D, V), alg::MAK.LAPACK_EigAlgorithm)
    MAK.check_input(MAK.eig_full!, t, (D, V))
    foreachblock(t, D, V) do (_, (b, d, v))
        d′, v′ = MAK.eig_full!(b, (d, v), alg)
        # deal with the case where the output is not the same as the input
        d === d′ || copyto!(d, d′)
        v === v′ || copyto!(v, v′)
        return nothing
    end
    return D, V
end

function MAK.default_eig_algorithm(::TensorMap{<:LinearAlgebra.BlasFloat}; kwargs...)
    return MAK.LAPACK_Expert(; kwargs...)
end
