# AdjointTensorMap
# ----------------
# map algorithms to their adjoint counterpart
# TODO: this probably belongs in MatrixAlgebraKit
_adjoint(alg::MAK.LAPACK_HouseholderQR) = MAK.LAPACK_HouseholderLQ(; alg.kwargs...)
_adjoint(alg::MAK.LAPACK_HouseholderLQ) = MAK.LAPACK_HouseholderQR(; alg.kwargs...)
_adjoint(alg::MAK.LAPACK_HouseholderQL) = MAK.LAPACK_HouseholderRQ(; alg.kwargs...)
_adjoint(alg::MAK.LAPACK_HouseholderRQ) = MAK.LAPACK_HouseholderQL(; alg.kwargs...)
_adjoint(alg::MAK.PolarViaSVD) = MAK.PolarViaSVD(_adjoint(alg.svdalg))
_adjoint(alg::AbstractAlgorithm) = alg

for f in
    [
        :svd_compact, :svd_full, :svd_trunc, :svd_vals, :qr_compact, :qr_full, :qr_null,
        :lq_compact, :lq_full, :lq_null, :eig_full, :eig_trunc, :eig_vals, :eigh_full,
        :eigh_trunc, :eigh_vals, :left_polar, :right_polar,
        :project_hermitian, :project_antihermitian, :project_isometric,
    ]
    f! = Symbol(f, :!)
    # just return the algorithm for the parent type since we are mapping this with
    # `_adjoint` afterwards anyways.
    # TODO: properly handle these cases
    @eval MAK.default_algorithm(::typeof($f!), ::Type{T}; kwargs...) where {T <: AdjointTensorMap} =
        MAK.default_algorithm($f!, TensorKit.parenttype(T); kwargs...)
end

# 1-arg functions
MAK.initialize_output(::typeof(qr_null!), t::AdjointTensorMap, alg::AbstractAlgorithm) =
    adjoint(MAK.initialize_output(lq_null!, adjoint(t), _adjoint(alg)))
MAK.initialize_output(::typeof(lq_null!), t::AdjointTensorMap, alg::AbstractAlgorithm) =
    adjoint(MAK.initialize_output(qr_null!, adjoint(t), _adjoint(alg)))

MAK.qr_null!(t::AdjointTensorMap, N, alg::AbstractAlgorithm) =
    lq_null!(adjoint(t), adjoint(N), _adjoint(alg))
MAK.lq_null!(t::AdjointTensorMap, N, alg::AbstractAlgorithm) =
    qr_null!(adjoint(t), adjoint(N), _adjoint(alg))

MAK.is_left_isometric(t::AdjointTensorMap; kwargs...) =
    MAK.is_right_isometric(adjoint(t); kwargs...)
MAK.is_right_isometric(t::AdjointTensorMap; kwargs...) =
    MAK.is_left_isometric(adjoint(t); kwargs...)

# 2-arg functions
for (left_f, right_f) in zip(
        (:qr_full, :qr_compact, :left_polar),
        (:lq_full, :lq_compact, :right_polar)
    )
    left_f! = Symbol(left_f, :!)
    right_f! = Symbol(right_f, :!)
    @eval function MAK.copy_input(::typeof($left_f!), t::AdjointTensorMap)
        return adjoint(MAK.copy_input($right_f!, adjoint(t)))
    end
    @eval function MAK.copy_input(::typeof($right_f!), t::AdjointTensorMap)
        return adjoint(MAK.copy_input($left_f!, adjoint(t)))
    end

    @eval function MAK.initialize_output(
            ::typeof($left_f!), t::AdjointTensorMap, alg::AbstractAlgorithm
        )
        return reverse(adjoint.(MAK.initialize_output($right_f!, adjoint(t), _adjoint(alg))))
    end
    @eval function MAK.initialize_output(
            ::typeof($right_f!), t::AdjointTensorMap, alg::AbstractAlgorithm
        )
        return reverse(adjoint.(MAK.initialize_output($left_f!, adjoint(t), _adjoint(alg))))
    end

    @eval function MAK.$left_f!(t::AdjointTensorMap, F, alg::AbstractAlgorithm)
        F′ = $right_f!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
        return reverse(adjoint.(F′))
    end
    @eval function MAK.$right_f!(t::AdjointTensorMap, F, alg::AbstractAlgorithm)
        F′ = $left_f!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
        return reverse(adjoint.(F′))
    end
end

# 3-arg functions
for f! in (:svd_full!, :svd_compact!, :svd_trunc!)
    @eval function MAK.copy_input(::typeof($f!), t::AdjointTensorMap)
        return adjoint(MAK.copy_input($f!, adjoint(t)))
    end

    @eval function MAK.initialize_output(
            ::typeof($f!), t::AdjointTensorMap, alg::AbstractAlgorithm
        )
        return reverse(adjoint.(MAK.initialize_output($f!, adjoint(t), _adjoint(alg))))
    end
    @eval function MAK.$f!(t::AdjointTensorMap, F, alg::AbstractAlgorithm)
        F′ = $f!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
        return reverse(adjoint.(F′))
    end

    # disambiguate by prohibition
    @eval function MAK.initialize_output(
            ::typeof($f!), t::AdjointTensorMap, alg::DiagonalAlgorithm
        )
        throw(MethodError($f!, (t, alg)))
    end
end
# avoid amgiguity
function MAK.initialize_output(
        ::typeof(svd_trunc!), t::AdjointTensorMap, alg::TruncatedAlgorithm
    )
    return MAK.initialize_output(svd_compact!, t, alg.alg)
end
# to fix ambiguity
function MAK.svd_trunc!(t::AdjointTensorMap, USVᴴ, alg::TruncatedAlgorithm)
    USVᴴ′ = svd_compact!(t, USVᴴ, alg.alg)
    return MAK.truncate(svd_trunc!, USVᴴ′, alg.trunc)
end
function MAK.svd_compact!(t::AdjointTensorMap, F, alg::DiagonalAlgorithm)
    F′ = svd_compact!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
    return reverse(adjoint.(F′))
end
