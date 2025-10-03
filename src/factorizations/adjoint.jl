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

# 1-arg functions
function MAK.initialize_output(::typeof(left_null!), t::AdjointTensorMap, alg::AbstractAlgorithm)
    return adjoint(MAK.initialize_output(right_null!, adjoint(t), _adjoint(alg)))
end
function MAK.initialize_output(
        ::typeof(right_null!), t::AdjointTensorMap,
        alg::AbstractAlgorithm
    )
    return adjoint(MAK.initialize_output(left_null!, adjoint(t), _adjoint(alg)))
end

function MAK.left_null!(t::AdjointTensorMap, N, alg::AbstractAlgorithm)
    right_null!(adjoint(t), adjoint(N), _adjoint(alg))
    return N
end
function MAK.right_null!(t::AdjointTensorMap, N, alg::AbstractAlgorithm)
    left_null!(adjoint(t), adjoint(N), _adjoint(alg))
    return N
end

function MAK.is_left_isometry(t::AdjointTensorMap; kwargs...)
    return is_right_isometry(adjoint(t); kwargs...)
end
function MAK.is_right_isometry(t::AdjointTensorMap; kwargs...)
    return is_left_isometry(adjoint(t); kwargs...)
end

# 2-arg functions
for (left_f!, right_f!) in zip(
        (:qr_full!, :qr_compact!, :left_polar!, :left_orth!),
        (:lq_full!, :lq_compact!, :right_polar!, :right_orth!)
    )
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
        $right_f!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
        return F
    end
    @eval function MAK.$right_f!(t::AdjointTensorMap, F, alg::AbstractAlgorithm)
        $left_f!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
        return F
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
        $f!(adjoint(t), reverse(adjoint.(F)), _adjoint(alg))
        return F
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
function MAK.svd_compact!(t::AdjointTensorMap, USVᴴ, alg::DiagonalAlgorithm)
    return MAK.svd_compact!(t, USVᴴ, alg.alg)
end
