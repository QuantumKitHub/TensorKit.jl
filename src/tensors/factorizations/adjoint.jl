# AdjointTensorMap
# ----------------
# 1-arg functions
function initialize_output(::typeof(left_null!), t::AdjointTensorMap,
                           alg::AbstractAlgorithm)
    return adjoint(initialize_output(right_null!, adjoint(t), alg))
end
function initialize_output(::typeof(right_null!), t::AdjointTensorMap,
                           alg::AbstractAlgorithm)
    return adjoint(initialize_output(left_null!, adjoint(t), alg))
end

function left_null!(t::AdjointTensorMap, N::AdjointTensorMap, alg::AbstractAlgorithm)
    right_null!(adjoint(t), adjoint(N), alg)
    return N
end
function right_null!(t::AdjointTensorMap, N::AdjointTensorMap, alg::AbstractAlgorithm)
    left_null!(adjoint(t), adjoint(N), alg)
    return N
end

# 2-arg functions
for (left_f!, right_f!) in zip((:qr_full!, :qr_compact!, :left_polar!, :left_orth!),
                               (:lq_full!, :lq_compact!, :right_polar!, :right_orth!))
    @eval function initialize_output(::typeof($left_f!), t::AdjointTensorMap,
                                     alg::AbstractAlgorithm)
        return reverse(adjoint.(initialize_output($right_f!, adjoint(t), alg)))
    end
    @eval function initialize_output(::typeof($right_f!), t::AdjointTensorMap,
                                     alg::AbstractAlgorithm)
        return reverse(adjoint.(initialize_output($left_f!, adjoint(t), alg)))
    end

    @eval function $left_f!(t::AdjointTensorMap,
                            F::Tuple{AdjointTensorMap,AdjointTensorMap},
                            alg::AbstractAlgorithm)
        $right_f!(adjoint(t), reverse(adjoint.(F)), alg)
        return F
    end
    @eval function $right_f!(t::AdjointTensorMap,
                             F::Tuple{AdjointTensorMap,AdjointTensorMap},
                             alg::AbstractAlgorithm)
        $left_f!(adjoint(t), reverse(adjoint.(F)), alg)
        return F
    end
end

# 3-arg functions
for f! in (:svd_full!, :svd_compact!, :svd_trunc!)
    @eval function initialize_output(::typeof($f!), t::AdjointTensorMap,
                                     alg::AbstractAlgorithm)
        return reverse(adjoint.(initialize_output($f!, adjoint(t), alg)))
    end
    _TS = f! === :svd_full! ? :AdjointTensorMap : DiagonalTensorMap
    @eval function $f!(t::AdjointTensorMap,
                       F::Tuple{AdjointTensorMap,$_TS,AdjointTensorMap},
                       alg::AbstractAlgorithm)
        $f!(adjoint(t), reverse(adjoint.(F)), alg)
        return F
    end
end

function leftorth!(t::AdjointTensorMap; alg::OFA=QRpos())
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftorth!)
    return map(adjoint, reverse(rightorth!(adjoint(t); alg=alg')))
end

function rightorth!(t::AdjointTensorMap; alg::OFA=LQpos())
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightorth!)
    return map(adjoint, reverse(leftorth!(adjoint(t); alg=alg')))
end

function leftnull!(t::AdjointTensorMap; alg::OFA=QR(), kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftnull!)
    return adjoint(rightnull!(adjoint(t); alg=alg', kwargs...))
end

function rightnull!(t::AdjointTensorMap; alg::OFA=LQ(), kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightnull!)
    return adjoint(leftnull!(adjoint(t); alg=alg', kwargs...))
end

function tsvd!(t::AdjointTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    u, s, vt, err = tsvd!(adjoint(t); trunc=trunc, p=p, alg=alg)
    return adjoint(vt), adjoint(s), adjoint(u), err
end
