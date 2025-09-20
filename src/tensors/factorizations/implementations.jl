_kindof(::Union{SVD,SDD}) = :svd
_kindof(::Union{QR,QRpos}) = :qr
_kindof(::Union{LQ,LQpos}) = :lq
_kindof(::Polar) = :polar

_kindof(::LAPACK_HouseholderQR) = :qr
_kindof(::LAPACK_HouseholderLQ) = :lq
_kindof(::LAPACK_SVDAlgorithm) = :svd
_kindof(::PolarViaSVD) = :polar
_kindof(::DiagonalAlgorithm) = :svd

leftorth!(t; alg=nothing, kwargs...) = _leftorth!(t, alg; kwargs...)

function _leftorth!(t::AbstractTensorMap, alg::Nothing, ; kwargs...)
    return isempty(kwargs) ? left_orth!(t) : left_orth!(t; trunc=(; kwargs...))
end
function _leftorth!(t::AbstractTensorMap, alg::Union{QL,QLpos}; kwargs...)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    if alg == QL() || alg == QLpos()
        _reverse!(t; dims=2)
        Q, R = left_orth!(t; kind=:qr, alg_qr=(; positive=alg == QLpos()), trunc)
        _reverse!(Q; dims=2)
        _reverse!(R)
        return Q, R
    end
end
function _leftorth!(t, alg::Union{OFA,AbstractAlgorithm}; kwargs...)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    alg isa OFA && Base.depwarn(lazy"$alg is deprecated", :leftorth!)

    kind = _kindof(alg)
    if kind == :svd
        alg_svd = alg === LAPACK_QRIteration() ? alg :
                  alg === LAPACK_DivideAndConquer() ? alg :
                  alg === DiagonalAlgorithm() ? alg :
                  alg === SVD() ? LAPACK_QRIteration() :
                  alg === SDD() ? LAPACK_DivideAndConquer() :
                  throw(ArgumentError(lazy"Unknown algorithm $alg"))
        return left_orth!(t; kind, alg_svd, trunc)
    elseif kind == :qr
        alg_qr = (; positive=(alg == QRpos()))
        return left_orth!(t; kind, alg_qr, trunc)
    elseif kind == :polar
        return left_orth!(t; kind, trunc)
    else
        throw(ArgumentError(lazy"Invalid algorithm: $alg"))
    end
end
# fallback to MatrixAlgebraKit version
_leftorth!(t, alg; kwargs...) = left_orth!(t, alg; kwargs...)

function leftnull!(t::AbstractTensorMap;
                   alg::Union{LAPACK_HouseholderQR,LAPACK_QRIteration,
                              LAPACK_DivideAndConquer,PolarViaSVD,QR,QRpos,SVD,SDD,Nothing}=nothing,
                   kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftnull!)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)
    alg isa OFA && Base.depwarn(lazy"$alg is deprecated", :leftnull!)

    isnothing(alg) && return left_null!(t; trunc)

    kind = _kindof(alg)
    if kind == :svd
        alg_svd = alg === LAPACK_QRIteration() ? alg :
                  alg === LAPACK_DivideAndConquer() ? alg :
                  alg === SVD() ? LAPACK_QRIteration() :
                  alg === SDD() ? LAPACK_DivideAndConquer() :
                  throw(ArgumentError(lazy"Unknown algorithm $alg"))
        return left_null!(t; kind, alg_svd, trunc)
    elseif kind == :qr
        alg_qr = (; positive=(alg == QRpos()))
        return left_null!(t; kind, alg_qr, trunc)
    else
        throw(ArgumentError(lazy"Invalid `leftnull!` algorithm: $alg"))
    end
end

function rightorth!(t::AbstractTensorMap;
                    alg::Union{LAPACK_HouseholderLQ,LAPACK_QRIteration,
                               LAPACK_DivideAndConquer,DiagonalAlgorithm,PolarViaSVD,LQ,
                               LQpos,RQ,RQpos,SVD,
                               SDD,Polar,Nothing}=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightorth!)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    alg isa OFA && Base.depwarn(lazy"$alg is deprecated", :rightorth!)

    isnothing(alg) && return right_orth!(t; trunc)

    if alg == RQ() || alg == RQpos()
        _reverse!(t; dims=1)
        L, Q = right_orth!(t; kind=:lq, alg_lq=(; positive=alg == RQpos()), trunc)
        _reverse!(Q; dims=1)
        _reverse!(L)
        return L, Q
    end

    kind = _kindof(alg)
    if kind == :svd
        alg_svd = alg === LAPACK_QRIteration() ? alg :
                  alg === LAPACK_DivideAndConquer() ? alg :
                  alg === DiagonalAlgorithm() ? alg :
                  alg === SVD() ? LAPACK_QRIteration() :
                  alg === SDD() ? LAPACK_DivideAndConquer() :
                  throw(ArgumentError(lazy"Unknown algorithm $alg"))
        return right_orth!(t; kind, alg_svd, trunc)
    elseif kind == :lq
        alg_lq = (; positive=(alg == LQpos()))
        return right_orth!(t; kind, alg_lq, trunc)
    elseif kind == :polar
        return right_orth!(t; kind, trunc)
    else
        throw(ArgumentError(lazy"Invalid `rightorth!` algorithm: $alg"))
    end
end

function rightnull!(t::AbstractTensorMap;
                    alg::Union{LAPACK_HouseholderLQ,LAPACK_QRIteration,
                               LAPACK_DivideAndConquer,PolarViaSVD,LQ,LQpos,SVD,SDD,
                               Nothing}=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightnull!)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    alg isa OFA && Base.depwarn(lazy"$alg is deprecated", :rightnull!)

    isnothing(alg) && return right_null!(t; trunc)

    kind = _kindof(alg)
    if kind == :svd
        alg_svd = alg === LAPACK_QRIteration() ? alg :
                  alg === LAPACK_DivideAndConquer() ? alg :
                  alg === SVD() ? LAPACK_QRIteration() :
                  alg === SDD() ? LAPACK_DivideAndConquer() :
                  throw(ArgumentError(lazy"Unknown algorithm $alg"))
        return right_null!(t; kind, alg_svd, trunc)
    elseif kind == :lq
        alg_lq = (; positive=(alg == LQpos()))
        return right_null!(t; kind, alg_lq, trunc)
    else
        throw(ArgumentError(lazy"Invalid `rightnull!` algorithm: $alg"))
    end
end

# Eigenvalue decomposition
# ------------------------
function eigh!(t::AbstractTensorMap; trunc=notrunc(), kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:eigh!)
    if trunc == notrunc()
        return eigh_full!(t; kwargs...)
    else
        return eigh_trunc!(t; trunc, kwargs...)
    end
end

function eig!(t::AbstractTensorMap; trunc=notrunc(), kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:eig!)

    if trunc == notrunc()
        return eig_full!(t; kwargs...)
    else
        return eig_trunc!(t; trunc, kwargs...)
    end
end

function eigen!(t::AbstractTensorMap; kwargs...)
    return ishermitian(t) ? eigh!(t; kwargs...) : eig!(t; kwargs...)
end

# Singular value decomposition
# ----------------------------
function tsvd!(t::AbstractTensorMap; trunc=notrunc(), p=nothing, alg=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    isnothing(p) || Base.depwarn("p is no longer supported", :tsvd!)

    if alg isa OFA
        Base.depwarn(lazy"$alg is deprecated", :tsvd!)
        alg = alg === SVD() ? LAPACK_QRIteration() :
              alg === SDD() ? LAPACK_DivideAndConquer() :
              throw(ArgumentError(lazy"Unknown algorithm $alg"))
    end

    if trunc == notrunc()
        return svd_compact!(t; alg, kwargs...)
    else
        return svd_trunc!(t; trunc, alg, kwargs...)
    end
end
