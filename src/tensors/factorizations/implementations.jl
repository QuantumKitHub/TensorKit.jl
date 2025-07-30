_kindof(::Union{SVD,SDD}) = :svd
_kindof(::Union{QR,QRpos}) = :qr
_kindof(::Union{LQ,LQpos}) = :lq
_kindof(::Polar) = :polar

for f! in (:svd_compact!, :svd_full!, :left_null_svd!, :right_null_svd!)
    @eval function select_algorithm(::typeof($f!), t::T, alg::SVD;
                                    kwargs...) where {T<:AbstractTensorMap}
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed"))
        return LAPACK_QRIteration()
    end
    @eval function select_algorithm(::typeof($f!), t::AbstractTensorMap, alg::SVD;
                                    kwargs...)
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed"))
        return LAPACK_QRIteration()
    end
    @eval function select_algorithm(::typeof($f!), ::Type{T}, alg::SVD;
                                    kwargs...) where {T<:AbstractTensorMap}
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed"))
        return LAPACK_QRIteration()
    end
    @eval function select_algorithm(::typeof($f!), t::T, alg::SDD;
                                    kwargs...) where {T}
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed"))
        return LAPACK_DivideAndConquer()
    end
    @eval function select_algorithm(::typeof($f!), t::AbstractTensorMap, alg::SDD;
                                    kwargs...)
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed"))
        return LAPACK_DivideAndConquer()
    end
    @eval function select_algorithm(::typeof($f!), ::Type{T}, alg::SDD;
                                    kwargs...) where {T<:AbstractTensorMap}
        isempty(kwargs) ||
            throw(ArgumentError("Additional keyword arguments are not allowed"))
        return LAPACK_DivideAndConquer()
    end
end

leftorth!(t::AbstractTensorMap; alg=nothing, kwargs...) = _leftorth!(t, alg; kwargs...)

function _leftorth!(t::AbstractTensorMap, ::Nothing; kwargs...)
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
function _leftorth!(t, alg::OFA; kwargs...)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    kind = _kindof(alg)
    if kind == :svd
        return left_orth!(t; kind, alg_svd=alg, trunc)
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
_leftorth!(t, alg; kwargs...) = left_orth!(t; alg, kwargs...)

function leftnull!(t::AbstractTensorMap;
                   alg::Union{QR,QRpos,SVD,SDD,Nothing}=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftnull!)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    isnothing(alg) && return left_null!(t; trunc)

    kind = _kindof(alg)
    if kind == :svd
        return left_null!(t; kind, alg_svd=alg, trunc)
    elseif kind == :qr
        alg_qr = (; positive=(alg == QRpos()))
        return left_null!(t; kind, alg_qr, trunc)
    else
        throw(ArgumentError(lazy"Invalid `leftnull!` algorithm: $alg"))
    end
end

leftpolar!(t::AbstractTensorMap; kwargs...) = left_polar!(t; kwargs...)

function rightorth!(t::AbstractTensorMap;
                    alg::Union{LQ,LQpos,RQ,RQpos,SVD,SDD,Polar,Nothing}=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightorth!)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

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
        return right_orth!(t; kind, alg_svd=alg, trunc)
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
                    alg::Union{LQ,LQpos,SVD,SDD,Nothing}=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightnull!)
    trunc = isempty(kwargs) ? nothing : (; kwargs...)

    isnothing(alg) && return right_null!(t; trunc)

    kind = _kindof(alg)
    if kind == :svd
        return right_null!(t; kind, alg_svd=alg, trunc)
    elseif kind == :lq
        alg_lq = (; positive=(alg == LQpos()))
        return right_null!(t; kind, alg_lq, trunc)
    else
        throw(ArgumentError(lazy"Invalid `rightnull!` algorithm: $alg"))
    end
end

rightpolar!(t::AbstractTensorMap; kwargs...) = right_polar!(t; kwargs...)

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
function tsvd!(t::AbstractTensorMap; trunc=notrunc(), p=nothing, kwargs...)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    isnothing(p) || Base.depwarn("p is no longer supported", :tsvd!)

    if trunc == notrunc()
        return svd_compact!(t; kwargs...)
    else
        return svd_trunc!(t; trunc, kwargs...)
    end
end
