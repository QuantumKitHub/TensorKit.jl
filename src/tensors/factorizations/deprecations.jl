# Factorization structs
@deprecate QR() LAPACK_HouseholderQR()
@deprecate QRpos() LAPACK_HouseholderQR(; positive=true)

@deprecate QL() LAPACK_HouseholderQL()
@deprecate QLpos() LAPACK_HouseholderQL(; positive=true)

@deprecate LQ() LAPACK_HouseholderLQ()
@deprecate LQpos() LAPACK_HouseholderLQ(; positive=true)

@deprecate RQ() LAPACK_HouseholderRQ()
@deprecate RQpos() LAPACK_HouseholderRQ(; positive=true)

@deprecate SDD() LAPACK_DivideAndConquer()
@deprecate SVD() LAPACK_QRIteration()

@deprecate Polar() PolarViaSVD(LAPACK_DivideAndConquer())

# truncations
const TruncationScheme = TruncationStrategy
@deprecate truncdim(d::Int) truncrank(d)
@deprecate truncbelow(ϵ::Real) trunctol(ϵ)

# factorizations
# --------------
_kindof(::LAPACK_HouseholderQR) = :qr
_kindof(::LAPACK_HouseholderLQ) = :lq
_kindof(::LAPACK_SVDAlgorithm) = :svd
_kindof(::PolarViaSVD) = :polar

_drop_alg(; alg=nothing, kwargs...) = kwargs
_drop_p(; p=nothing, kwargs...) = kwargs

# orthogonalization
export leftorth, leftorth!, rightorth, rightorth!
function leftorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`leftorth` is no longer supported, use `left_orth` instead", :leftorth)
    return leftorth!(permutedcopy_oftype(t, factorisation_scalartype(leftorth, t), p);
                     kwargs...)
end
function rightorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`rightorth` is no longer supported, use `right_orth` instead", :rightorth)
    return rightorth!(permutedcopy_oftype(t, factorisation_scalartype(rightorth, t), p);
                      kwargs...)
end
function leftorth(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftorth` is no longer supported, use `left_orth` instead", :leftorth)
    return leftorth!(copy_oftype(t, factorisation_scalartype(leftorth, t)); kwargs...)
end
function rightorth(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightorth` is no longer supported, use `right_orth` instead", :rightorth)
    return rightorth!(copy_oftype(t, factorisation_scalartype(rightorth, t)); kwargs...)
end
function leftorth!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftorth!` is no longer supported, use `left_orth!` instead", :leftorth!)
    haskey(kwargs, :alg) || return left_orth!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return left_orth!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :qr && return left_orth!(t; kind, alg_qr=alg, _drop_alg(; kwargs...)...)
    kind === :polar && return left_orth!(t; kind, alg_polar=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid leftorth kind"))
end
function rightorth!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightorth!` is no longer supported, use `right_orth!` instead",
                 :rightorth!)
    haskey(kwargs, :alg) || return right_orth!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return right_orth!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :lq && return right_orth!(t; kind, alg_lq=alg, _drop_alg(; kwargs...)...)
    kind === :polar && return right_orth!(t; kind, alg_polar=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid rightorth kind"))
end

# nullspaces
export leftnull, leftnull!, rightnull, rightnull!
function leftnull(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftnull` is no longer supported, use `left_null` instead", :leftnull)
    return leftnull!(copy_oftype(t, factorisation_scalartype(leftnull, t)); kwargs...)
end
function leftnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`leftnull` is no longer supported, use `left_null` instead", :leftnull)
    return leftnull!(permutedcopy_oftype(t, factorisation_scalartype(leftnull, t), p); kwargs...)
end
function rightnull(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightnull` is no longer supported, use `right_null` instead", :rightnull)
    return rightnull!(copy_oftype(t, factorisation_scalartype(rightnull, t)); kwargs...)
end
function rightnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`rightnull` is no longer supported, use `right_null` instead", :rightnull)
    return rightnull!(permutedcopy_oftype(t, factorisation_scalartype(rightnull, t), p); kwargs...)
end
function leftnull!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`left_null!` is no longer supported, use `left_null!` instead",
                 :leftnull!)
    haskey(kwargs, :alg) || return left_null!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return left_null!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :qr && return left_null!(t; kind, alg_qr=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid leftnull kind"))
end
function rightnull!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightnull!` is no longer supported, use `right_null!` instead",
                 :rightnull!)
    haskey(kwargs, :alg) || return right_null!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return right_null!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :lq && return right_null!(t; kind, alg_lq=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid rightnull kind"))
end

# eigen values
export eig, eig!, eigh, eigh!, eigen, eigen!
@deprecate(eig(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           eig!(permutedcopy_oftype(t, factorisation_scalartype(eig, t), p); kwargs...))
@deprecate(eigh(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           eigh!(permutedcopy_oftype(t, factorisation_scalartype(eigen, t), p); kwargs...))
@deprecate(eigen(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           eigen!(permutedcopy_oftype(t, factorisation_scalartype(eigen, t), p); kwargs...))
function eig(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eig` is no longer supported, use `eig_full` or `eig_trunc` instead",
                 :eig)
    return haskey(kwargs, :trunc) ? eig_trunc(t; kwargs...) : eig_full(t; kwargs...)
end
function eig!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eig!` is no longer supported, use `eig_full!` or `eig_trunc!` instead",
                 :eig!)
    return haskey(kwargs, :trunc) ? eig_trunc!(t; kwargs...) : eig_full!(t; kwargs...)
end
function eigh(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eigh` is no longer supported, use `eigh_full` or `eigh_trunc` instead",
                 :eigh)
    return haskey(kwargs, :trunc) ? eigh_trunc(t; kwargs...) : eigh_full(t; kwargs...)
end
function eigh!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eigh!` is no longer supported, use `eigh_full!` or `eigh_trunc!` instead",
                 :eigh!)
    return haskey(kwargs, :trunc) ? eigh_trunc!(t; kwargs...) : eigh_full!(t; kwargs...)
end

# singular values
export tsvd, tsvd!
@deprecate(tsvd(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           tsvd!(permutedcopy_oftype(t, factorisation_scalartype(tsvd, t), p); kwargs...))
function tsvd(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`tsvd` is no longer supported, use `svd_compact`, `svd_full` or `svd_trunc` instead",
                 :tsvd)
    if haskey(kwargs, :p)
        Base.depwarn("p is no longer a supported kwarg, and should be specified through the truncation strategy",
                     :tsvd)
        kwargs = _drop_p(; kwargs...)
    end
    return haskey(kwargs, :trunc) ? svd_trunc(t; kwargs...) : svd_compact(t; kwargs...)
end
function tsvd!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`tsvd!` is no longer supported, use `svd_compact!`, `svd_full!` or `svd_trunc!` instead",
                 :tsvd!)
    if haskey(kwargs, :p)
        Base.depwarn("p is no longer a supported kwarg, and should be specified through the truncation strategy",
                     :tsvd!)
        kwargs = _drop_p(; kwargs...)
    end
    return haskey(kwargs, :trunc) ? svd_trunc!(t; kwargs...) : svd_compact!(t; kwargs...)
end
