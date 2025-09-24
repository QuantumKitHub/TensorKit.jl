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
# orthogonalization
@deprecate(leftorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           leftorth!(permutedcopy_oftype(t, factorization_scalartype(leftorth, t), p);
                     kwargs...))
@deprecate(rightorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           rightorth!(permutedcopy_oftype(t, factorisation_scalartype(rightorth, t), p);
                      kwargs...))
function leftorth(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftorth` is no longer supported, use `left_orth` instead", :leftorth)
    return left_orth(t; kwargs...)
end
function leftorth!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftorth!` is no longer supported, use `left_orth!` instead", :leftorth!)
    return left_orth!(t; kwargs...)
end
function rightorth(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightorth` is no longer supported, use `right_orth` instead", :rightorth)
    return right_orth(t; kwargs...)
end
function rightorth!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightorth!` is no longer supported, use `right_orth!` instead",
                 :rightorth!)
    return right_orth!(t; kwargs...)
end

# nullspaces
@deprecate(leftnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           leftnull!(permutedcopy_oftype(t, factorization_scalartype(leftnull, t), p);
                     kwargs...))
@deprecate(rightnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           rightnull!(permutedcopy_oftype(t, factorisation_scalartype(rightnull, t), p);
                      kwargs...))
function leftnull(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftnull` is no longer supported, use `left_null` instead", :leftnull)
    return left_null(t; kwargs...)
end
function leftnull!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`left_null!` is no longer supported, use `left_null!` instead",
                 :leftnull!)
    return left_null!(t; kwargs...)
end
function rightnull(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightnull` is no longer supported, use `right_null` instead", :rightnull)
    return right_null(t; kwargs...)
end
function rightnull!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightnull!` is no longer supported, use `right_null!` instead",
                 :rightnull!)
    return right_null!(t; kwargs...)
end

# eigen values
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
_drop_p(; p=nothing, kwargs...) = kwargs
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
