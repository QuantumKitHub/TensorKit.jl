import Base: transpose

#! format: off

for f in (:rand, :randn, :zeros, :ones)
    @eval begin
        Base.@deprecate TensorMap(::typeof($f), T::Type, P::HomSpace) $f(T, P)
        Base.@deprecate TensorMap(::typeof($f), P::HomSpace) $f(P)
        Base.@deprecate TensorMap(::typeof($f), T::Type, cod::TensorSpace, dom::TensorSpace) $f(T, cod, dom)
        Base.@deprecate TensorMap(::typeof($f), cod::TensorSpace, dom::TensorSpace) $f(cod, dom)
        Base.@deprecate Tensor(::typeof($f), T::Type, space::TensorSpace) $f(T, space)
        Base.@deprecate Tensor(::typeof($f), space::TensorSpace) $f(space)
    end
end

Base.@deprecate(randuniform(dims::Base.Dims), rand(dims))
Base.@deprecate(randuniform(T::Type{<:Number}, dims::Base.Dims), rand(T, dims))
Base.@deprecate(randnormal(dims::Base.Dims), randn(dims))
Base.@deprecate(randnormal(T::Type{<:Number}, dims::Base.Dims), randn(T, dims))
Base.@deprecate(randhaar(dims::Base.Dims), randisometry(dims))
Base.@deprecate(randhaar(T::Type{<:Number}, dims::Base.Dims), randisometry(T, dims))

for (f1, f2) in ((:randuniform, :rand), (:randnormal, :randn), (:randisometry, :randisometry), (:randhaar, :randisometry))
    @eval begin
        Base.@deprecate TensorMap(::typeof($f1), T::Type, P::HomSpace) $f2(T, P)
        Base.@deprecate TensorMap(::typeof($f1), P::HomSpace) $f2(P)
        Base.@deprecate TensorMap(::typeof($f1), T::Type, cod::TensorSpace, dom::TensorSpace) $f2(T, P, cod, dom)
        Base.@deprecate TensorMap(::typeof($f1), cod::TensorSpace, dom::TensorSpace) $f2(cod, dom)
        Base.@deprecate Tensor(::typeof($f1), T::Type, space::TensorSpace) $f2(T, space)
        Base.@deprecate Tensor(::typeof($f1), space::TensorSpace) $f2(space)
    end
end

Base.@deprecate EuclideanProduct() EuclideanInnerProduct()

Base.@deprecate insertunit(P::ProductSpace, args...; kwargs...) insertleftunit(args...; kwargs...)

# Factorization structs
@deprecate QR() MatrixAlgebraKit.LAPACK_HouseholderQR()
@deprecate QRpos() MatrixAlgebraKit.LAPACK_HouseholderQR(; positive=true)

@deprecate QL() MatrixAlgebraKit.LAPACK_HouseholderQL()
@deprecate QLpos() MatrixAlgebraKit.LAPACK_HouseholderQL(; positive=true)

@deprecate LQ() MatrixAlgebraKit.LAPACK_HouseholderLQ()
@deprecate LQpos() MatrixAlgebraKit.LAPACK_HouseholderLQ(; positive=true)

@deprecate RQ() MatrixAlgebraKit.LAPACK_HouseholderRQ()
@deprecate RQpos() MatrixAlgebraKit.LAPACK_HouseholderRQ(; positive=true)

@deprecate SDD() MatrixAlgebraKit.LAPACK_DivideAndConquer()
@deprecate SVD() MatrixAlgebraKit.LAPACK_QRIteration()

@deprecate Polar() MatrixAlgebraKit.PolarViaSVD(MatrixAlgebraKit.LAPACK_DivideAndConquer())

# truncations
const TruncationScheme = MatrixAlgebraKit.TruncationStrategy
@deprecate truncdim(d::Int) truncrank(d)
@deprecate truncbelow(ϵ::Real) trunctol(ϵ)

# factorizations
# --------------
_kindof(::MatrixAlgebraKit.LAPACK_HouseholderQR) = :qr
_kindof(::MatrixAlgebraKit.LAPACK_HouseholderLQ) = :lq
_kindof(::MatrixAlgebraKit.LAPACK_SVDAlgorithm) = :svd
_kindof(::MatrixAlgebraKit.PolarViaSVD) = :polar
_kindof(::DiagonalAlgorithm) = :svd # shouldn't really matter

_drop_alg(; alg=nothing, kwargs...) = kwargs
_drop_p(; p=nothing, kwargs...) = kwargs

function permutedcopy_oftype(t::AbstractTensorMap, T::Type{<:Number}, p::Index2Tuple)
    return permute!(similar(t, T, permute(space(t), p)), t, p)
end

# orthogonalization
export leftorth, leftorth!, rightorth, rightorth!
function leftorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`leftorth` is deprecated, use `left_orth` instead", :leftorth)
    return leftorth!(permutedcopy_oftype(t, factorisation_scalartype(leftorth, t), p); kwargs...)
end
function rightorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`rightorth` is deprecated, use `right_orth` instead", :rightorth)
    return rightorth!(permutedcopy_oftype(t, factorisation_scalartype(rightorth, t), p); kwargs...)
end
function leftorth(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftorth` is deprecated, use `left_orth` instead", :leftorth)
    return leftorth!(copy_oftype(t, factorisation_scalartype(leftorth, t)); kwargs...)
end
function rightorth(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightorth` is deprecated, use `right_orth` instead", :rightorth)
    return rightorth!(copy_oftype(t, factorisation_scalartype(rightorth, t)); kwargs...)
end
function leftorth!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`leftorth!` is deprecated, use `left_orth!` instead", :leftorth!)
    haskey(kwargs, :alg) || return left_orth!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return left_orth!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :qr && return left_orth!(t; kind, alg_qr=alg, _drop_alg(; kwargs...)...)
    kind === :polar && return left_orth!(t; kind, alg_polar=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid leftorth kind"))
end
function rightorth!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightorth!` is deprecated, use `right_orth!` instead", :rightorth!)
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
    Base.depwarn("`leftnull` is deprecated, use `left_null` instead", :leftnull)
    return leftnull!(copy_oftype(t, factorisation_scalartype(leftnull, t)); kwargs...)
end
function leftnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`leftnull` is deprecated, use `left_null` instead", :leftnull)
    return leftnull!(permutedcopy_oftype(t, factorisation_scalartype(leftnull, t), p); kwargs...)
end
function rightnull(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightnull` is deprecated, use `right_null` instead", :rightnull)
    return rightnull!(copy_oftype(t, factorisation_scalartype(rightnull, t)); kwargs...)
end
function rightnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    Base.depwarn("`rightnull` is deprecated, use `right_null` instead", :rightnull)
    return rightnull!(permutedcopy_oftype(t, factorisation_scalartype(rightnull, t), p); kwargs...)
end
function leftnull!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`left_null!` is deprecated, use `left_null!` instead", :leftnull!)
    haskey(kwargs, :alg) || return left_null!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return left_null!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :qr && return left_null!(t; kind, alg_qr=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid leftnull kind"))
end
function rightnull!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`rightnull!` is deprecated, use `right_null!` instead", :rightnull!)
    haskey(kwargs, :alg) || return right_null!(t; kwargs...)
    alg = kwargs[:alg]
    kind = _kindof(alg)
    kind === :svd && return right_null!(t; kind, alg_svd=alg, _drop_alg(; kwargs...)...)
    kind === :lq && return right_null!(t; kind, alg_lq=alg, _drop_alg(; kwargs...)...)
    throw(ArgumentError("invalid rightnull kind"))
end

# eigen values
export eig!, eigh!, eigen, eigen!
@deprecate(eig(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           eig!(permutedcopy_oftype(t, factorisation_scalartype(eig, t), p); kwargs...))
@deprecate(eigh(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           eigh!(permutedcopy_oftype(t, factorisation_scalartype(eigen, t), p); kwargs...))
@deprecate(LinearAlgebra.eigen(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           eigen!(permutedcopy_oftype(t, factorisation_scalartype(eigen, t), p); kwargs...),
           false)
function eig(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eig` is deprecated, use `eig_full` or `eig_trunc` instead", :eig)
    return haskey(kwargs, :trunc) ? eig_trunc(t; kwargs...) : eig_full(t; kwargs...)
end
function eig!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eig!` is deprecated, use `eig_full!` or `eig_trunc!` instead", :eig!)
    return haskey(kwargs, :trunc) ? eig_trunc!(t; kwargs...) : eig_full!(t; kwargs...)
end
function eigh(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eigh` is deprecated, use `eigh_full` or `eigh_trunc` instead", :eigh)
    return haskey(kwargs, :trunc) ? eigh_trunc(t; kwargs...) : eigh_full(t; kwargs...)
end
function eigh!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`eigh!` is deprecated, use `eigh_full!` or `eigh_trunc!` instead", :eigh!)
    return haskey(kwargs, :trunc) ? eigh_trunc!(t; kwargs...) : eigh_full!(t; kwargs...)
end

# singular values
export tsvd, tsvd!
@deprecate(tsvd(t::AbstractTensorMap, p::Index2Tuple; kwargs...),
           tsvd!(permutedcopy_oftype(t, factorisation_scalartype(tsvd, t), p); kwargs...))
function tsvd(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`tsvd` is deprecated, use `svd_compact`, `svd_full` or `svd_trunc` instead", :tsvd)
    if haskey(kwargs, :p)
        Base.depwarn("p is a deprecated kwarg, and should be specified through the truncation strategy", :tsvd)
        kwargs = _drop_p(; kwargs...)
    end
    return haskey(kwargs, :trunc) ? svd_trunc(t; kwargs...) : svd_compact(t; kwargs...)
end
function tsvd!(t::AbstractTensorMap; kwargs...)
    Base.depwarn("`tsvd!` is deprecated, use `svd_compact!`, `svd_full!` or `svd_trunc!` instead", :tsvd!)
    if haskey(kwargs, :p)
        Base.depwarn("p is a deprecated kwarg, and should be specified through the truncation strategy", :tsvd!)
        kwargs = _drop_p(; kwargs...)
    end
    return haskey(kwargs, :trunc) ? svd_trunc!(t; kwargs...) : svd_compact!(t; kwargs...)
end

#! format: on
