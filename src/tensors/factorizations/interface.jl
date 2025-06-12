@doc """
    tsvd(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple];
         trunc::TruncationScheme = notrunc(), p::Real = 2, alg::Union{SVD, SDD} = SDD())
        -> U, S, V, ϵ
    tsvd!(t::AbstractTensorMap, trunc::TruncationScheme = notrunc(), p::Real = 2, alg::Union{SVD, SDD} = SDD())
        -> U, S, V, ϵ

Compute the (possibly truncated) singular value decomposition such that
`norm(permute(t, (leftind, rightind)) - U * S * V) ≈ ϵ`, where `ϵ` thus represents the truncation error.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `tsvd!(t, trunc = notrunc(), p = 2)`.

A truncation parameter `trunc` can be specified for the new internal dimension, in which
case a truncated singular value decomposition will be computed. Choices are:
*   `notrunc()`: no truncation (default);
*   `truncerr(η::Real)`: truncates such that the p-norm of the truncated singular values is
    smaller than `η`;
*   `truncdim(χ::Int)`: truncates such that the equivalent total dimension of the internal
    vector space is no larger than `χ`;
*   `truncspace(V)`: truncates such that the dimension of the internal vector space is
    smaller than that of `V` in any sector.
*   `truncbelow(η::Real)`: truncates such that every singular value is larger then `η` ;

Truncation options can also be combined using `&`, i.e. `truncbelow(η) & truncdim(χ)` will
choose the truncation space such that every singular value is larger than `η`, and the
equivalent total dimension of the internal vector space is no larger than `χ`.

The method `tsvd` also returns the truncation error `ϵ`, computed as the `p` norm of the
singular values that were truncated.

THe keyword `alg` can be equal to `SVD()` or `SDD()`, corresponding to the underlying LAPACK
algorithm that computes the decomposition (`_gesvd` or `_gesdd`).

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and `tsvd(!)`
is currently only implemented for `InnerProductStyle(t) === EuclideanInnerProduct()`.
""" tsvd, tsvd!

@doc """
    eig(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple]; kwargs...) -> D, V
    eig!(t::AbstractTensorMap; kwargs...) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.
The function `eig` assumes that the linear map is not hermitian and returns type stable
complex valued `D` and `V` tensors for both real and complex valued `t`. See `eigh` for
hermitian linear maps

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `eig!(t)`. Note that the permuted tensor on
which `eig!` is called should have equal domain and codomain, as otherwise the eigenvalue
decomposition is meaningless and cannot satisfy
```
permute(t, (leftind, rightind)) * V = V * D
```

Accepts the same keyword arguments `scale` and `permute` as `eigen` of dense
matrices. See the corresponding documentation for more information.

See also `eigen` and `eigh`.
""" eig

@doc """
    eigh(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple]; kwargs...) -> D, V
    eigh!(t::AbstractTensorMap; kwargs...) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.
The function `eigh` assumes that the linear map is hermitian and `D` and `V` tensors with
the same `scalartype` as `t`. See `eig` and `eigen` for non-hermitian tensors. Hermiticity
requires that the tensor acts on inner product spaces, and the current implementation
requires `InnerProductStyle(t) === EuclideanInnerProduct()`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `eigh!(t)`. Note that the permuted tensor on
which `eigh!` is called should have equal domain and codomain, as otherwise the eigenvalue
decomposition is meaningless and cannot satisfy
```
permute(t, (leftind, rightind)) * V = V * D
```

See also `eigen` and `eig`.
""" eigh, eigh!

@doc """
    leftorth(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = QRpos()) -> Q, R

Create orthonormal basis `Q` for indices in `leftind`, and remainder `R` such that
`permute(t, (leftind, rightind)) = Q*R`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `leftorth!(t, alg = QRpos())`.

Different algorithms are available, namely `QR()`, `QRpos()`, `SVD()` and `Polar()`. `QR()`
and `QRpos()` use a standard QR decomposition, producing an upper triangular matrix `R`.
`Polar()` produces a Hermitian and positive semidefinite `R`. `QRpos()` corrects the
standard QR decomposition such that the diagonal elements of `R` are positive. Only
`QRpos()` and `Polar()` are unique (no residual freedom) so that they always return the same
result for the same input tensor `t`.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`leftorth(!)` is currently only implemented for 
    `InnerProductStyle(t) === EuclideanInnerProduct()`.
""" leftorth, leftorth!

@doc """
    rightorth(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple];
                alg::OrthogonalFactorizationAlgorithm = LQpos()) -> L, Q
    rightorth!(t::AbstractTensorMap; alg) -> L, Q

Create orthonormal basis `Q` for indices in `rightind`, and remainder `L` such that
`permute(t, (leftind, rightind)) = L*Q`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `rightorth!(t, alg = LQpos())`.

Different algorithms are available, namely `LQ()`, `LQpos()`, `RQ()`, `RQpos()`, `SVD()` and
`Polar()`. `LQ()` and `LQpos()` produce a lower triangular matrix `L` and are computed using
a QR decomposition of the transpose. `RQ()` and `RQpos()` produce an upper triangular
remainder `L` and only works if the total left dimension is smaller than or equal to the
total right dimension. `LQpos()` and `RQpos()` add an additional correction such that the
diagonal elements of `L` are positive. `Polar()` produces a Hermitian and positive
semidefinite `L`. Only `LQpos()`, `RQpos()` and `Polar()` are unique (no residual freedom)
so that they always return the same result for the same input tensor `t`.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`rightorth(!)` is currently only implemented for 
`InnerProductStyle(t) === EuclideanInnerProduct()`.
""" rightorth, rightorth!

@doc """
    leftnull(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple];
                alg::OrthogonalFactorizationAlgorithm = QRpos()) -> N
    leftnull!(t::AbstractTensorMap; alg) -> N

Create orthonormal basis for the orthogonal complement of the support of the indices in
`leftind`, such that `N' * permute(t, (leftind, rightind)) = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `leftnull!(t, alg = QRpos())`.

Different algorithms are available, namely `QR()` (or equivalently, `QRpos()`), `SVD()` and
`SDD()`. The first assumes that the matrix is full rank and requires `iszero(atol)` and
`iszero(rtol)`. With `SVD()` and `SDD()`, `rightnull` will use the corresponding singular
value decomposition, and one can specify an absolute or relative tolerance for which
singular values are to be considered zero, where `max(atol, norm(t)*rtol)` is used as upper
bound.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`leftnull(!)` is currently only implemented for 
`InnerProductStyle(t) === EuclideanInnerProduct()`.
""" leftnull, leftnull!

@doc """
    rightnull(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple];
                alg::OrthogonalFactorizationAlgorithm = LQ(),
                atol::Real = 0.0,
                rtol::Real = eps(real(float(one(scalartype(t)))))*iszero(atol)) -> N
    rightnull!(t::AbstractTensorMap; alg, atol, rtol)

Create orthonormal basis for the orthogonal complement of the support of the indices in
`rightind`, such that `permute(t, (leftind, rightind))*N' = 0`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `rightnull!(t, alg = LQpos())`.

Different algorithms are available, namely `LQ()` (or equivalently, `LQpos`), `SVD()` and
`SDD()`. The first assumes that the matrix is full rank and requires `iszero(atol)` and
`iszero(rtol)`. With `SVD()` and `SDD()`, `rightnull` will use the corresponding singular
value decomposition, and one can specify an absolute or relative tolerance for which
singular values are to be considered zero, where `max(atol, norm(t)*rtol)` is used as upper
bound.

Orthogonality requires `InnerProductStyle(t) <: HasInnerProduct`, and
`rightnull(!)` is currently only implemented for 
`InnerProductStyle(t) === EuclideanInnerProduct()`.
""" rightnull, rightnull!

@doc """
    leftpolar(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple]; kwargs...) -> W, P
    leftpolar!(t::AbstractTensorMap; kwargs...) -> W, P

Compute the polar decomposition of tensor `t` as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `eigh!(t)`.

See also [`rightpolar(!)`](@ref rightpolar).

""" leftpolar, leftpolar!

@doc """
    eigen(t::AbstractTensorMap, [(leftind, rightind)::Index2Tuple]; kwargs...) -> D, V
    eigen!(t::AbstractTensorMap; kwargs...) -> D, V

Compute eigenvalue factorization of tensor `t` as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in `t`
to be destroyed/overwritten, by using `eigen!(t)`. Note that the permuted tensor on which
`eigen!` is called should have equal domain and codomain, as otherwise the eigenvalue
decomposition is meaningless and cannot satisfy
```
permute(t, (leftind, rightind)) * V = V * D
```

Accepts the same keyword arguments `scale` and `permute` as `eigen` of dense
matrices. See the corresponding documentation for more information.

See also [`eig(!)`](@ref eig) and [`eigh(!)`](@ref)
""" eigen(::AbstractTensorMap), eigen!(::AbstractTensorMap)

@doc """
    isposdef(t::AbstractTensor, [(leftind, rightind)::Index2Tuple]) -> ::Bool

Test whether a tensor `t` is positive definite as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `isposdef!(t)`. Note that the permuted tensor on
which `isposdef!` is called should have equal domain and codomain, as otherwise it is
meaningless.
""" isposdef(::AbstractTensorMap), isposdef!(::AbstractTensorMap)

for f in
    (:tsvd, :eig, :eigh, :eigen, :leftorth, :rightorth, :leftpolar, :rightpolar, :leftnull,
     :rightnull, :isposdef)
    f! = Symbol(f, :!)
    @eval function $f(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
        tcopy = permutedcopy_oftype(t, factorisation_scalartype($f, t), p)
        return $f!(tcopy; kwargs...)
    end
    @eval function $f(t::AbstractTensorMap; kwargs...)
        tcopy = copy_oftype(t, factorisation_scalartype($f, t))
        return $f!(tcopy; kwargs...)
    end
end

function LinearAlgebra.eigvals(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, factorisation_scalartype(eigen, t))
    return LinearAlgebra.eigvals!(tcopy; kwargs...)
end

function LinearAlgebra.svdvals(t::AbstractTensorMap)
    tcopy = copy_oftype(t, factorisation_scalartype(tsvd, t))
    return LinearAlgebra.svdvals!(tcopy)
end
