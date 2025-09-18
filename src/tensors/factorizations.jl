# Tensor factorization
#----------------------
function factorisation_scalartype(t::AbstractTensorMap)
    T = scalartype(t)
    return promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))
end
factorisation_scalartype(f, t) = factorisation_scalartype(t)

function permutedcopy_oftype(t::AbstractTensorMap, T::Type{<:Number}, p::Index2Tuple)
    return permute!(similar(t, T, permute(space(t), p)), t, p)
end
function copy_oftype(t::AbstractTensorMap, T::Type{<:Number})
    return copy!(similar(t, T), t)
end

"""
    tsvd(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple;
        trunc::TruncationScheme = notrunc(), p::Real = 2, alg::Union{SVD, SDD} = SDD())
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
*   `truncspace(V)`: truncates such that the dimension of the internal vector space is no
    greater than that of `V` in any sector.
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
"""
function tsvd(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(tsvd, t), p)
    return tsvd!(tcopy; kwargs...)
end

function LinearAlgebra.svdvals(t::AbstractTensorMap)
    tcopy = copy_oftype(t, factorisation_scalartype(tsvd, t))
    return LinearAlgebra.svdvals!(tcopy)
end

"""
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
"""
function leftorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(leftorth, t), p)
    return leftorth!(tcopy; kwargs...)
end

"""
    rightorth(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = LQpos()) -> L, Q

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
"""
function rightorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(rightorth, t), p)
    return rightorth!(tcopy; kwargs...)
end

"""
    leftnull(t::AbstractTensor, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = QRpos()) -> N

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
"""
function leftnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(leftnull, t), p)
    return leftnull!(tcopy; kwargs...)
end

"""
    rightnull(t::AbstractTensor, (leftind, rightind)::Index2Tuple;
                alg::OrthogonalFactorizationAlgorithm = LQ(),
                atol::Real = 0.0,
                rtol::Real = eps(real(float(one(scalartype(t)))))*iszero(atol)) -> N

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
"""
function rightnull(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(rightnull, t), p)
    return rightnull!(tcopy; kwargs...)
end

"""
    eigen(t::AbstractTensor, (leftind, rightind)::Index2Tuple; kwargs...) -> D, V

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

See also `eig` and `eigh`
"""
function LinearAlgebra.eigen(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(eigen, t), p)
    return eigen!(tcopy; kwargs...)
end

function LinearAlgebra.eigvals(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, factorisation_scalartype(eigen, t))
    return LinearAlgebra.eigvals!(tcopy; kwargs...)
end

"""
    eig(t::AbstractTensor, (leftind, rightind)::Index2Tuple; kwargs...) -> D, V

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
"""
function eig(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(eig, t), p)
    return eig!(tcopy; kwargs...)
end

"""
    eigh(t::AbstractTensorMap, (leftind, rightind)::Index2Tuple) -> D, V

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
"""
function eigh(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(eigh, t), p)
    return eigh!(tcopy; kwargs...)
end

"""
    isposdef(t::AbstractTensor, (leftind, rightind)::Index2Tuple) -> ::Bool

Test whether a tensor `t` is positive definite as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `isposdef!(t)`. Note that the permuted tensor on
which `isposdef!` is called should have equal domain and codomain, as otherwise it is
meaningless.
"""
function LinearAlgebra.isposdef(t::AbstractTensorMap, (p₁, p₂)::Index2Tuple)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(isposdef, t), p)
    return isposdef!(tcopy)
end

function tsvd(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return tsvd!(tcopy; kwargs...)
end
function leftorth(t::AbstractTensorMap; alg::OFA=QRpos(), kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return leftorth!(tcopy; alg=alg, kwargs...)
end
function rightorth(t::AbstractTensorMap; alg::OFA=LQpos(), kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return rightorth!(tcopy; alg=alg, kwargs...)
end
function leftnull(t::AbstractTensorMap; alg::OFA=QR(), kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return leftnull!(tcopy; alg=alg, kwargs...)
end
function rightnull(t::AbstractTensorMap; alg::OFA=LQ(), kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return rightnull!(tcopy; alg=alg, kwargs...)
end
function LinearAlgebra.eigen(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return eigen!(tcopy; kwargs...)
end
function eig(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return eig!(tcopy; kwargs...)
end
function eigh(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return eigh!(tcopy; kwargs...)
end
function LinearAlgebra.isposdef(t::AbstractTensorMap)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return isposdef!(tcopy)
end

# Orthogonal factorizations (mutation for recycling memory):
# only possible if scalar type is floating point
# only correct if Euclidean inner product
#------------------------------------------------------------------------------------------
const RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

function leftorth!(t::TensorMap{<:RealOrComplexFloat};
                   alg::Union{QR,QRpos,QL,QLpos,SVD,SDD,Polar}=QRpos(),
                   atol::Real=zero(float(real(scalartype(t)))),
                   rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                              eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute QR factorization for each block
    if !isempty(blocks(t))
        generator = Base.Iterators.map(blocks(t)) do (c, b)
            Qc, Rc = MatrixAlgebra.leftorth!(b, alg, atol)
            dims[c] = size(Qc, 2)
            return c => (Qc, Rc)
        end
        QRdata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    V = S(dims)
    if alg isa Polar
        @assert V ≅ domain(t)
        W = domain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    else
        W = ProductSpace(V)
    end

    # construct output tensors
    T = float(scalartype(t))
    Q = similar(t, T, codomain(t) ← W)
    R = similar(t, T, W ← domain(t))
    if !isempty(blocks(t))
        for (c, (Qc, Rc)) in QRdata
            copy!(block(Q, c), Qc)
            copy!(block(R, c), Rc)
        end
    end
    return Q, R
end

function leftnull!(t::TensorMap{<:RealOrComplexFloat};
                   alg::Union{QR,QRpos,SVD,SDD}=QRpos(),
                   atol::Real=zero(float(real(scalartype(t)))),
                   rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                              eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:leftnull!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute QR factorization for each block
    V = codomain(t)
    if !isempty(blocksectors(V))
        generator = Base.Iterators.map(blocksectors(V)) do c
            Nc = MatrixAlgebra.leftnull!(block(t, c), alg, atol)
            dims[c] = size(Nc, 2)
            return c => Nc
        end
        Ndata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    W = S(dims)

    # construct output tensor
    T = float(scalartype(t))
    N = similar(t, T, V ← W)
    if !isempty(blocksectors(V))
        for (c, Nc) in Ndata
            copy!(block(N, c), Nc)
        end
    end
    return N
end

function rightorth!(t::TensorMap{<:RealOrComplexFloat};
                    alg::Union{LQ,LQpos,RQ,RQpos,SVD,SDD,Polar}=LQpos(),
                    atol::Real=zero(float(real(scalartype(t)))),
                    rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                               eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute LQ factorization for each block
    if !isempty(blocks(t))
        generator = Base.Iterators.map(blocks(t)) do (c, b)
            Lc, Qc = MatrixAlgebra.rightorth!(b, alg, atol)
            dims[c] = size(Qc, 1)
            return c => (Lc, Qc)
        end
        LQdata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    V = S(dims)
    if alg isa Polar
        @assert V ≅ codomain(t)
        W = codomain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    else
        W = ProductSpace(V)
    end

    # construct output tensors
    T = float(scalartype(t))
    L = similar(t, T, codomain(t) ← W)
    Q = similar(t, T, W ← domain(t))
    if !isempty(blocks(t))
        for (c, (Lc, Qc)) in LQdata
            copy!(block(L, c), Lc)
            copy!(block(Q, c), Qc)
        end
    end
    return L, Q
end

function rightnull!(t::TensorMap{<:RealOrComplexFloat};
                    alg::Union{LQ,LQpos,SVD,SDD}=LQpos(),
                    atol::Real=zero(float(real(scalartype(t)))),
                    rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                               eps(real(float(one(scalartype(t))))) * iszero(atol))
    InnerProductStyle(t) === EuclideanInnerProduct() ||
        throw_invalid_innerproduct(:rightnull!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute LQ factorization for each block
    V = domain(t)
    if !isempty(blocksectors(V))
        generator = Base.Iterators.map(blocksectors(V)) do c
            Nc = MatrixAlgebra.rightnull!(block(t, c), alg, atol)
            dims[c] = size(Nc, 1)
            return c => Nc
        end
        Ndata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    W = S(dims)

    # construct output tensor
    T = float(scalartype(t))
    N = similar(t, T, W ← V)
    if !isempty(blocksectors(V))
        for (c, Nc) in Ndata
            copy!(block(N, c), Nc)
        end
    end
    return N
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

#------------------------------#
# Singular value decomposition #
#------------------------------#
function LinearAlgebra.svdvals!(t::TensorMap{<:RealOrComplexFloat})
    return SectorDict(c => LinearAlgebra.svdvals!(b) for (c, b) in blocks(t))
end
LinearAlgebra.svdvals!(t::AdjointTensorMap) = svdvals!(adjoint(t))

function tsvd!(t::TensorMap{<:RealOrComplexFloat};
               trunc=NoTruncation(), p::Real=2, alg=SDD())
    return _tsvd!(t, alg, trunc, p)
end
function tsvd!(t::AdjointTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    u, s, vt, err = tsvd!(adjoint(t); trunc=trunc, p=p, alg=alg)
    return adjoint(vt), adjoint(s), adjoint(u), err
end

# implementation dispatches on algorithm
function _tsvd!(t::TensorMap{<:RealOrComplexFloat}, alg::Union{SVD,SDD},
                trunc::TruncationScheme, p::Real=2)
    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return _empty_svdtensors(t)..., truncerr
    end

    # compute SVD factorization for each block
    S = spacetype(t)
    SVDdata, dims = _compute_svddata!(t, alg)
    Σdata = SectorDict(c => Σ for (c, (U, Σ, V)) in SVDdata)
    truncdim = _compute_truncdim(Σdata, trunc, p)
    truncerr = _compute_truncerr(Σdata, truncdim, p)

    # construct output tensors
    U, Σ, V⁺ = _create_svdtensors(t, SVDdata, truncdim)
    return U, Σ, V⁺, truncerr
end

# helper functions
function _compute_svddata!(t::TensorMap, alg::Union{SVD,SDD})
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    dims = SectorDict{I,Int}()
    generator = Base.Iterators.map(blocks(t)) do (c, b)
        U, Σ, V = MatrixAlgebra.svd!(b, alg)
        dims[c] = length(Σ)
        return c => (U, Σ, V)
    end
    SVDdata = SectorDict(generator)
    return SVDdata, dims
end

function _create_svdtensors(t::TensorMap{<:RealOrComplexFloat}, SVDdata, dims)
    T = scalartype(t)
    S = spacetype(t)
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    Σ = DiagonalTensorMap{Tr,S,A}(undef, W)

    U = similar(t, codomain(t) ← W)
    V⁺ = similar(t, W ← domain(t))
    for (c, (Uc, Σc, V⁺c)) in SVDdata
        r = Base.OneTo(dims[c])
        copy!(block(U, c), view(Uc, :, r))
        copy!(block(Σ, c), Diagonal(view(Σc, r)))
        copy!(block(V⁺, c), view(V⁺c, r, :))
    end
    return U, Σ, V⁺
end

function _empty_svdtensors(t::TensorMap{<:RealOrComplexFloat})
    T = scalartype(t)
    S = spacetype(t)
    I = sectortype(t)
    dims = SectorDict{I,Int}()
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    Σ = DiagonalTensorMap{Tr,S,A}(undef, W)

    U = similar(t, codomain(t) ← W)
    V⁺ = similar(t, W ← domain(t))
    return U, Σ, V⁺
end

#--------------------------#
# Eigenvalue decomposition #
#--------------------------#
function LinearAlgebra.eigen!(t::TensorMap{<:RealOrComplexFloat})
    return ishermitian(t) ? eigh!(t) : eig!(t)
end

function LinearAlgebra.eigvals!(t::TensorMap{<:RealOrComplexFloat}; kwargs...)
    return SectorDict(c => complex(LinearAlgebra.eigvals!(b; kwargs...))
                      for (c, b) in blocks(t))
end
function LinearAlgebra.eigvals!(t::AdjointTensorMap{<:RealOrComplexFloat}; kwargs...)
    return SectorDict(c => conj!(complex(LinearAlgebra.eigvals!(b; kwargs...)))
                      for (c, b) in blocks(t))
end

function eigh!(t::TensorMap{<:RealOrComplexFloat})
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:eigh!)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`eigh!` requires domain and codomain to be the same"))

    T = scalartype(t)
    I = sectortype(t)
    S = spacetype(t)
    dims = SectorDict{I,Int}(c => size(b, 1) for (c, b) in blocks(t))
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    D = DiagonalTensorMap{Tr,S,A}(undef, W)
    V = similar(t, domain(t) ← W)
    for (c, b) in blocks(t)
        values, vectors = MatrixAlgebra.eigh!(b)
        copy!(block(D, c), Diagonal(values))
        copy!(block(V, c), vectors)
    end
    return D, V
end

function eig!(t::TensorMap{<:RealOrComplexFloat}; kwargs...)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`eig!` requires domain and codomain to be the same"))

    T = scalartype(t)
    I = sectortype(t)
    S = spacetype(t)
    dims = SectorDict{I,Int}(c => size(b, 1) for (c, b) in blocks(t))
    W = S(dims)

    Tc = complex(T)
    A = similarstoragetype(t, Tc)
    D = DiagonalTensorMap{Tc,S,A}(undef, W)
    V = similar(t, Tc, domain(t) ← W)
    for (c, b) in blocks(t)
        values, vectors = MatrixAlgebra.eig!(b; kwargs...)
        copy!(block(D, c), Diagonal(values))
        copy!(block(V, c), vectors)
    end
    return D, V
end

#--------------------------------------------------#
# Checks for hermiticity and positive definiteness #
#--------------------------------------------------#
function LinearAlgebra.ishermitian(t::TensorMap)
    domain(t) == codomain(t) || return false
    InnerProductStyle(t) === EuclideanInnerProduct() || return false # hermiticity only defined for euclidean
    for (c, b) in blocks(t)
        ishermitian(b) || return false
    end
    return true
end

function LinearAlgebra.isposdef!(t::TensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("`isposdef` requires domain and codomain to be the same"))
    InnerProductStyle(spacetype(t)) === EuclideanInnerProduct() || return false
    for (c, b) in blocks(t)
        isposdef!(b) || return false
    end
    return true
end
