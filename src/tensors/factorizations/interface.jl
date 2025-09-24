@doc """
    isposdef(t::AbstractTensor, [(leftind, rightind)::Index2Tuple]) -> ::Bool

Test whether a tensor `t` is positive definite as linear map from `rightind` to `leftind`.

If `leftind` and `rightind` are not specified, the current partition of left and right
indices of `t` is used. In that case, less memory is allocated if one allows the data in
`t` to be destroyed/overwritten, by using `isposdef!(t)`. Note that the permuted tensor on
which `isposdef!` is called should have equal domain and codomain, as otherwise it is
meaningless.
""" isposdef(::AbstractTensorMap), isposdef!(::AbstractTensorMap)

function LinearAlgebra.eigvals(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, factorisation_scalartype(eigen, t))
    return LinearAlgebra.eigvals!(tcopy; kwargs...)
end

function LinearAlgebra.svdvals(t::AbstractTensorMap)
    tcopy = copy_oftype(t, factorisation_scalartype(tsvd, t))
    return LinearAlgebra.svdvals!(tcopy)
end
