function TensorKit._copyto!(A::StridedView{TA, 1, <:CuArray{TA}}, B::StridedView{TB, 2, <:CuArray{TB}}) where {TA, TB}
    length(A) == length(B) || throw(DimensionMismatch(lazy"length of A ($(length(A))) does not match length of B ($(length(B))"))

    Adata = parent(A)
    Astr = stride(A, 1)
    IA = A.offset

    Bdata = parent(B)
    Bstr = strides(B)

    IB_1 = B.offset
    # build index arrays
    IAs = Int[]
    IBs = Int[]
    @inbounds for _ in axes(B, 2)
        IB = IB_1
        for _ in axes(B, 1)
            IA += Astr
            append!(IAs, IA)
            IB += Bstr[1]
            append!(IBs, IB)
        end
        IB_1 += Bstr[2]
    end
    Adata[IAs] .= Bdata[IBs]

    return A
end
