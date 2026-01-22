function MatrixAlgebraKit._ind_intersect(A::AbstractVector{Bool}, B::CuVector{Int})
    return MatrixAlgebraKit._ind_intersect(findall(A), B)
end

# TODO: intersect doesn't work on GPU
MatrixAlgebraKit._ind_intersect(A::CuVector{Int}, B::CuVector{Int}) =
    MatrixAlgebraKit._ind_intersect(collect(A), collect(B))
