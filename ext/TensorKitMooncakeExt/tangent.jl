function Mooncake.arrayify(A_dA::CoDual{<:TensorMap})
    A = Mooncake.primal(A_dA)
    dA_fw = Mooncake.tangent(A_dA)
    data = dA_fw.data.data
    dA = typeof(A)(data, A.space)
    return A, dA
end
