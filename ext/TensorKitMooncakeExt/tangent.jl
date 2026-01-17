function Mooncake.arrayify(A_dA::CoDual{<:TensorMap})
    A = Mooncake.primal(A_dA)
    dA_fw = Mooncake.tangent(A_dA)
    data = dA_fw.data.data
    dA = typeof(A)(data, A.space)
    return A, dA
end

function Mooncake.arrayify(Aᴴ_ΔAᴴ::CoDual{<:TensorKit.AdjointTensorMap})
    Aᴴ = Mooncake.primal(Aᴴ_ΔAᴴ)
    ΔAᴴ = Mooncake.tangent(Aᴴ_ΔAᴴ)
    A_ΔA = CoDual(Aᴴ', ΔAᴴ.data.parent)
    A, ΔA = arrayify(A_ΔA)
    return A', ΔA'
end
