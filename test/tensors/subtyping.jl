using Test
using TensorKit
using TensorKit: AdjointTensorMap

@testset "Tensor map subtyping" begin
    for TT in (TensorMap, DiagonalTensorMap, AdjointTensorMap, BraidingTensor)
        @test TT <: AbstractTensorMap
        for T in (Float64, ComplexF64)
            @test TT{T} <: AbstractTensorMap{T}
        end
    end
end
