using Test, TestExtras
using TensorKit
using TensorKit: type_repr


spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("Tensor contractions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensor contractions with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.test_tensors_contractions(V)
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "Deligne tensor product: test via conversion" begin
    using .TestSetup: Vtr, VRepℤ₂, VRepSU₂, VRepA4
    @testset for Vlist1 in (Vtr, VRepSU₂), Vlist2 in (VRepℤ₂, VRepA4)
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = rand(T, V2 ⊗ V3, (V4 ⊗ V5)')
            t2 = rand(T, W2, (W3 ⊗ W4)')
            t = @constinferred (t1 ⊠ t2)
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            At = convert(Array, t)
            @test reshape(At, (d1, d2, d3, d4)) ≈
                reshape(convert(Array, t1), (d1, 1, d3, 1)) .*
                reshape(convert(Array, t2), (1, d2, 1, d4))
        end
    end
end
