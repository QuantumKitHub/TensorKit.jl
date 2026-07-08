using Test, TestExtras
using TensorKit
using TensorKit: type_repr


spacelist = default_spacelist(fast_tests)

for V in spacelist
    I = sectortype(first(V))
    Istr = type_repr(I)
    println("---------------------------------------")
    println("Tensor constructions with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensor constructions with symmetry: $Istr" verbose = true begin
        TensorKitTestSuite.run_testsuite(:tensors, "basic properties", V)
        TensorKitTestSuite.run_testsuite(:tensors, "dict conversion", V)
        TensorKitTestSuite.run_testsuite(:tensors, "array conversion", V)
        TensorKitTestSuite.run_testsuite(:tensors, "real and imaginary parts", V)
        TensorKitTestSuite.run_testsuite(:tensors, "tensor conversion", V)
    end
    TensorKit.empty_globalcaches!()
end

@timedtestset "show tensors" begin
    for V in (ℂ^2, Z2Space(0 => 2, 1 => 2), SU2Space(0 => 2, 1 => 2))
        t1 = ones(Float32, V ⊗ V, V)
        t2 = randn(ComplexF64, V ⊗ V ⊗ V)
        t3 = randn(Float64, zero(V), zero(V))
        # test unlimited output
        for t in (t1, t2, t1', t2', t3)
            output = IOBuffer()
            summary(output, t)
            print(output, ":\n codomain: ")
            show(output, MIME("text/plain"), codomain(t))
            print(output, "\n domain: ")
            show(output, MIME("text/plain"), domain(t))
            print(output, "\n blocks: \n")
            first = true
            for (c, b) in blocks(t)
                first || print(output, "\n\n")
                print(output, " * ")
                show(output, MIME("text/plain"), c)
                print(output, " => ")
                show(output, MIME("text/plain"), b)
                first = false
            end
            outputstr = String(take!(output))
            @test outputstr == sprint(show, MIME("text/plain"), t)
        end

        # test limited output with a single block
        t = randn(Float64, V ⊗ V, V)' # we know there is a single space in the codomain, so that blocks have 2 rows
        output = IOBuffer()
        summary(output, t)
        print(output, ":\n codomain: ")
        show(output, MIME("text/plain"), codomain(t))
        print(output, "\n domain: ")
        show(output, MIME("text/plain"), domain(t))
        print(output, "\n blocks: \n")
        c = unit(sectortype(t))
        b = block(t, c)
        print(output, " * ")
        show(output, MIME("text/plain"), c)
        print(output, " => ")
        show(output, MIME("text/plain"), b)
        if length(blocks(t)) > 1
            print(output, "\n\n *   …   [output of 1 more block(s) truncated]")
        end
        outputstr = String(take!(output))
        @test outputstr == sprint(show, MIME("text/plain"), t; context = (:limit => true, :displaysize => (12, 100)))
    end
end
