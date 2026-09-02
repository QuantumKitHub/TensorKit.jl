using Test, TestExtras
using TensorKit
using TensorOperations
using TimerOutputs
using TimerOutputs: @timeit

@testset "timer API (no-op by default)" begin
    @test TensorKit.timer() isa TimerOutput
    @test TensorKit.timeit_debug_enabled() === false
    @test TensorKit.timers_enabled() === false

    V = SU2Space(0 => 2, 1 // 2 => 2, 1 => 1)
    t = rand(V ⊗ V ← V)
    TensorKit.reset_timers!()
    permute(t, ((2, 1), (3,)))
    # nothing is recorded while timers are disabled
    @test isempty(TensorKit.timer().root.children)
end

@testset "timer_summary aggregation" begin
    to = TimerOutput()
    @timeit to "permute!" begin
        @timeit to "bookkeeping: cache treebraider" begin
            @timeit to "symmetry: compute treebraider" sleep(0.01)
        end
        @timeit to "dense: pack" sleep(0.01)
    end
    summary = TensorKit.timer_summary(nothing; to)

    @test keys(summary) == Set(TensorKit.TIMER_CATEGORIES)
    @test summary[:bookkeeping].ncalls == 1
    @test summary[:symmetry].ncalls == 1
    @test summary[:dense].ncalls == 1
    @test summary[:alloc] == (time = 0, allocated = 0, ncalls = 0)
    @test summary[:symmetry].time > 0
    @test summary[:dense].time > 0
    # the unprefixed top-level section contributes its exclusive time to :other
    @test summary[:other].ncalls == 1

    # exclusive-time attribution: category totals sum to the total measured time
    total = sum(x -> x.time, values(summary))
    @test total ≈ TimerOutputs.time(only(to.root.children)) rtol = 0.01

    # printing form does not error
    @test sprint(io -> TensorKit.timer_summary(io; to)) isa String
end

@testset "smoke test with timers enabled" begin
    TensorKit.enable_timers!()
    try
        @test TensorKit.timeit_debug_enabled() === true
        TensorKit.reset_timers!()
        empty_globalcaches!() # ensure the (symmetry) construction work is not cached

        V = SU2Space(0 => 2, 1 // 2 => 2)
        t = rand(V ⊗ V ← V ⊗ V)
        permute(t, ((1, 3), (2, 4)))
        @tensor t2[a; b] := t[a c; b c]
        @tensor t3[a b; c d] := t[a x; c y] * t[y b; x d]
        svd_compact(t)

        names = [child.name for child in TensorKit.timer().root.children]
        @test "permute!" in names
        @test "contract!" in names
        # also verifies that `enable_timers!` reached the Factorizations submodule
        @test "svd_compact!" in names

        summary = TensorKit.timer_summary(nothing)
        @test summary[:dense].ncalls > 0
        @test summary[:symmetry].ncalls > 0
        @test summary[:bookkeeping].ncalls > 0
        @test summary[:alloc].ncalls > 0

        # the full printed table renders without error
        @test sprint(TensorKit.print_timers) isa String
    finally
        TensorKit.disable_timers!()
    end
    @test TensorKit.timeit_debug_enabled() === false
end
