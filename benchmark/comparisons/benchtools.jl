function distributeZ2(D; p = 0.5)
    D0 = ceil(Int, p * D)
    D1 = D - D0
    return [(0, D0), (1, D1)]
end

distributeU1(D; p = 0.25) = distributeU1_poisson(D; p = p)

function distributeU1_exponential(D; p = 0.25)
    λ = (1 - p) / (1 + p)
    D0 = ceil(Int, p * D)
    if isodd(D - D0)
        D0 = D0 == 1 ? 2 : D0 - 1
    end
    sectors = [(0, D0)]
    Drem = D - D0
    n = 1
    while Drem > 0
        pn = p * λ^n
        Dn = ceil(Int, pn * D)
        sectors = push!(sectors, (n, Dn), (-n, Dn))
        Drem -= 2 * Dn
        n += 1
    end
    return sort!(sectors; by = first)
end

function distributeU1_poisson(D; p = 0.25)
    λ = log((1 / p + 1) / 2)
    D0 = ceil(Int, p * D)
    if isodd(D - D0)
        D0 = D0 == 1 ? 2 : D0 - 1
    end
    sectors = [(0, D0)]
    Drem = D - D0
    n = 1
    while Drem > 0
        pn = p * λ^n / factorial(n)
        Dn = ceil(Int, pn * D)
        sectors = push!(sectors, (n, Dn), (-n, Dn))
        Drem -= 2 * Dn
        n += 1
    end
    return sort!(sectors; by = first)
end

module Timers
    struct Timer{F, D <: Tuple}
        f::F
        argsref::Base.RefValue{D}
        Timer(f, args...) = new{typeof(f), typeof(args)}(f, Ref(args))
    end

    @noinline donothing(arg) = arg
    function (t::Timer)(; inner = 1, outer = 1)
        args = t.argsref[]
        f = t.f
        f(args...) # run once to compile
        times = zeros(Float64, (outer,))
        gctimes = zeros(Float64, (outer,))
        @inbounds for i in 1:outer
            if inner == 1
                gcstart = Base.gc_num()
                start = Base.time_ns()
                donothing(f(args...))
                stop = Base.time_ns()
                gcstop = Base.gc_num()
            else
                gcstart = Base.gc_num()
                start = Base.time_ns()
                for _ in 1:inner
                    donothing(f(args...))
                end
                stop = Base.time_ns()
                gcstop = Base.gc_num()
            end
            times[i] = (stop - start) / (1.0e9) / inner
            gctimes[i] = Base.GC_Diff(gcstop, gcstart).total_time / 1.0e9 / inner
        end
        return times, gctimes
    end
end
