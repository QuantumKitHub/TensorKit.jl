# Run once per fresh process, with TensorKit's precompile workloads disabled,
# against the baseline and prototype using identical dependency versions:
# jld --project=<env> --name=<fresh-name> --no-revise --idle-timeout=2h \
#     run benchmark/factorization_latency.jl
#
# The prepared mode isolates execution of the tensor-level factorization kernel;
# public mode includes copying, algorithm selection and output initialization.
# Set TK_LATENCY_MODE=public to measure the latter in a separate fresh process.
using TensorKit
using Random
using Printf
import MatrixAlgebraKit as MAK

Random.seed!(494)
const latency_mode = get(ENV, "TK_LATENCY_MODE", "prepared")
latency_mode in ("prepared", "public") || error("Expected prepared or public mode")

# Avoid specializing the timing harness on the tensors under measurement.
@noinline function measure_call(@nospecialize(f), @nospecialize(args))
    return @timed f(args...)
end
measure_call(identity, (nothing,))

# Internal diagnostic only: count QR traversal specializations, without forcing
# inference. The prototype should share these across sectors and tensor arities.
function qr_loop_specializations()
    f = isdefined(TensorKit, :foreachalignedblock) ? TensorKit.foreachalignedblock : TensorKit.foreachblock
    count = 0
    for method in methods(f), mi in Base.specializations(method)
        isnothing(mi) && continue
        isconcretetype(mi.specTypes) || continue
        callback = Base.unwrap_unionall(mi.specTypes).parameters[2]
        # QR is the only Householder operation in this workload. The generated
        # callback names are anonymous, so identify it by its captured algorithm.
        if callback isa DataType && isconcretetype(callback) &&
                !isempty(callback.parameters) && first(callback.parameters) <: MAK.Householder
            count += 1
        end
    end
    return count
end

spaces = (
    ("Trivial", ComplexSpace(4)),
    ("Z2", Z2Space(0 => 3, 1 => 2)),
    ("U1", U1Space(0 => 3, 1 => 2)),
    ("SU2", SU2Space(0 => 3, 1 // 2 => 2)),
    ("Fibonacci", Vect[FibonacciAnyon](:I => 3, :τ => 2)),
)
functions = (qr_compact, svd_compact, svd_vals, eigh_full, project_hermitian, exponential)
functions! = (qr_compact!, svd_compact!, svd_vals!, eigh_full!, project_hermitian!, exponential!)

println("Julia: ", VERSION, "; TensorKit: ", pathof(TensorKit), "; mode: ", latency_mode)
println("sector,type,first_seconds,compile_seconds,recompile_seconds,warm_seconds,qr_loop_specializations")
for (label, V) in spaces, T in (Float64, ComplexF64)
    first_seconds = compile_seconds = recompile_seconds = warm_seconds = 0.0
    for (f, f!) in zip(functions, functions!)
        t = randn(T, V ← V)
        # Build a Hermitian input without calling a factorization or projection.
        for (_, b) in blocks(t)
            b .= b + b'
        end
        if latency_mode == "prepared"
            alg = MAK.select_algorithm(f!, t, nothing)
            out = MAK.initialize_output(f!, t, alg)
            twarm = copy(t)
            outwarm = MAK.initialize_output(f!, twarm, alg)
            result = measure_call(f!, (t, out, alg))
            warm = measure_call(f!, (twarm, outwarm, alg))
        else
            result = measure_call(f, (t,))
            warm = measure_call(f, (t,))
        end
        first_seconds += result.time
        compile_seconds += result.compile_time
        recompile_seconds += result.recompile_time
        warm_seconds += warm.time
    end
    @printf("%s,%s,%.6f,%.6f,%.6f,%.6f,%d\n", label, T, first_seconds, compile_seconds, recompile_seconds, warm_seconds, qr_loop_specializations())
end

if isdefined(TensorKit, :foreachalignedblock)
    # Base.specializations is an internal diagnostic; do not turn these counts
    # into tests or rely on them as a stable API.
    sigs = [s.specTypes for s in Base.specializations(which(TensorKit.foreachalignedblock, Tuple{Function, Tuple})) if s !== nothing && isconcretetype(Base.unwrap_unionall(s.specTypes).parameters[2])]
    println("Positional loop specializations: ", length(sigs))
end
