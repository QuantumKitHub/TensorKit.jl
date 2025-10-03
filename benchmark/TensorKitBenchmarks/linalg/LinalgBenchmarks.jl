module LinalgBenchmarks

include(joinpath(@__DIR__, "..", "utils", "BenchUtils.jl"))

using .BenchUtils
using BenchmarkTools
using TensorKit
using TOML

const SUITE = BenchmarkGroup()
const all_parameters = TOML.parsefile(joinpath(@__DIR__, "benchparams.toml"))

# mul!
# ----
function init_mul_tensors(T, V)
    A = randn(T, V[1] ← V[2])
    B = randn(T, V[2] ← V[3])
    C = randn(T, V[1] ← V[3])
    return A, B, C
end

function benchmark_mul!(benchgroup, params::Dict)
    haskey(benchgroup, "mul") || addgroup!(benchgroup, "mul")
    bench = benchgroup["mul"]
    for kwargs in expand_kwargs(params)
        benchmark_mul!(bench; kwargs...)
    end
    return nothing
end

function benchmark_mul!(bench; sigmas = nothing, T = "Float64", I = "Trivial", dims)
    T_ = parse_type(T)
    I_ = parse_type(I)

    Vs = generate_space.(I_, dims, sigmas)
    init() = init_mul_tensors(T_, Vs)

    bench[T, I, dims, sigmas] = @benchmarkable mul!(C, A, B) setup = ((A, B, C) = $init())

    return nothing
end

if haskey(all_parameters, "mul")
    g = addgroup!(SUITE, "mul")
    for params in all_parameters["mul"]
        benchmark_mul!(g, params)
    end
end

# svd!
# ----
function init_svd_tensor(T, V)
    A = randn(T, V[1] ← V[2])
    return A
end

function benchmark_svd!(benchgroup, params::Dict)
    haskey(benchgroup, "svd") || addgroup!(benchgroup, "svd")
    bench = benchgroup["svd"]
    for kwargs in expand_kwargs(params)
        benchmark_svd!(bench; kwargs...)
    end
    return nothing
end
function benchmark_svd!(bench; sigmas = nothing, T = "Float64", I = "Trivial", dims)
    T_ = parse_type(T)
    I_ = parse_type(I)
    Vs = generate_space.(I_, dims, sigmas)
    init() = init_svd_tensor(T_, Vs)
    bench[T, I, dims, sigmas] = @benchmarkable tsvd!(A) setup = (A = $init())
    return nothing
end

if haskey(all_parameters, "svd")
    g = addgroup!(SUITE, "svd")
    for params in all_parameters["svd"]
        benchmark_svd!(g, params)
    end
end

end
