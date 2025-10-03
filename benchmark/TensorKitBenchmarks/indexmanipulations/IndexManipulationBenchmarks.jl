module IndexManipulationBenchmarks

include(joinpath(@__DIR__, "..", "utils", "BenchUtils.jl"))

using .BenchUtils
using BenchmarkTools
using TensorKit
using TOML

const SUITE = BenchmarkGroup()
const all_parameters = TOML.parsefile(joinpath(@__DIR__, "benchparams.toml"))

# permute!
# --------
function init_permute_tensors(T, W, p)
    C = randn(T, permute(W, p))
    A = randn(T, W)
    return C, A
end
function benchmark_permute!(benchgroup, params::Dict)
    haskey(benchgroup, "permute") || addgroup!(benchgroup, "permute")
    bench = benchgroup["permute"]
    for kwargs in expand_kwargs(params)
        benchmark_permute!(bench; kwargs...)
    end
    return nothing
end
function benchmark_permute!(bench; sigmas = nothing, T = "Float64", I = "Trivial", dims, p)
    T_ = parse_type(T)
    I_ = parse_type(I)

    p_ = (Tuple(p[1]), Tuple(p[2]))
    Vs = generate_space.(I_, dims, sigmas)

    codomain = mapreduce(Base.Fix1(getindex, Vs), ⊗, p_[1]; init = one(eltype(Vs)))
    domain = mapreduce(Base.Fix1(getindex, Vs), ⊗, p_[2]; init = one(eltype(Vs)))
    init() = init_permute_tensors(T_, codomain ← domain, p_)

    bench[T, I, dims, sigmas, p] = @benchmarkable permute!(C, A, $p_) setup = ((C, A) = $init())
    return nothing
end

if haskey(all_parameters, "permute")
    g = addgroup!(SUITE, "permute")
    for params in all_parameters["permute"]
        benchmark_permute!(g, params)
    end
end

# transpose!
# ----------

# TODO

end
